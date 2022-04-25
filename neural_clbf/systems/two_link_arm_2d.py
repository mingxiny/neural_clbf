"""Define a dynamical system for a 2D 2-link robot arm"""
from typing import Tuple, List, Optional, Callable

import torch
import numpy as np
import tqdm

import os
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, Simulator, 
                         LeafSystem, BasicVector, ConnectMeshcatVisualizer, AbstractValue,
                         Solve, InverseDynamicsController, QueryObject, JacobianWrtVariable)
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.systems.primitives import LogVectorOutput
from pydrake.geometry import Role
from neural_clbf.systems.drake_utils import *

from .control_affine_system import ControlAffineSystem
from .utils import grav, Scenario


class TwoLinkArm2D(ControlAffineSystem):
    """
    Represents a planar 2-link robot arm.

    The system has state

        x = [theta1, theta2, theta1_dot, theta2_dot, o, do/dq]

    representing the angles and velocities of the robot arm, and it
    has control inputs

        u = [u1, u2]

    representing the thrust at the right and left rotor.
    """

    # Number of states and controls
    Q_DIMS = 2
    N_DIMS = int(2 * Q_DIMS)
    N_CONTROLS = 2

    O_DIMS = 1

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        dis_threshold:float = 0.01
    ):
        """
        Initialize the quadrotor.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["m", "I", "r"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(nominal_params, dt, controller_dt, use_linearized_controller=False)
        
        self.dt = dt
        # Minimum distance threshold for avoiding collission
        self.dis_threshold = dis_threshold

        # Set up Drake diagram for simulation
        builder = DiagramBuilder()
        # TODO: @lujieyang Try 0 for continuous simulation
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)
        inspector = scene_graph.model_inspector()
        parser = Parser(plant)
        self.model_dir = os.path.join(os.getcwd(),"neural_clbf/systems/models")
        parser.AddModelFromFile(os.path.join(self.model_dir, "2d_robot.sdf"), "2d_robot")
        parser.AddModelFromFile(os.path.join(self.model_dir, "obstacle.sdf"), "obstacle")
        plant.Finalize()
        self.plant = plant

        # Collect collision geometries
        self.collision_group = {"obstacle":["box_0", "box_1"], "2d_robot":["link_1", "link_2"]}
        self.geometries = {"obstacle":[], "2d_robot":[]}
        for group in self.collision_group.keys():
            for body_name in self.collision_group[group]:
                body = plant.GetBodyByName(body_name)
                frame_id = plant.GetBodyFrameIdIfExists(body.index())
                geometry = inspector.GetGeometryIdByName(frame_id, role=Role.kProximity, name=group +"::"+ body_name)
                self.geometries[group].append(geometry)

        # Set up InverseDynamics Controller to drive the arms to the goal state without considering obstacles
        kp = np.array([10, 10])
        ki = np.array([0.1, 0.1])
        kd = np.array([5, 5])
        controller = builder.AddSystem(InverseDynamicsController(plant, kp, ki, kd, has_reference_acceleration=False))
        builder.Connect(controller.get_output_port(),
                        plant.get_actuation_input_port())
        builder.Connect(plant.get_state_output_port(), controller.get_input_port(0))
        self.controller = controller

        # Set up observer: scalar as shortest distance from all obstacles to all robot arms
        self.inspector = inspector
        observer = builder.AddSystem(Observer(plant, inspector, self.geometries))
        builder.Connect(scene_graph.get_query_output_port(), observer.get_input_port())

        # Set up meshcat visualization
        proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])
        viz = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url)
        self.viz = viz
        self.vis = viz.vis
        set_orthographic_camera_xy(self.vis)

        # Data logging
        self.state_logger = LogVectorOutput(plant.get_state_output_port(), builder, dt)
        self.controller_logger = LogVectorOutput(controller.get_output_port(), builder, dt)
        self.obs_logger = LogVectorOutput(observer.get_output_port(), builder, dt)

        # Set up simulator
        self.diagram = builder.Build()
        self.simulator = Simulator(self.diagram)
        self.context = self.simulator.get_mutable_context()
        self.controller_context = controller.GetMyMutableContextFromRoot(self.context)
        self.plant_context = plant.GetMyMutableContextFromRoot(self.context)
        
        self.simulator.set_target_realtime_rate(0.5)
        self.query_object = plant.get_geometry_query_input_port().Eval(self.plant_context)

        # Set goal
        self.ee_body = self.plant.GetBodyByName("ee")
        self.ee_goal_pos = np.array([-0.06, 0.18, 0])  # Drake returns pos in 3D
        add_goal_meshcat(self.vis, self.ee_goal_pos)
        goal_pos = self.inverse_kinematics(self.ee_goal_pos)
        self.goal_state = np.concatenate((goal_pos, np.zeros(2)))
        self.desired_state_port = controller.get_input_port_desired_state().FixValue(self.controller_context, self.goal_state)

        self.collision_state = []
        for obstacle_name in self.collision_group["obstacle"]:
            body = plant.GetBodyByName(obstacle_name)
            obstacle_pose = self.plant.EvalBodyPoseInWorld(self.plant_context, body)
            self.collision_state.append(self.inverse_kinematics(obstacle_pose.translation()))

        # self.compute_linearized_controller(None)

    @property
    def n_dims(self) -> int:
        return TwoLinkArm2D.N_DIMS

    @property
    def q_dims(self) -> int:
        return TwoLinkArm2D.Q_DIMS

    @property
    def o_dims(self) -> int:
        return TwoLinkArm2D.O_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [] #[0, 1]

    @property
    def n_controls(self) -> int:
        return TwoLinkArm2D.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[:self.q_dims] = torch.tensor(self.plant.GetPositionUpperLimits())
        upper_limit[self.q_dims:] = torch.tensor(self.plant.GetVelocityUpperLimits())

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.Tensor([320, 320])
        # upper_limit = self.plant.GetEffortUpperLimits()
        lower_limit = -upper_limit

        return (upper_limit, lower_limit)
    
    def validate_params(self, params: Scenario) -> bool:
        return True

    def u_nominal(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        u_nominal = torch.zeros([batch_size, self.n_controls])

        for i in range(batch_size):
            self.plant.SetPositions(self.plant_context, x[i, :self.q_dims])
            self.plant.SetVelocities(self.plant_context, x[i, self.q_dims:self.n_dims])
            output = self.controller.AllocateOutput()
            self.controller.CalcOutput(self.controller_context, output)
            u_nominal[i] = torch.tensor(output.get_vector_data(0).value()).type_as(x)
        
        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(self.n_controls):
            u_nominal[:, dim_idx] = torch.clamp(
                u_nominal[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        return u_nominal

    def compute_B_matrix(self, scenario: Optional[Scenario]) -> np.ndarray:
        """Compute the linearized continuous-time state-state derivative transfer matrix
        about the goal point"""
        B = np.zeros([self.n_dims, self.n_controls])
        self.plant.SetPositions(self.plant_context, self.goal_point[:self.q_dims])
        self.plant.SetVelocities(self.plant_context, self.goal_point[self.q_dims:self.n_dims])
        M = self.plant.CalcMassMatrix(self.plant_context)
        B[self.n_controls:, :] = np.linalg.inv(M)

        return B

    def inverse_kinematics(self, ee_pos):
        ee_frame = self.ee_body.body_frame()
        ref_config = np.array([-0.1, np.pi/4])

        ik = InverseKinematics(self.plant, self.plant_context)
        ik.prog().AddBoundingBoxConstraint(self.plant.GetPositionLowerLimits(), self.plant.GetPositionUpperLimits(), ik.q())
        ik.prog().SetInitialGuess(ik.q(), ref_config)
        ik.prog().AddQuadraticCost((ik.q() - ref_config).dot(ik.q() - ref_config))

        ik.AddPositionConstraint(ee_frame, [0, 0, 0], self.plant.world_frame(), ee_pos, ee_pos)

        result  = Solve(ik.prog())
        return result.GetSolution(ik.q())

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Check if robot links are in collision with any obstacles
        for i in range(x.shape[0]):
            self.plant.SetPositions(self.plant_context, x[i, :self.q_dims])
            safe_mask[i].logical_and_(torch.tensor(self.check_collision_soft()))

        # Also constrain to be within the state limit
        x_max, x_min = self.state_limits
        up_mask = torch.all(x[:, :self.n_dims] <= 0.8 * x_max, dim=1)
        lo_mask = torch.all(x[:, :self.n_dims] >= 0.8 * x_min, dim=1)
        safe_mask.logical_and_(up_mask)
        safe_mask.logical_and_(lo_mask)

        return safe_mask
    
    def check_collision_soft(self):
        for robot in self.geometries["2d_robot"]:
            for obstacle in self.geometries["obstacle"]:
                sd = self.query_object.ComputeSignedDistancePairClosestPoints(robot, obstacle)
                if sd.distance < self.dis_threshold:
                    return False  # unsafe
        return True  # safe

    def check_collision_hard(self):
        return self.query_object.HasCollisions()

    def get_observation(self, min_type="hard"):
        dis = []
        do_dqs = []
        for robot in self.geometries["2d_robot"]:
            for obstacle in self.geometries["obstacle"]:
                sd = self.query_object.ComputeSignedDistancePairClosestPoints(robot, obstacle)
                dis.append(sd.distance)
                do_dqs.append(self.calc_do_dq(sd))

        dis = np.array(dis)
        do_dqs = np.array(do_dqs)
        if min_type == "hard":
            shortest_dis = np.min(dis)
            idx = np.where(dis == shortest_dis)[0]
            if len(idx) == 1:  
                do_dq = do_dqs[idx]  # Robot is closest to only 1 object
            else:
                do_dq_candidate = do_dqs[idx]  # Robot is right in the middle of multiple objects
                q_dot = self.plant.GetVelocities(self.plant_context)
                do_dq = np.ones(self.q_dims) * np.infty
                for do_dq in do_dq_candidate:
                    if (do_dq @q_dot) <=0:
                        break

        return np.concatenate(([shortest_dis], do_dq.squeeze()))

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # Check if robot links are in collision with any obstacles
        for i in range(x.shape[0]):
            self.plant.SetPositions(self.plant_context, x[i, :self.q_dims])
            unsafe_mask[i].logical_or_(torch.tensor(self.check_collision_hard()))

        # Also constrain to be within the state limit
        x_max, x_min = self.state_limits
        limit_mask = (torch.all(x[:, :self.n_dims] >= x_max, dim=1)).logical_or_(torch.all(x[:, :self.n_dims] <= x_min, dim=1))
        unsafe_mask.logical_or_(limit_mask)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set (within 0.2 m of the
        goal).

        args:
            x: a tensor of points in the state space
        """
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Define the goal region as being near the goal
        # near_goal_xy = torch.ones_like(x[:, 0], dtype=torch.bool)
        # Forward kinematics
        # for i in range(x.shape[0]):
        #     self.plant.SetPositions(self.plant_context, x[i, :2])
        #     pos = self.plant.EvalBodyPoseInWorld(self.plant_context, self.ee_body).translation()[:2]  # Drake returns pos in 3D, project into 2D for our system
        #     near_goal_xy[i] = torch.tensor(np.abs(pos - self.ee_goal_pos) <= 0.1, dtype=torch.bool)
        
        near_goal_xy = (x[:, :2] - torch.tensor(self.goal_state[:2]).type_as(x)).norm(dim=1) <= 0.2
        goal_mask.logical_and_(near_goal_xy)
        near_goal_theta_velocity_1 = x[:, 2].abs() <= 0.1
        near_goal_theta_velocity_2 = x[:, 3].abs() <= 0.1
        goal_mask.logical_and_(near_goal_theta_velocity_1)
        goal_mask.logical_and_(near_goal_theta_velocity_2)

        # The goal set has to be a subset of the safe set
        goal_mask.logical_and_(self.safe_mask(x))

        return goal_mask
    
    def sample_goal(self, num_samples: int, eps=0.1) -> torch.Tensor:
        """Sample uniformly from the goal. May return some points that are not in the
        goal, so watch out (only a best-effort sampling)."""

        x = torch.Tensor(num_samples, self.n_dims + self.o_dims + self.q_dims).uniform_(-1.0, 1.0)
        for i in range(num_samples):
            x[i, :self.n_dims] = x[i, :self.n_dims] * eps + self.goal_state
            self.plant.SetPositions(self.plant_context, x[i, :self.q_dims])
            self.plant.SetVelocities(self.plant_context, x[i, self.q_dims:self.n_dims])
            o = self.get_observation()
            x[i, self.n_dims:] = torch.tensor(o).type_as(x)

        return x

    def sample_safe(self, num_samples: int, max_tries: int = 5000) -> torch.Tensor:
        """Sample uniformly from the goal. May return some points that are not in the
        goal, so watch out (only a best-effort sampling)."""
        x = ControlAffineSystem.sample_safe(self, num_samples, max_tries)

        return self.complete_sample_with_observations(x, num_samples)

    def sample_unsafe(self, num_samples: int, max_tries: int = 5000) -> torch.Tensor:
        """Sample uniformly from the goal. May return some points that are not in the
        goal, so watch out (only a best-effort sampling)."""
        x = ControlAffineSystem.sample_unsafe(self, num_samples, max_tries)

        return self.complete_sample_with_observations(x, num_samples)

    def complete_sample_with_observations(self, x, num_samples: int) -> torch.Tensor:
        samples = torch.zeros(num_samples, self.n_dims + self.o_dims + self.q_dims).type_as(x)
        samples[:, :self.n_dims] = x
        for i in range(num_samples):
            self.plant.SetPositions(self.plant_context, x[i, :self.q_dims])
            self.plant.SetVelocities(self.plant_context, x[i, self.q_dims:self.n_dims])
            o = self.get_observation()
            samples[i, self.n_dims:] = torch.tensor(o).type_as(x)
        return samples

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        for i in range(batch_size):
            q_dot = x[i, self.q_dims:self.n_dims]
            self.plant.SetPositions(self.plant_context, x[i, :self.q_dims])
            self.plant.SetVelocities(self.plant_context, x[i, self.q_dims:self.n_dims])
            M = self.plant.CalcMassMatrix(self.plant_context)
            C = self.plant.CalcBiasTerm(self.plant_context)
            tau = self.plant.CalcGravityGeneralizedForces(self.plant_context) 
            f[i, :self.q_dims, :] = q_dot.unsqueeze(1)
            f[i, self.q_dims:self.n_dims, :] = torch.tensor(np.linalg.inv(M) @ (tau - C@q_dot.numpy())).unsqueeze(1)

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))

        for i in range(batch_size):
            self.plant.SetPositions(self.plant_context, x[i, :self.q_dims])
            self.plant.SetVelocities(self.plant_context, x[i, self.q_dims:self.n_dims])
            M = self.plant.CalcMassMatrix(self.plant_context)
            # B = self.plant.MakeActuationMatrix()
            g[i, self.n_controls:, :] = torch.tensor(np.linalg.inv(M))

        return g

    def closed_loop_dynamics(
        self, x: torch.Tensor, u: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        x_next = torch.zeros_like(x)

        for i in range(batch_size):
            q_dot = x[i, self.q_dims:self.n_dims]
            self.plant.SetPositions(self.plant_context, x[i, :self.q_dims])
            self.plant.SetVelocities(self.plant_context, x[i, self.q_dims:self.n_dims])
            M = self.plant.CalcMassMatrix(self.plant_context)
            C = self.plant.CalcBiasTerm(self.plant_context)
            tau = self.plant.CalcGravityGeneralizedForces(self.plant_context) 
            xdot = torch.zeros(self.n_dims)
            xdot[:self.q_dims] = q_dot
            xdot[self.q_dims:self.n_dims] = torch.tensor(np.linalg.inv(M) @ (tau + u[i].numpy() - C@q_dot.numpy())).type_as(x)
            x_next[i, :self.n_dims] = xdot * self.dt + x[i, :self.n_dims]
            self.plant.SetPositions(self.plant_context, x_next[i, :self.q_dims])
            self.plant.SetVelocities(self.plant_context, x_next[i, self.q_dims:self.n_dims])
            o = self.get_observation()
            x_next[i, self.n_dims:] = torch.tensor(o).type_as(x_next)

        return x_next


    @property
    def u_eq(self):
        u_eq = torch.zeros(
                (
                    1,
                    self.n_controls,
                )
            )

        return u_eq
    
    @property
    def goal_point(self):
        return torch.tensor(self.goal_state)
    
    def get_observation_with_state(self, state):
        self.plant.SetPositions(self.plant_context, state[:self.q_dims])
        self.plant.SetVelocities(self.plant_context, state[self.q_dims:self.n_dims])
        return self.get_observation()

    def calc_do_dq(self, sd):
        frame_A_id = self.inspector.GetFrameId(sd.id_A)
        frame_B_id = self.inspector.GetFrameId(sd.id_B)
        frameA = self.plant.GetBodyFromFrameId(frame_A_id).body_frame()
        frameB = self.plant.GetBodyFromFrameId(frame_B_id).body_frame()
        p_ACa = sd.p_ACa
        nhat_BA_W = sd.nhat_BA_W
        Jacobian = self.plant.CalcJacobianTranslationalVelocity(self.plant_context, JacobianWrtVariable.kQDot, frameA, p_ACa, frameB, self.plant.world_frame())
        do_dx = nhat_BA_W.T @ Jacobian
        return do_dx

    def simulate(
        self,
        x_init: torch.Tensor,
        num_steps: int,
        controller: Callable[[torch.Tensor], torch.Tensor],
        controller_period: Optional[float] = None,
        guard: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        params: Optional[Scenario] = None,
    ) -> torch.Tensor:
        """
        Simulate the system for the specified number of steps using the given controller

        args:
            x_init - bs x n_dims tensor of initial conditions
            num_steps - a positive integer
            controller - a mapping from state to control action
            controller_period - the period determining how often the controller is run
                                (in seconds). If none, defaults to self.dt
            guard - a function that takes a bs x n_dims tensor and returns a length bs
                    mask that's True for any trajectories that should be reset to x_init
            params - a dictionary giving the parameter values for the system. If None,
                     default to the nominal parameters used at initialization
        returns
            a bs x num_steps x self.n_dims tensor of simulated trajectories. If an error
            occurs on any trajectory, the simulation of all trajectories will stop and
            the second dimension will be less than num_steps
        """
        # Create a tensor to hold the simulation results
        x_sim = torch.zeros(x_init.shape[0], num_steps, self.n_dims).type_as(x_init)
        obs = torch.zeros(x_init.shape[0], num_steps, self.o_dims + self.q_dims).type_as(x_init)

        # Run the simulation until it's over or an error occurs
        duration = self.dt * num_steps
        for i in tqdm.tqdm(range(x_init.shape[0])):
            self.state_logger.FindMutableLog(self.context).Clear()
            self.controller_logger.FindMutableLog(self.context).Clear()
            self.obs_logger.FindMutableLog(self.context).Clear()
            self.simulator.get_mutable_context().SetTime(0.)
            self.simulator.Initialize()
            self.plant.SetPositions(self.plant_context, x_init[i, :2])
            self.plant.SetVelocities(self.plant_context, x_init[i, 2:4])

            self.simulator.AdvanceTo(duration)
            x = self.state_logger.FindLog(self.context).data().T
            x_sim[i, :, :] = torch.tensor(x[:num_steps]).type_as(x_init)
            o = self.obs_logger.FindLog(self.context).data().T
            obs[i, :, :] = torch.tensor(o[:num_steps]).type_as(x_init)

        return torch.cat((x_sim, obs), 2)


class Observer(LeafSystem):
    def __init__(self, plant, inspector, geometries):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.inspector = inspector
        self.geometries = geometries
        self.min_type = "hard"

        self.query_obj_input_port =  self.DeclareAbstractInputPort(
            "body_pose", AbstractValue.Make(QueryObject()))
        self.obs_output_port = self.DeclareVectorOutputPort(
            "observation", BasicVector(3), self.get_observation)

    def get_observation(self, context, output):
        query_object = self.query_obj_input_port.Eval(context)
        dis = []
        do_dqs = []
        for robot in self.geometries["2d_robot"]:
            for obstacle in self.geometries["obstacle"]:
                sd = query_object.ComputeSignedDistancePairClosestPoints(robot, obstacle)
                dis.append(sd.distance)
                do_dqs.append(self.calc_do_dq(sd))

        dis = np.array(dis)
        do_dqs = np.array(do_dqs)
        if self.min_type == "hard":
            shortest_dis = np.min(dis)
            idx = np.where(dis == shortest_dis)[0]
            if len(idx) == 1:  
                do_dq = do_dqs[idx]  # Robot is closest to only 1 object
            else:
                do_dq_candidate = do_dqs[idx]  # Robot is right in the middle of multiple objects
                q_dot = self.plant.GetVelocities(self.plant_context)
                do_dq = np.ones(self.q_dims) * np.infty
                for do_dq in do_dq_candidate:
                    if (do_dq @q_dot) <=0:
                        break
        
        y = output.get_mutable_value()
        y[:] = np.concatenate(([shortest_dis], do_dq.squeeze()))
    
    def calc_do_dq(self, sd):
        frame_A_id = self.inspector.GetFrameId(sd.id_A)
        frame_B_id = self.inspector.GetFrameId(sd.id_B)
        frameA = self.plant.GetBodyFromFrameId(frame_A_id).body_frame()
        frameB = self.plant.GetBodyFromFrameId(frame_B_id).body_frame()
        p_ACa = sd.p_ACa
        nhat_BA_W = sd.nhat_BA_W
        Jacobian = self.plant.CalcJacobianTranslationalVelocity(self.plant_context, JacobianWrtVariable.kQDot, frameA, p_ACa, frameB, self.plant.world_frame())
        do_dq = nhat_BA_W.T @ Jacobian
        return do_dq

    