import numpy as np
import os
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, Simulator, 
                         LeafSystem, BasicVector, ConnectMeshcatVisualizer, QueryObject,
                         Solve, AbstractValue, InverseDynamicsController, JacobianWrtVariable)
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.geometry import Role
from pydrake.systems.primitives import LogVectorOutput
import meshcat


def set_orthographic_camera_xy(vis: meshcat.Visualizer) -> None:
    # use orthographic camera, show XY plane.
    camera = meshcat.geometry.OrthographicCamera(
        left=-1, right=0.5, bottom=-0.5, top=1, near=-1000, far=1000)
    vis['/Cameras/default/rotated'].set_object(camera)
    vis['/Cameras/default/rotated/<object>'].set_property(
        "position", [0, 0, 0])
    vis['/Cameras/default'].set_transform(
        meshcat.transformations.translation_matrix([0, 0, 1]))

def add_goal_meshcat(vis: meshcat.Visualizer, x_goal):
    vis["goal/cylinder"].set_object(
            meshcat.geometry.Cylinder(height=0.001, radius=0.01),
            meshcat.geometry.MeshLambertMaterial(color=0xFF0000, reflectivity=0.8))

    pos_goal = meshcat.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    vis['goal'].set_transform(meshcat.transformations.translation_matrix(x_goal) @ pos_goal)
    

class Observer(LeafSystem):
    def __init__(self, geometries):
        LeafSystem.__init__(self)
        self.geometries = geometries

        self.query_obj_input_port =  self.DeclareAbstractInputPort(
            "body_pose", AbstractValue.Make(QueryObject()))
        self.obs_output_port = self.DeclareVectorOutputPort(
            "observation", BasicVector(1), self.get_observation)

    def get_observation(self, context, output):
        query_object = self.query_obj_input_port.Eval(context)
        shortest_dis = np.inf
        for robot in self.geometries["2d_robot"]:
            for obstacle in self.geometries["obstacle"]:
                sd = query_object.ComputeSignedDistancePairClosestPoints(robot, obstacle)
                shortest_dis = np.min((shortest_dis, sd.distance))
        y = output.get_mutable_value()
        y[:] = shortest_dis


def calc_do_dx(inspector, sd):
    frame_A_id = inspector.GetFrameId(sd.id_A)
    frame_B_id = inspector.GetFrameId(sd.id_B)
    frameA = plant.GetBodyFromFrameId(frame_A_id).body_frame()
    frameB = plant.GetBodyFromFrameId(frame_B_id).body_frame()
    p_ACa = sd.p_ACa
    nhat_BA_W = sd.nhat_BA_W
    Jacobian = plant.CalcJacobianTranslationalVelocity(plant_context, JacobianWrtVariable.kQDot, frameA, p_ACa, frameB, plant.world_frame())
    do_dx = nhat_BA_W.T @ Jacobian
    return do_dx




proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.01)
inspector = scene_graph.model_inspector()
parser = Parser(plant)
model_dir = os.path.join(os.getcwd(),"neural_clbf/systems/models")
parser.AddModelFromFile(os.path.join(model_dir, "2d_robot.sdf"), "2d_robot")
parser.AddModelFromFile(os.path.join(model_dir, "obstacle.sdf"), "obstacle")
plant.Finalize()
idx_robot = plant.GetModelInstanceByName("2d_robot")
goal_state = np.array([-0.75482975,  2.15316058, 0, 0])
collision_group = {"obstacle":["box_0", "box_1"], "2d_robot":["link_1", "link_2"]}
geometries = {"obstacle":[], "2d_robot":[]}
for group in collision_group.keys():
    for body_name in collision_group[group]:
        body = plant.GetBodyByName(body_name)
        frame_id = plant.GetBodyFrameIdIfExists(body.index())
        geometry = inspector.GetGeometryIdByName(frame_id, role=Role.kProximity, name=group +"::"+ body_name)
        geometries[group].append(geometry)

kp = np.array([10, 10])
ki = np.array([0.1, 0.1])
kd = np.array([5, 5])
controller = builder.AddSystem(InverseDynamicsController(plant, kp, ki, kd, has_reference_acceleration=False))
builder.Connect(controller.get_output_port(),
                plant.get_actuation_input_port())
builder.Connect(plant.get_state_output_port(), controller.get_input_port(0))

viz = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url)
vis = viz.vis
set_orthographic_camera_xy(vis)
dt = 0.01
state_logger = LogVectorOutput(plant.get_state_output_port(), builder, dt)
observer = builder.AddSystem(Observer(geometries))
builder.Connect(scene_graph.get_query_output_port(), observer.get_input_port())

obs_logger = LogVectorOutput(observer.get_output_port(), builder, dt)
diagram = builder.Build()

simulator = Simulator(diagram)

context = simulator.get_mutable_context()
controller_context = controller.GetMyMutableContextFromRoot(context)
plant_context = plant.GetMyMutableContextFromRoot(context)
desired_state_port = controller.get_input_port_desired_state().FixValue(controller_context, goal_state)

query_object = plant.get_geometry_query_input_port().Eval(plant_context)

# print(query_object.HasCollisions())

# ee_frame = plant.GetBodyByName("ee").body_frame()
ee_pos = np.array([-0.06, 0.18, 0])
add_goal_meshcat(vis, ee_pos)
# ref_config = np.array([-0.1, np.pi/4])

# ik = InverseKinematics(plant, plant_context)
# ik.prog().AddBoundingBoxConstraint(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits(), ik.q())
# ik.prog().SetInitialGuess(ik.q(), ref_config)
# ik.prog().AddQuadraticCost((ik.q() - ref_config).dot(ik.q() - ref_config))

# ik.AddPositionConstraint(ee_frame, [0, 0, 0], plant.world_frame(), ee_pos, ee_pos)

# result = Solve(ik.prog())
# viz_q = result.GetSolution(ik.q())
# print(repr(viz_q))

# plant.SetPositions(plant_context, [0, np.pi/2])
# diagram.Publish(context)
sd = query_object.ComputeSignedDistancePairClosestPoints(geometries["obstacle"][0],geometries["2d_robot"][1])
calc_do_dx(inspector, sd)

simulator.set_target_realtime_rate(0.5)
simulator.Initialize()
viz.start_recording()
theta2 = np.linspace(0, np.pi/2, 10)
simulator.AdvanceTo(2)
viz.publish_recording()

state_logger.FindLog(context).data()