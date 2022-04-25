import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import (AddMultibodyPlantSceneGraph, ConstantVectorSource,
                         ControllabilityMatrix, DiagramBuilder,
                         FirstOrderTaylorApproximation, Linearize,
                         LinearQuadraticRegulator, MatrixGain,
                         MeshcatVisualizerCpp, MultibodyPlant, Parser,
                         Saturation, SceneGraph, Simulator, StartMeshcat,
                         ToLatex, WrapToSystem)


from underactuated import FindResource, running_as_notebook

meshcat = StartMeshcat()

def cartpole_balancing_example():
    def UprightState():
        state = (0, np.pi, 0, 0)
        return state

    def Controllability(plant):
        context = plant.CreateDefaultContext()
        plant.get_actuation_input_port().FixValue(context, [0])

        context.get_mutable_continuous_state_vector().SetFromVector(
            UprightState())

        linearized_plant = Linearize(
            plant,
            context,
            input_port_index=plant.get_actuation_input_port().get_index(), output_port_index=plant.get_state_output_port().get_index())
        print(
            f"The singular values of the controllability matrix are: {np.linalg.svd(ControllabilityMatrix(linearized_plant), compute_uv=False)}"
        )

    def BalancingLQR(plant):
        # Design an LQR controller for stabilizing the CartPole around the upright.
        # Returns a (static) AffineSystem that implements the controller (in
        # the original CartPole coordinates).

        context = plant.CreateDefaultContext()
        plant.get_actuation_input_port().FixValue(context, [0])

        context.get_mutable_continuous_state_vector().SetFromVector(UprightState())

        Q = np.diag((10., 10., 1., 1.))
        R = [1]

        # MultibodyPlant has many (optional) input ports, so we must pass the
        # input_port_index to LQR.
        return LinearQuadraticRegulator(
            plant,
            context,
            Q,
            R,
            input_port_index=plant.get_actuation_input_port().get_index())


    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    file_name = FindResource("models/cartpole.urdf")
    Parser(plant).AddModelFromFile(file_name)
    plant.Finalize()

    Controllability(plant)

    controller = builder.AddSystem(BalancingLQR(plant))
    builder.Connect(plant.get_state_output_port(), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0),
                    plant.get_actuation_input_port())

    # Setup visualization
    meshcat.Delete()
    meshcat.Set2dRenderMode(xmin=-2.5, xmax=2.5, ymin=-1.0, ymax=2.5)
    MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    # Simulate
    simulator.set_target_realtime_rate(1.0 if running_as_notebook else 0.0)
    duration = 5.0 #if running_as_notebook else 0.1
    for i in range(5):
        context.SetTime(0.)
        context.SetContinuousState(UprightState() + 0.1 * np.random.randn(4,))
        simulator.Initialize()
        simulator.AdvanceTo(duration)

cartpole_balancing_example()