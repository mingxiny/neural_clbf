import os
import time
import pickle
import numpy as np
import multiprocessing as mp

from pydrake.geometry import CollisionFilterDeclaration, GeometrySet, Role, SceneGraph

from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import LoadModelDirectives, Parser, ProcessModelDirectives
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import RevoluteJoint
from pydrake.solvers.mathematicalprogram import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.systems.primitives import TrajectorySource
from pydrake.systems.rendering import MultibodyPositionToGeometryPose

from meshcat import Visualizer
import meshcat.geometry as g

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])

vis = Visualizer(zmq_url=zmq_url)
vis.delete()
# display(vis.jupyter_cell())

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
parser = Parser(plant)
model_path = os.path.join(os.getcwd(),"neural_clbf/systems")
parser.package_map().Add("neural_clbf", model_path)

# directives_file = os.path.join(model_path, "models/iiwa14_spheres_collision_welded_gripper.yaml")
directives_file = os.path.join(model_path, "models/bimanual_iiwa.yaml")
directives = LoadModelDirectives(directives_file)
models = ProcessModelDirectives(directives, plant, parser)
# [iiwa, wsg, shelf, binR, binL, table] =  models
 

plant.Finalize()

viz_role = Role.kIllustration
visualizer = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url,
                                      delete_prefix_on_load=False, role=viz_role)
diagram = builder.Build()

visualizer.load()
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyMutableContextFromRoot(context)
diagram.Publish(context)