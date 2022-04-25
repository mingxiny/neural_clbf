import meshcat
import numpy as np

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