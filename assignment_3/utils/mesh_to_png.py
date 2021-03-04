import os
from skimage.io import imsave
import numpy as np
import trimesh
from pathlib import Path
import pyrender
import sys

import h5py

p = Path(__file__).resolve().parents[2]
sys.path.append(str(p))

from assignment_3.utils.data_def import PCAModel, Mesh

FILE_MODEL = p / 'assignment_3/data/model2017-1_face12_nomouth.h5'

bfm = h5py.File(FILE_MODEL, 'r')

mean_shape = np.asarray(bfm['shape/model/mean'], dtype=np.float32).reshape((-1, 3))
mean_tex = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape((-1, 3))

triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T

def mesh_to_png(file_name, mesh, width=640, height=480, z_camera_translation=400):
    mesh = trimesh.base.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.triangles,
        vertex_colors=mesh.colors)

    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, wireframe=False)

    # compose scene
    scene = pyrender.Scene(ambient_light=np.array([1.7, 1.7, 1.7, 1.0]), bg_color=[255, 255, 255])
    camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)

    scene.add(mesh, pose=np.eye(4))
    scene.add(light, pose=np.eye(4))

    # Added camera translated z_camera_translation in the 0z direction w.r.t. the origin
    scene.add(camera, pose=[[ 1,  0,  0,  0],
                            [ 0,  1,  0,  0],
                            [ 0,  0,  1,  z_camera_translation],
                            [ 0,  0,  0,  1]])

    # render scene
    r = pyrender.OffscreenRenderer(width, height)
    color, _ = r.render(scene)

    imsave(file_name, color)


def file_to_mesh(file):
    """
    Reads the face model and transforms it to a Mesh
    :param file: to be transformed to mesh
    :return: the Mesh class
    """
    # read the file using h5py
    bfm = h5py.File(file, 'r')

    # get the facial identity
    mean_shape = np.asarray(bfm['shape/model/mean'], dtype=np.float32).reshape((-1, 3))
    # get the facial expression
    mean_exp = np.asarray(bfm['expression/model/mean'], dtype=np.float32).reshape((-1, 3))

    # get the mean face color
    mean_tex = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape((-1, 3))
    # get the triangles
    triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T

    # return the Mesh
    return Mesh(mean_shape + mean_exp, mean_tex, triangles)


if __name__ == '__main__':
    mesh = Mesh(mean_shape, mean_tex, triangles)
    mesh_to_png("debug.png", mesh)
    mesh_to_png("debug1234.png", mesh)