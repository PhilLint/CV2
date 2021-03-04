import numpy as np
import trimesh
import h5py

from assignment_3.utils.data_def import Mesh


def mesh_to_png(file_name, mesh):
    mesh = trimesh.base.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.triangles,
        vertex_colors=mesh.colors
    )

    png = mesh.scene().save_image()
    with open(file_name, 'wb') as f:
        f.write(png)


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
