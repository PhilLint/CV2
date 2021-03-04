"""
Pinhole Camera Model functions
"""
import numpy as np
from copy import deepcopy

from assignment_3.utils.data_def import Mesh


def project_mesh(mesh: Mesh, transformation, viewport, projection):
    """
    Projects the given mesh using the
    given transformation, viewport,
    and projection matrices.
    :param mesh: to be projected
    :param transformation: matrix
    :param viewport: matrix
    :param projection: matrix
    :return: the projected mesh
    """
    mesh_c = deepcopy(mesh)
    # get the points of mesh and transpose it
    points = mesh_c.vertices.T
    # add the fourth dimension
    s = np.vstack((points, np.ones(points.shape[1])))

    # compute the PI matrix
    pi_matrix = np.dot(viewport, projection)
    # compute the transformation
    transformed = np.dot(transformation, s)
    # compute the projection
    projected = np.dot(pi_matrix, transformed)

    # transform to homogeneous coordinates
    projected /= projected[3, :]

    # add to the new mesh
    mesh_c.vertices = projected[:3].T

    return mesh_c


def transform_mesh(mesh: Mesh, transformation):
    """
    Transforms the given mesh using the
    given transformation matrix.
    This is not an in place operation
    :param mesh: to be transformed
    :param transformation: matrix
    :return: the transformed mesh
    """
    mesh_c = deepcopy(mesh)
    # get the points of mesh and transpose it
    points = mesh_c.vertices.T
    # add the fourth dimension
    s = np.vstack((points, np.ones(points.shape[1])))

    # transform
    transformed = np.dot(transformation, s)[:3].T

    # add the transformed points to the mesh and return
    mesh_c.vertices = transformed
    return mesh_c


def get_transformation_matrix(angle, x_t=0, y_t=0, z_t=0, axis='y'):
    """
    Constructs a transformation matrix given the angle
    and the offsets for all coordinates.
    :param angle: the rotation angle in degrees.
    :param x_t: the translation for coordinate x
    :param y_t: the translation for coordinate y
    :param z_t: the translation for coordinate z
    :param axis: the axis around which to rotate
    :return: the transformation matrix
    """
    # transform angle
    angle = angle * np.pi / 180
    # initialize the transformation matrix
    transformation = np.eye(4, dtype=np.float32)

    # put the rotation in the transformation depending on axis
    if axis == 'x':
        # rotation according to axis x
        transformation[:3, :3] = np.asarray(
            [[1, 0, 0],
             [0, np.cos(angle), -np.sin(angle)],
             [0, np.sin(angle), np.cos(angle)]],
            dtype=np.float32
        )
    elif axis == 'y':
        # rotation according to axis y
        transformation[:3, :3] = np.asarray(
            [[np.cos(angle), 0, np.sin(angle)],
             [0, 1, 0],
             [-np.sin(angle), 0, np.cos(angle)]],
            dtype=np.float32
        )
    elif axis == 'z':
        # rotation according to axis z
        transformation[:3, :3] = np.asarray(
            [[np.cos(angle), -np.sin(angle), 0],
             [np.sin(angle), np.cos(angle), 0],
             [0, 0, 1]],
            dtype=np.float32
        )
    else:
        raise ValueError('Invalid axis')

    # put the translation in the transformation
    transformation[:3, -1] = np.asarray([x_t, y_t, z_t], dtype=np.float32)

    return transformation


def get_viewport_matrix(width=1024, height=768):
    """
    Constructs and returns the viewport matrix
    :param width: width of the viewport
    :param height: width of the height
    :return: the viewport matrix
    """
    c_x = width / 2
    c_y = height / 2
    viewport = np.zeros((4, 4), dtype=np.float32)
    viewport += np.diag([c_x, -c_y, 0.5, 1])
    viewport[:3, -1] = np.asarray([c_x, c_y, 0.5], dtype=np.float32)
    return viewport


def get_projection_matrix(far=2000, near=300, fov_y=0.5, width=1024, height=768):
    """
    Constructs and returns the perspective projection matrix.
    :param far: far plane
    :param near: near plane
    :param fov_y: field of view y
    :param width: width of the viewport
    :param height: width of the height
    :return: the perspective projection matrix
    """
    # compute top, bottom, right, and left
    aspect_ration = width / height
    top = np.tan(fov_y / 2) * near
    bottom = -top
    right = top * aspect_ration
    left = -top * aspect_ration

    # define projection matrix
    projection = np.zeros((4, 4), dtype=np.float32)

    # compute projection matrix
    projection[0, 0] = 2 * near / (right - left)
    projection[0, 2] = (right + left) / (right - left)
    projection[1, 1] = 2 * near / (top - bottom)
    projection[1, 2] = (top + bottom) / (top - bottom)
    projection[2, 2] = -(far + near) / (far - near)
    projection[2, 3] = -2 * far * near / (far - near)
    projection[3, 2] = -1

    return projection
