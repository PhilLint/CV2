"""
Module used for visualizing PCD
TODO: Fix float to rgb (they now lead to different colors)
"""
import struct
import numpy as np
import open3d
import matplotlib.pyplot as plt
from copy import deepcopy


def plot(metrics):
    plt.plot(metrics[0], label='ICP-1')
    plt.plot(metrics[1], label='ICP-2')
    plt.plot(metrics[2], label='ICP-3')
    plt.plot(metrics[3], label='ICP-4')
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('RMSE')
    plt.title('RMSE over time')
    plt.show()


def draw_registration_results(source, target, transformation):
    """
    Draws the transformed source and target together.
    :param source: open3d PointCloud
    :param target: open3d PointCloud
    :param transformation: the transformation matrix used on source
    """
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.draw_geometries([source_temp, target_temp])


def draw_frames(frames):
    """
    Draws all frames
    :param frames: list of PointCloud
    """
    open3d.draw_geometries(frames)


def visualize_pcd(pcd):
    """
    Visualizes PCD.
    If the given input is not a PointCloud,
    then the numpy array is converted to PointCloud.
    :param pcd: either numpy array of size (N, 4)
                or directly a PointCloud class
    """
    if type(pcd) is not open3d.PointCloud:
        pc = open3d.PointCloud()
        pc.points = open3d.Vector3dVector(pcd[:, [0, 1, 2]])
        pc.colors = open3d.Vector3dVector(float_to_rgb(pcd[:, 3]))
        pcd = pc
    open3d.draw_geometries([pcd])


def float_to_rgb(floats):
    """
    Converts floats to RGBs.
    :param floats: the input floats of shape (N, 1)
    :return: the RGB colors of shape (N, 3)
    """
    rgb = np.zeros((floats.shape[0], 3), dtype=np.float)
    for i in range(len(floats)):
        f = struct.unpack('Q', floats[i])[0]
        r = (f >> 16) & 0x0000f
        g = (f >> 8) & 0x0000ff
        b = f & 0x0000ff
        rgb[i, 0] = r
        rgb[i, 1] = g
        rgb[i, 2] = b
    return rgb / 255.0
