"""
Parsing module
"""
import re
import numpy as np
import open3d
import scipy.io
from tqdm import tqdm


def parse_pcd_numpy(filename):
    """
    Parses a pcd file using Numpy.
    :param filename: the file to be parsed
    :return: a (N, 4) array, where N is number of points
    """
    pcd = np.genfromtxt(filename, delimiter='', skip_header=11)
    return pcd


def parse_pcd_open3d(filename):
    """
    Parses a pcd file using Open3D
    :param filename: the file to be parsed
    :return: an open3d PointCloud class
    """
    pcd = open3d.read_point_cloud(filename)
    return pcd


def parse_mat(filename):
    """
    Parses a mat file and returns the data.
    :param filename: the file to be parsed
    :return: open3d PointCloud with the data
    """
    mat = scipy.io.loadmat(filename)
    key = re.findall(r"[\w']+", filename)[1]
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(mat[key].T)
    return pcd


def parse_all_frames():
    """
    Parses all frames in the data folder.
    The filenames are 00000000XX.pcd
    :return: the list of frames
    """
    frames = []
    print('Parsing frames..')
    for i in tqdm(range(100)):
        filename = './data/data/00000000'
        if i < 10:
            filename += ('0' + str(i))
        else:
            filename += str(i)
        filename += '.pcd'
        frames.append(parse_pcd_open3d(filename))
    return frames
