"""
This file contains all the functions and classes
related to the merging of scenes using ICP.
"""
from assignment_1.icp.iterative_closest_point import *
from copy import deepcopy


def merge_scene(frames, icp: ICP, step, threshold, iterative=False):
    """
    Merges the scene using the list of frames
    and the icp algorithm that finds
    transformations between the frames.
    :param frames: list of PointClouds
    :param icp: ICP class
    :param step: how many frames to skip
    :param threshold: used for ICP
    :param iterative: iterative merging or not
    :return: merged PointCloud
    """
    # initializations
    no_frames = len(frames)
    frame_index = 0
    base = frames[frame_index]
    frame_index += step
    merged = open3d.PointCloud()
    merged += deepcopy(base)

    # start loop
    pbar = tqdm(total=no_frames - 1)
    while frame_index < no_frames:
        # get target
        target = frames[frame_index]
        # get the transformation using ICP
        transformation = icp.registration_icp(base, target, threshold)

        # update the merged cloud
        merged.transform(transformation)
        merged += deepcopy(target)

        # update base
        if iterative:
            idx = np.random.choice(len(merged.points), size=30000)
            base = open3d.select_down_sample(input=merged, indices=idx)
        else:
            base = target

        # updates
        frame_index += step
        pbar.update(step)

    pbar.close()
    return merged


def merge_scene_naive(frames, icp: ICP, step, threshold):
    """
    Merges the scene using the list of frames
    and the icp algorithm that finds
    transformations between the frames.
    This version only uses consecutive frames
    to determine the transformations.
    :param frames: list of PointClouds
    :param icp: ICP class
    :param step: how many frames to skip
    :param threshold: used for ICP
    :return: merged PointCloud
    """
    # initializations
    no_frames = len(frames)
    frame_index = 0
    base = frames[frame_index]
    frame_index += step
    merged = open3d.PointCloud()

    # initialize matrices
    rotation = np.eye(3)
    translation = np.zeros(3)
    transformation = np.eye(4)

    # start loop
    pbar = tqdm(total=no_frames - 1)
    while frame_index < no_frames:
        # get target
        target = frames[frame_index]
        # get the transformation using ICP
        trans = icp.registration_icp(base, target, threshold)
        r = trans[:3, :3]
        t = trans[:3, -1]

        # update the merged cloud
        rotation = np.dot(r, rotation)
        translation = np.dot(r, translation) + t
        transformation[:3, :3] = rotation
        transformation[:3, -1] = translation
        base.transform(transformation)
        merged += base

        # update base
        base = target

        # updates
        frame_index += step
        pbar.update(step)

    pbar.close()
    return merged


def merge_scene_iterative(frames, icp: ICP, step, threshold):
    """
    Merges the scene using the list of frames
    and the icp algorithm that finds
    transformations between the frames.
    This version uses the already transformed
    concatenation of frames to determine
    the next transformation.
    :param frames: list of PointClouds
    :param icp: ICP class
    :param step: how many frames to skip
    :param threshold: used for ICP
    :return: merged PointCloud
    """
    # initializations
    no_frames = len(frames)
    sample_size = 10000
    merged = frames[0]
    idx = np.random.choice(len(merged.points), size=sample_size)
    vis_points = open3d.select_down_sample(input=merged, indices=idx)
    # start looping over frames to transform the point clouds
    for i_frame in tqdm(range(step, no_frames, step)):
        # get sub sampled base
        idx = np.random.choice(len(merged.points), size=sample_size)
        base = open3d.select_down_sample(input=merged, indices=idx)
        # get target a
        target = frames[i_frame]
        idx = np.random.choice(len(base.points), size=5 * sample_size)
        sub_target = open3d.select_down_sample(input=base, indices=idx)
        transformation = icp.registration_icp(base, sub_target, threshold)
        merged.transform(transformation)
        merged += target
        vis_points.transform(transformation)
        vis_points += sub_target
    return merged
