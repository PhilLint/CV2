"""
This module contains different implementation of the
Iterative Closest Point (ICP) algorithm.
The base implementation is contained in the ICP class,
with more refined implementations contained in the
subclasses derived from the base ICP class.
"""
import numpy as np
import open3d
from sys import maxsize
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans


class ICP:
    """
    The base class of the ICP algorithm.
    If you want to make a new implementation for the ICP,
    you should subclass this and implement the following methods:
     - registration_icp
    """
    def __init__(self, max_iterations=300,
                 rmse_criterion=1e-4,
                 sampling='all',
                 selection='all',
                 matching='kd',
                 weighting='equal'):
        """
        :param max_iterations: maximum number of iterations
        :param rmse_criterion: stoppage criterion for difference between RMSE score
        :param sampling: the sampling used to select points
        :param selection: the selection used to select points
        :param matching: the matching used to match points
        :param weighting: weighting used to calculate the centroids
        """
        self.max_iterations = max_iterations
        self.rmse_criterion = rmse_criterion
        self.sample = self._get_sampling(sampling)
        self.select = self._get_selection(selection)
        self.match = self._get_matching(matching)
        self.get_weights = self._get_weighting(weighting)

    def registration_icp(self,
                         source,
                         target,
                         threshold,
                         transformation=np.eye(4),
                         has_normals=False,
                         subsamples=1000):
        """
        Applies the ICP algorithm to get the transformation matrix such that
        when applied to the source point cloud, the RMSE between the resulting
        point cloud and the target point cloud is minimized.
        :param source: open3d PointCloud
        :param target: open3d PointCloud
        :param transformation: initial transformation matrix of shape (4, 4
        :param threshold: threshold used for rejection of pairs
        :param has_normals: boolean True if the source and target PointClouds have normals
                            and False otherwise; If False calculate, the normals
        :param subsamples: number of samples to select from each point cloud
        :return: the revised transformation matrix that minimizes the RMSE
        """
        # check for translation matrix shape
        if transformation.shape != (4, 4):
            raise ValueError('Translation matrix is not 4x4')
        # check source and target input
        if type(source) is not open3d.PointCloud:
            raise ValueError('source must be an open3d PointCloud')
        if type(target) is not open3d.PointCloud:
            raise ValueError('target must be an open3d PointCloud')

        # calculate normals if needed
        if not has_normals:
            open3d.estimate_normals(source)
            open3d.estimate_normals(target)

        # convert to numpy arrays
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)

        # sample points based on the sample function
        source_points, target_points = self.sample(source_points, target_points, subsamples)
        # get first matching between source and target
        matching = self.match(source_points, target_points)

        # initialize the metrics, iterations, list for metrics and rotation and translation
        rmse = 0.0
        old_rmse = self.rmse(source_points, matching)
        metrics = [old_rmse]
        iteration = 0
        rotation = transformation[:3, :3]
        translation = transformation[:3, -1]

        # start loop
        while abs(rmse - old_rmse) > self.rmse_criterion and iteration < self.max_iterations:
            # select points based on the selection function
            sel_source, sel_target = self.select(source_points, target_points, subsamples)
            # get the matching of those selected points
            sel_matching = self.match(sel_source, sel_target)

            # reject by distance
            sel_source, sel_matching = self.reject(sel_source, sel_matching, threshold)
            # if the threshold is too low, reset it and continue
            if len(sel_source) == 0:
                threshold = 1.0
                continue
            # update threshold
            threshold = self.update_threshold(sel_source, sel_matching, threshold)

            # get the weights used for calculating the centroids
            weights = self.get_weights(sel_source, sel_matching)
            # get the centroids
            source_centroid = self.get_centroid(sel_source, weights)
            target_centroid = self.get_centroid(sel_matching, weights)
            # compute the centered vectors
            x = sel_source - source_centroid
            y = sel_matching - target_centroid

            # compute the covariance matrix S
            s = np.dot(x.T, y * weights.reshape((weights.shape[0], -1)))
            # use SVD to get U and V
            u, _, vh = np.linalg.svd(s)
            v = vh.T
            # compute matrix M
            m = np.eye(s.shape[0])
            m[-1, -1] = np.linalg.det(np.dot(v, u.T))
            # compute the rotation and translation updates
            r = np.dot(np.dot(v, m), u.T)
            t = target_centroid - np.dot(r, source_centroid)

            # update source points
            source_points = np.dot(source_points, r.T) + t
            # update rotation and translation
            rotation = np.dot(r, rotation)
            translation = np.dot(r, translation) + t

            # recompute matching and update rmse
            matching = self.match(source_points, target_points)
            old_rmse = rmse
            rmse = self.rmse(source_points, matching)
            print('Epoch {}: RMSE = {}'.format(iteration, abs(rmse - old_rmse)))
            metrics.append(rmse)

            # updates
            iteration += 1

        # compute transformation
        transformation[:3, :3] = rotation
        transformation[:3, -1] = translation
        return transformation, metrics

    #########################
    # Redirection functions #
    #########################
    def _get_sampling(self, sampling):
        """
        Gets the sampling function given the sampling string
        :param sampling: string
        :return: the appropriate sampling function
        """
        if sampling == 'all':
            return self._select_all
        elif sampling == 'uniform':
            return self.uniform_sampling
        elif sampling == 'k-means':
            return self.k_means_sampling
        else:
            raise ValueError('No sampling function that corresponds to the given string')

    def _get_selection(self, selection):
        """
        Gets the selection function given the selection string
        :param selection: string
        :return: the appropriate selection function
        """
        if selection == 'all':
            return self._select_all
        elif selection == 'uniform':
            return self.uniform_sampling
        elif selection == 'k-means':
            return self.k_means_sampling
        else:
            raise ValueError('No selection function that corresponds to the given string')

    def _get_matching(self, matching):
        """
        Gets the matching function given the matching string
        :param matching: string
        :return: the appropriate matching function
        """
        if matching == 'brute-force':
            return self._match_brute_force
        elif matching == 'kd':
            return self._match_kd_tree
        else:
            raise ValueError('No matching function that corresponds to the given string')

    def _get_weighting(self, weighting):
        """
        Gets the weighting function given the weighting string
        :param weighting: string
        :return: the appropriate weighting function
        """
        if weighting == 'equal':
            return self._equal_weights
        else:
            raise ValueError('No weighting function that corresponds to the given string')

    ################################
    # Sampling/Selection functions #
    ################################
    @staticmethod
    def _select_all(source, target, *args):
        """
        Return all the points from source and target
        :param source: points
        :param target: points
        :return:
        """
        return source, target

    @staticmethod
    def uniform_sampling(source, target, subsamples):
        """
        Performs uniform sampling on both source and target data.
        :param source: points
        :param target: points
        :param subsamples: number of samples to select
        :return: a random sample from both source and target
        """
        idx_source = np.random.choice(source.shape[0], subsamples)
        idx_target = np.random.choice(target.shape[0], subsamples)
        s = source[idx_source, :]
        t = target[idx_target, :]
        return s, t

    @staticmethod
    def k_means_sampling(source, target, subsamples):
        """
        Performs K-Means sampling on both source and target
        :param source: points
        :param target: points
        :param subsamples: number of samples to select
        :return: the KMeans centers
        """
        k_means_source = KMeans(n_clusters=subsamples).fit(source)
        k_means_target = KMeans(n_clusters=subsamples).fit(target)
        s = k_means_source.cluster_centers_
        t = k_means_target.cluster_centers_
        return s, t

    ######################
    # Matching functions #
    ######################
    @staticmethod
    def _match_brute_force(source, target):
        """
        For each point p_s in source, this algorithm finds
        the closest point p_t in target such that the
        the Eucledian distance between them is minimized.
        This version implements the brute force algorithm.
        :param source: source points of shape (N_s, 3)
        :param target: target points of shape (N_t, 3)
        :return: the matching points
        """
        n_s = source.shape[0]
        n_t = target.shape[0]
        # initialize matching
        matching = np.zeros((n_s, 3), dtype=np.float)
        print('Matching...')
        # loop over all points in source
        for s_i in tqdm(range(n_s)):
            # initialize minimum distance
            min_dist = maxsize
            # loop over all points in target
            for t_i in range(1, n_t):
                # compute norm
                dist = np.linalg.norm(target[t_i] - source[s_i])
                # if the new distance is smaller, update
                if dist < min_dist:
                    min_dist = dist
                    matching[s_i] = target[t_i]
        return matching

    @staticmethod
    def _match_kd_tree(source, target):
        """
        For each point p_s in source, this algorithm finds
        the closest point p_t in target such that the
        the Eucledian distance between them is minimized.
        This version uses a KD tree for efficient search
        :param source: source points of shape (N_s, 3)
        :param target: target points of shape (N_t, 3)
        :return: the matching points
        """
        n_s = source.shape[0]
        # initialize PointCloud from target
        target_pcd = open3d.PointCloud()
        target_pcd.points = open3d.Vector3dVector(target)
        # initialize KD tree from target
        kd_tree = open3d.KDTreeFlann(target_pcd)
        # initialize matching
        matching = np.zeros((n_s, 3), dtype=np.float)
        # loop over all points in source
        for s_i in range(n_s):
            _, idx, _ = kd_tree.search_knn_vector_3d(source[s_i], 1)
            t_i = idx[0]
            matching[s_i] = target[t_i]
        return matching

    #######################
    # Weighting functions #
    #######################
    @staticmethod
    def _equal_weights(source, *args):
        """
        Get a vector of ones.
        :param source: shape (N, 3)
        :return: vector of ones of shape (N, )
        """
        _ = args
        return np.ones(source.shape[0], dtype=np.float)

    ###########################
    # Other utilities for ICP #
    ###########################
    @staticmethod
    def rmse(source, target):
        return np.sqrt(mean_squared_error(source, target))

    @staticmethod
    def reject(source, target, threshold):
        """
        Removes the (source, target) pairs where the distance
        between the points is higher than the threshold.
        :param source: points shape (N, 3)
        :param target: points shape (N, 3)
        :param threshold: scalar
        :return: the remaining pairs
        """
        norm = np.linalg.norm(target - source, axis=1)
        idx = np.where(norm < threshold)
        return source[idx], target[idx]

    @staticmethod
    def update_threshold(source, target, threshold):
        """
        Updates the threshold.
        :param source: points of shape (N, 3)
        :param target: points of shape (N, 3)
        :param threshold: scalar to be updated
        :return: the new threshold
        """
        norm = np.linalg.norm(target - source, axis=1)
        mean = norm.mean()
        std = norm.std()
        if mean < threshold:
            threshold = mean + 3 * std
        elif mean < 3 * threshold:
            threshold = mean + 2 * std
        elif mean < 6 * threshold:
            threshold = mean + std
        else:
            threshold = 1.0
        return threshold

    @staticmethod
    def get_centroid(points, weights):
        """
        Calculates the centroid of the points given the weights.
        :param points: of shape (N, 3)
        :param weights: of shape (N, )
        :return:
        """
        return np.mean(points * weights.reshape((weights.shape[0], -1)), axis=0)


class Open3dICP(ICP):
    """
    Uses the open3D registration_icp method
    to compute the transformation matrix.
    """
    @staticmethod
    def registration_icp(source,
                         target,
                         threshold,
                         transformation=np.eye(4),
                         **kwargs):
        # call the open3d registration_icp
        reg_p2p = open3d.registration_icp(
            source,
            target,
            threshold,
            transformation,
            open3d.TransformationEstimationPointToPoint(),
            open3d.ICPConvergenceCriteria(max_iteration=30)
        )
        return reg_p2p.transformation
