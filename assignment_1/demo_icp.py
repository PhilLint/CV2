"""
Demo file used to run experiments with the different ICP algorithms
"""
import time
import assignment_1.utils.parsing as parsing
import assignment_1.utils.visualize as viz
from assignment_1.icp.iterative_closest_point import *


# used ICP algorithm
icp1 = ICP(
    sampling='all',
    selection='all',
)
# threshold
threshold = 1.0
# eval
do_eval = True


if __name__ == '__main__':
    filename1 = 'data/source.mat'
    filename2 = 'data/target.mat'
    pcd1 = parsing.parse_mat(filename1)
    pcd2 = parsing.parse_mat(filename2)
    start = time.time()
    transformation, _ = icp1.registration_icp(pcd1, pcd2, threshold)
    end = time.time()
    print(end - start)
    # print(transformation)
    if do_eval:
        eval_icp = open3d.evaluate_registration(pcd1, pcd2, threshold, transformation)
        print(eval_icp)
        viz.draw_registration_results(pcd1, pcd2, transformation)
