"""
Demo file used to run experiments for the merge scene algorithm
"""
import assignment_1.utils.parsing as parsing
from assignment_1.icp.merging import *


# used ICP algorithm
icp = ICP()
# threshold
threshold = 1.0
# eval
draw = True


if __name__ == '__main__':
    frames = parsing.parse_all_frames()
    merged = merge_scene(frames, icp, 1, threshold, iterative=False)
    if draw:
        open3d.draw_geometries([merged])
