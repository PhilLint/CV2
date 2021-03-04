import copy
import matplotlib.pyplot as plt
import numpy as np
import random


from mpl_toolkits.mplot3d import axes3d, Axes3D

from fundamental_matrix import *
from chaining import *
from procrustes import *



def structure_from_motion(measurement_matrix, take_sqrt=True):
    """Summary
    
    Args:
        measurement_matrix (TYPE): Description

    Returns:
        TYPE: Description
    """

    # Apply SVD

    measurement_matrix = normalizeMM(measurement_matrix)
    U, W, Vt = np.linalg.svd(measurement_matrix)

    # Reduce U,W,Vt to rank 3
    U_3 = U[:,:3]
    W_3 = np.diag(W[:3])
    Vt_3 = Vt[:3,:]

    # Get D = MS
    D = U_3 @ W_3 @ Vt_3
    if take_sqrt:
        M = U_3 @ np.sqrt(W_3)
        S = np.sqrt(W_3) @ Vt_3
    else:
        M = U_3 @ W_3
        S = Vt_3
    # return M, S
    return M,S

def pvm2MM(match_list, kp_list, descr_list):
    """ Turns Point-View-Matrices into Measurement Matrix

    Args:
        match_list (Match() object): fully dense match_list; contains attributes im_ids, descr_ids
        kp_list (TYPE): containing the keypoints for images in match_list
        descr_list (list of descr-lists for the images): containing the desriptors for the images in match_list
    Return:
        2MxN Measurement Matrix
    """

    mm = np.zeros((2*len(kp_list), len(match_list)))

    for m_id, match_obj in enumerate(match_list):
        im_ids = match_obj.im_ids
        descr_ids = match_obj.descr_ids
        for i, im_idx in enumerate(im_ids):
            list_index = im_ids.index(im_idx)
            descr_id = descr_ids[list_index]
            keypoint = kp_list[im_idx][descr_id]

            x_value = keypoint.pt[0]
            y_value = keypoint.pt[1]

            mm[2*i, m_id] = x_value
            mm[2*i+1, m_id] = y_value

    return mm

def normalizeMM(mm):
    """Normalizing the measurement matrix
    
    Args:
        mm (2D-np.array): measurement matrix
    
    Returns:
        2D-np.array: normalized measurment matrix
    """
    normalized_mm = mm - mm.mean(axis=1,keepdims=True)

    return normalized_mm

def get_dense_match_list(match_list, image_count, start_im_id=0):#, point_count=None):
    """Get the match list that corresponds to a fully dense point-view-matrix

    This function returns a match_list that is part of the original match_list.
    It only contains match_obj's that contain the first image and are matched to
    regions in the first "image_count" images.
    In the end this match_list corresponds to a fully dense point-view-matrix.
    
    Args:
        match_list (list of Match() objects):
        image_count (int): Number of images that should be contained in all
            in the new match_list.
    
    Returns:
        list of Match() objects:
    """
    match_list = copy.deepcopy(match_list)

    new_match_list = []
    match_ids = []

    for idx, match_obj in enumerate(match_list):
        im_ids = match_obj.im_ids
        if start_im_id in im_ids:
            im_ids = im_ids[im_ids.index(start_im_id):]
            if len(im_ids) >= image_count:
                new_match_list.append(match_obj)
                match_ids.append(idx)

    for match_obj in new_match_list:
        start_idx = match_obj.im_ids.index(start_im_id)
        match_obj.im_ids = match_obj.im_ids[start_idx:(start_idx+image_count)]
        match_obj.descr_ids = match_obj.descr_ids[start_idx:(start_idx+image_count)]

    return new_match_list, match_ids



def readPVM(filename):
    """Allows the import of the PointViewMatrix.txt file 
    
    Args:
        filename (string): Description
    
    Returns:
        2-D np.array: Measurement matrix of the PointViewMatrix.txt file
    """
    data = []

    f = open(filename, "r")
    for x in f:
        data.append(x.strip().split(" "))

    mm = np.array([[float(entry) for entry in row] for row in data])

    return mm


#########################
####### Test Area #######
#########################

if __name__ == '__main__':

    random.seed(2)

    IMAGE_COUNT = 10


    # image directory
    imgs_dir = os.path.join("Data")
    # get all image paths
    images_path = []
    images = []
    for img_path in glob.glob(os.path.join(imgs_dir, '*.png')):
        images_path.append(img_path)
    # sort names
    
    images_path = sorted(images_path)
    #random.shuffle(images_path)

    for idx, im in enumerate(images_path):
        if idx < IMAGE_COUNT:
            images.append(cv2.imread(im))

    matcher = cv2.BFMatcher()

    keypoints = []
    descriptors = []

    for idx, im in enumerate(images):
        kp, d = get_keypoints_and_descriptors(im)
        keypoints.append(kp)
        descriptors.append(d)

    pvm, match_list = get_PVM(matcher, keypoints, descriptors, images)

    # 1. Get dense match_list
    dense_match_list, match_ids = get_dense_match_list(match_list, IMAGE_COUNT)
    print(match_ids)

    # 2. Get MM
    mm = pvm2MM(dense_match_list, keypoints, descriptors)
    mm = normalizeMM(mm)

    #mm = readPVM("PointViewMatrix.txt")

    M, S = structure_from_motion(mm)
    M, S = AAremoval(M,S)


    fig = plt.figure()
    ax = Axes3D(fig)
    
    plot_points(S,4,ax,True)

    plt.show()
