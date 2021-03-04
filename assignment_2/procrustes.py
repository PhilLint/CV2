from chaining import *
from structure_from_motion import *
from main import *
import random


import numpy as np
from scipy.linalg import orthogonal_procrustes



"""
The following Code for the "procrustes"-function was an adaption of the Matlab-version of the procrustes method.
We found the code online on the following website:
https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
and adapted it to our needs.
As people were allowed to use the procrustes function in matlab, we assumed that it would
be fine for people using python, to use already existing functions.

"""


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection:# != have_reflection:
            if s[-1] < 0:
                V[:,-1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform, muX, muY, s


def get_pair_lists(match_ids1, match_ids2):

    match_id_pair1 = []
    match_id_pair2 = []

    for idx, m1 in enumerate(match_ids1):
        if m1 in match_ids2:
            match_id_pair1.append(idx)
            match_id_pair2.append(match_ids2.index(m1))


    return match_id_pair1, match_id_pair2


def stitch(point_set1, point_set2, match_ids1, match_ids2):
    """Stitches point_set2 onto point_set1 using procrustes
    
    Parameters
    ----------
    point_set1 : 3xm np.array
        Description
    point_set2 : 3xn np.array
        Description
    """
    if len(set(match_ids2) - set(match_ids1)) == 0:
        return point_set1, match_ids1, np.zeros(0), 0

    im_ids1, im_ids2 = get_pair_lists(match_ids1, match_ids2)

    X = point_set1[:,np.array(im_ids1)]
    Y = point_set2[:,np.array(im_ids2)]

    new_match_ids = list(set([i for i in range(point_set2.shape[1])]) - set(im_ids2))

    new_points_ids_arr = np.array(new_match_ids)
    #new_match_ids = sorted(set(match_ids1 + match_ids))

    new_match_ids = [idx for i,idx in enumerate(match_ids2) if i in new_match_ids]

    total_match_ids = match_ids1 + new_match_ids

    #R, s, Xmean, Ymean, Xnorm, Ynorm = my_procrustes(X,Y)
    Y_new = point_set2[:,new_points_ids_arr].T

    #print(X.shape, Y.shape)
    d, Z, tform, muX, muY, s = procrustes(X.T,Y.T,scaling=True, reflection='best')
    #print(Z.shape)
    Y0 = Y_new - muY

    Y_tf = (tform['scale']*np.dot(Y0, tform['rotation']) + muX).T

    return np.concatenate((point_set1, Y_tf),axis=1), total_match_ids, Y_tf, s



def AAremoval(M, S):
    L = np.linalg.pinv(M) @ np.linalg.pinv(M.T)

    C = np.linalg.cholesky(L)

    M = M @ C
    S = np.linalg.inv(C) @ S

    return M, S

def prone_points(S,value=6):
    proned_S = S[:,S[2,:] < value]
    proned_S = proned_S[:,proned_S[2,:] > -value]
    return proned_S

def plot_points(S,i,ax,prone=True,prone_value=6):
    if prone:
        pS = prone_points(S,prone_value)
        ax.scatter3D(pS[0,:], pS[1,:], pS[2,:])
    else:
        ax.scatter3D(S[0,:], S[1,:], S[2,:])


def main():

    random.seed(42)
    IMAGE_COUNT = 49
    CONSEQ_IMAGES = 4

    # image directory
    imgs_dir = os.path.join("Data")
    # get all image paths
    images_path = []
    images = []
    for img_path in glob.glob(os.path.join(imgs_dir, '*.png')):
        images_path.append(img_path)
    
    images_path = sorted(images_path)


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

    kp_list = keypoints
    descr_list = descriptors

    main_S = None
    main_match_ids = None

    fig = plt.figure()
    global ax
    ax = Axes3D(fig)
 
    for i in range(0,IMAGE_COUNT - CONSEQ_IMAGES+1):

        current_match_list, match_ids = get_dense_match_list(match_list, CONSEQ_IMAGES, i)

        mm = pvm2MM(current_match_list,kp_list, descr_list) 

        M, S = structure_from_motion(mm)

        #S -= np.amin(S)
        #M -= np.amin(M)
        M, S = AAremoval(M,S)

        if i == 0:
            main_S = S
            main_match_ids = match_ids
            plot_points(main_S, i, ax)

        else:
            main_S, main_match_ids, Y_tf, s = stitch(main_S, S, main_match_ids, match_ids)
            plot_points(main_S, i, ax)


    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    main()

