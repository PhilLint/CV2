import numpy as np
import os
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt

def eight_point_algorithm(key_points1, key_points2):
    """
    Performs eight point algorithm, given a list of key_points, that are
    found by SIFT features, that are matched for both images. Key points
    are column vectors as specified in the Hartley paper. At first, matrix A
    is specified  as in the task sheet.
    :param key_points: (x1, y1) (x2, y2): coordinates of keypoints in im1 and im2
    :return: Fundamental matrix F
    """
    A = []
    # fill A rowwise
    for i in range(len(key_points1)):
        # coordinates of keypoints in im1 and im2
        x1, y1 = key_points1[i]
        x2, y2 = key_points2[i]
        # create A as in equation (2) of the task sheet
        A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
    # perform SVD
    U, D, Vt = np.linalg.svd(A)
    # column with smalles eigenvalue -> last diag element is smallest
    F = Vt[-1, :].T.reshape(3, 3).T
    # correct to achieve rank 2
    Uf, Df, Vt_f = np.linalg.svd(F)
    # smallest eigenvalue in Df zero
    Df[-1] = 0.0
    F = Uf @ np.diag(Df) @ Vt_f
    return F

def get_transformation_matrix(key_points):
    """
    Create transformation matrix T to set of key_points of one image, so that
    their mean is 0 and average distance to mean is sqrt(2).
    :param key_points: (x, y) : coordinates of keypoints in one image
    :return: T
    """
    # concatenate x and y coordinates
    x = [x for (x, y) in key_points]
    y = [y for (x, y) in key_points]
    # calculate mean x and mean y
    m_x = np.mean(x)
    m_y = np.mean(y)
    # mean distance
    xt = np.power((x - m_x), 2)
    yt = np.power((y - m_y), 2)
    den = np.mean(np.sqrt(xt + yt))
    # create matrix T row wise
    T = np.array([[np.sqrt(2)/den, 0, -m_x*(np.sqrt(2)/den)],
                  [0, np.sqrt(2)/den, -m_y*(np.sqrt(2)/den)],
                  [0, 0, 1]])
    return T

def normalize_points(key_points1, key_points2, T1, T2):
    """
    Normalize Keypoints by multiplication with the respective transformation matrices
    :param key_points1: coordinates kp1
    :param key_points2: coordinates kp2
    :param T1: transformation matrix 1
    :param T2: transformation matrix 2
    :return: normalized keypoint arrays norm1 and norm2
    """
    normalized_kp1 = []
    normalized_kp2 = []
    for i in range(len(key_points1)):
        # coordinates of keypoints in im1 and im2
        (x1, y1) = key_points1[i]
        (x2, y2) = key_points2[i]
        # T1 @ [x1, y1, 1] = x1_norm, y1_norm ,--- for im2 points the same
        # task sheet: p_hat = T*p
        x1, y1, _ = T1 @ [x1, y1, 1]
        x2, y2, _ = T1 @ [x2, y2, 1]
        # append now normalized keypoint pairs to normalized_points
        # for correct input format of eight_point_algorithm
        normalized_kp1.append((x1, y1))
        normalized_kp2.append((x2, y2))
    # transform to np.array
    norm1 = np.array(normalized_kp1)
    norm2 = np.array(normalized_kp2)

    return norm1, norm2

def denormlize_points(F_hat, T1, T2):
    """
    Denormalization as in point 3.2.3 in task sheet.
    :param F_hat: return of normalized_eight_point_algorithm
    :param T1: return of normalize_points(im1_keypoints)
    :param T2: -.-
    :return: F
    """
    return T2.T @ F_hat @ T1

def normalized_eight_point_algorithm(key_points1, key_points2, T1, T2, norm=True, denormalize=False):
    """
    Performs normalized version of eight point algorithm with normalized points of
    im1 and im2.
    :param points1:
    :param points2:
    :return: F_hat
    """
    if norm:
        # get normlized points 
        key_points1, key_points2 = normalize_points(key_points1, key_points2, T1, T2)
    # apply eight_point_algorithm from before
    F_hat = eight_point_algorithm(key_points1, key_points2)
    # denormalization not necessarily applied
    if denormalize:
        F = denormlize_points(F_hat, T1, T2)
        return F, F_hat
    else:
        return None, F_hat

def sampson_distance(key_p1, key_p2, F_hat):
    """
    Calculate sampson distance given a pair of key_points according to given formula.
    :param key_points1: normalized coordinates in im1: (x1, y1)
    :param key_points2: normalized coordinates in im2: (x2, y2)
    :param F_hat: transformation Fundamental matrix
    :return: d: sampson distances for each key_point pair
    """
    # numerator:  (p_1^T*F*p2)²
    numerator = np.power(key_p2 @ F_hat@ key_p1, 2)
    F_p1 = np.power(F_hat @ key_p1, 2)
    F_p2 = np.power(F_hat @ key_p2, 2)
    denominator = F_p1[0] + F_p1[1] + F_p2[0] + F_p2[1]
    d = numerator / denominator

    return d

def get_distances(kp1, kp2, F_hat):
    """
    For sampled keypoints calculate sampson distance for all points.
    :param key_points1:
    :param key_points2:
    :param F_hat:
    :return: vector of errors / sampson distances
    """
    errors = np.zeros(len(kp1))
    for i in range(len(kp1)):
        tmp_kp1 = np.array([kp1[i, :][0],kp1[i, :][1], 1])
        tmp_kp2 = np.array([kp2[i, :][0],kp2[i, :][1], 1])
        errors[i] = sampson_distance(tmp_kp1, tmp_kp2, F_hat)
    return errors


def get_keypoints_and_descriptors(im, contrast = 0.06, edge = 10, type="SIFT"):
    """
    Given an image get the sift key_points and descriptors. Either SIFT or ORB
    :param im1:
    :param sift:
    :return: key_points, descriptors both no arrays
    """
    if type =="SIFT":
        # initialize detector with parameters contrast to counteract keypoints in the
        # background. Edge to downweigh edges for keypoints
        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = contrast, edgeThreshold = edge)
        # extract key points and descriptors
        (key_points, descriptors) = sift.detectAndCompute(im, None)
    elif type =="ORB":
        # Initiate ORB detector and BFMatcher
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        (key_points, descriptors) = orb.detectAndCompute(im, None)
    return np.array(key_points), np.array(descriptors)

def get_matches(matcher, descriptors1, descriptors2):
    """
    Return matched key_points from im1 and im2 based on the descriptors by
    applying knn with number of neighbors = 2
    :param matcher:
    :param descriptors1:
    :param descriptors2:
    :return: knn matches
    """
    return matcher.knnMatch(descriptors1, descriptors2, k=2)


def filter_matches(matches, filter_ratio):
    """
    Given a result of the knn matching of descriptors, it filters the matches
    by distance and ratio test described in Lowes SIFT paper
    :param matches:
    :param filter_ratio:
    :return: matched keypoints fulfilling the distance ratio
    """
    filtered_match_points = []
    for match in matches:
        # append to filtered list if distance ratio is fulfilled
        if match[0].distance < match[1].distance * filter_ratio:
            filtered_match_points.append([match[0]])
    return filtered_match_points

def get_matched_keypoint_coordinates(matches, key_points1, key_points2):
    """
    Given a result of filter_matches and before get_matches, the coordinates of the matched
    keypoints being stored in the key_points are returned.
    :param matches: matches = filter_matches(get_matches(matcher, d1, d2), ratio)
    :param key_points1: keypoints of first image
    :param key_points2: keypoints of second image
    :return: point coordinates of matched keypoints as np.arrays
    """
    p1 = []
    p2 = []
    for i in range(len(matches)):
        match = matches[i]
        p1.append(key_points1[match[0].queryIdx].pt)
        p2.append(key_points2[match[0].trainIdx].pt)
    return np.array(p1), np.array(p2)

def transform_image(im, blur=False, gauss_params=None):
    """
    Transform image to gray and maybe blur too to reduce noise for keypoint detectors
    as preparation for keypoint detection.
    :param im: RGB image
    :param blur: boolean if blurring is supposed to be done or not
    :param gauss_params: if not None: one value specifying the standard deviation in both directions
                                      two values if not the same in x and y direction
    :return: transformed image
    """
    # rbg to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    if blur:
        return cv2.GaussianBlur(gray, gauss_params, 0)
    else:
        return gray


def sample_keypoints(points1, points2, n_samples):
    """
    Sample possible keypoints for RANSAC and eight point algorithm and return sample ids as well as according
    keypoints and the complementary non sampled ids for the ransac algorithm.
    :param coordinates1:
    :param coordinates2:
    :return:
    """
    # sample n_samples ids from matched keypoint coordinates
    # replace by default true, so no repeated draws possible
    sample_ids = np.random.choice(points1.shape[0], n_samples, replace=False)
    #print("sample ids:" + str(sample_ids))
    # get coordinates of sampled keypoints in both images
    key_points1_sampled = points1[sample_ids]
    key_points2_sampled = points2[sample_ids]
    # to avoid ''overfitting'', we only evaluate on the points not
    # sampled here -> need ids of not sampled points
    not_sample_ids = list(set(range(points1.shape[0])) - set(sample_ids))
    # respective not sampled points are
    key_points1_not_sampled = points1[not_sample_ids]
    key_points2_not_sampled = points2[not_sample_ids]

    return sample_ids, not_sample_ids, key_points1_sampled, key_points2_sampled, key_points1_not_sampled, key_points2_not_sampled

def RANSAC(points1, points2, n_iteration=1000, threshold=1e-4, n_samples=8):
    """
    Third alternative was RANSAC with normalized eight point algorithm.
    :param key_points:
    :param n_samples:
    :return:
    """
    # initialize lists for best results after n_iterations
    best_fit = 0
    best_error = 0
    best_ids = []

    # Obtain transformation matrix T for normalized eight point
    T1 = get_transformation_matrix(points1)
    T2 = get_transformation_matrix(points2)
    norm_points1, norm_points2  = normalize_points(points1, points2, T1, T2)
    
    for i in range(n_iteration):
        sample_ids, not_sample_ids, _, _, _, _ = sample_keypoints(points1, points2, n_samples)
        # get F_hat as a result of performing the normalized eight point algorithm
        points1_sampled = norm_points1[sample_ids]
        points2_sampled = norm_points2[sample_ids]
        # not sampled normalized points for evaluation
        points1_not_sampled = norm_points1[not_sample_ids]
        points2_not_sampled = norm_points2[not_sample_ids]
        # normalized eight point on the already normalized sampled points
        F, F_hat = normalized_eight_point_algorithm(points1_sampled, points2_sampled, T1, T2, norm=False, denormalize=False)
        # calculate errors / sampson distance of all points
        test_distances = get_distances(points1_not_sampled, points2_not_sampled, F_hat)
        # look for inliers: indices of test_distances that are smaller than threshold
        inlier_ids = np.where(test_distances < threshold)[0]
        # total distances is the sum of test_distances
        total_distances = np.sum(test_distances)
        # if there are more inliers or for same number smaller total distance sum we take the new model
        # as the new best model
        if len(inlier_ids) > best_fit or (len(inlier_ids) == best_fit and total_distances < best_error):
            # replace best value of this iteration
            best_fit = len(inlier_ids)
            # save best ids
            best_ids = inlier_ids
            print("bestids:" + str(best_ids))
            best_error = total_distances
            # get new best keypoints
            best_points1 = np.array(points1_not_sampled[inlier_ids])
            best_points2 = np.array(points2_not_sampled[inlier_ids])
    # apply normalized eight point on best key_points
    F, F_hat_prime = normalized_eight_point_algorithm(best_points1, best_points2, T1, T2, norm=False, denormalize=True)

    return F, F_hat_prime, best_ids

def get_fundamental_matrix(im1, im2, method, n_samples=8, filter_ratio=0.5, n_iteration=1000):
    """
    Procedure to obtain fundamental matrix F_hat with RANSAC algorithm.
    :param im1: rgb image 1
    :param im2: rgb image 2
    :param filter_ratio: value regulating the distance that determines matches
    :param n_iteration: number of iterations of ransac algorithm
    :return: F_hat, best_points1, best_points2 (keypoints in 1 and 2 leading to best model
    """

    # transform images to gray or gray and blur
    img1 = transform_image(im1, blur=True, gauss_params=(3, 3))
    img2 = transform_image(im2, blur=True, gauss_params=(3, 3))
    # get keypoints
    key_points1, descriptors1 = get_keypoints_and_descriptors(img1)
    key_points2, descriptors2 = get_keypoints_and_descriptors(img2)
    print("number of keypoints im1: " + str(len(key_points1)) + "   number of keypoints im2: " + str(len(key_points2)))
    print("keypoints: {}, descriptors: {}".format(len(key_points1), descriptors1.shape))
    print("keypoints: {}, descriptors: {}".format(len(key_points2), descriptors2.shape))
    # initialize matcher
    matcher = cv2.BFMatcher()
    # get matches
    matches = get_matches(matcher, descriptors1, descriptors2)
    # print("number of matches: " + str(len(matches)))
    # filter matches
    matches = filter_matches(matches, filter_ratio)
    print("number of matches after filtering: " + str(len(matches)))
    # extract coordinates of remaining key_points for eight point and ransac algorithm
    points1, points2 = get_matched_keypoint_coordinates(matches, key_points1, key_points2)
    # differentiate per method
    if method == "RANSAC":
        # perform RANSAC on keypoints
        F, F_hat, best_ids = RANSAC(points1, points2, n_iteration=n_iteration)
        left_points1 = points1[best_ids]
        left_points2 = points2[best_ids]
    else:
        # sample key_points, if too many are given
        if len(points1) < n_samples:
            print("too little matched key points to draw from")
            return
        else:
            # sample n_sample keypoints of all the matched keypoints
            _,_,left_points1, left_points2,_,_ = sample_keypoints(points1, points2, n_samples=n_samples)

        if method == "EIGHT":
            # perform eight_point algorithm on the sampled points
            F_hat = eight_point_algorithm(left_points1, left_points2)
            F = None
            # for eight and normeight take all points into consideration for evaluation
            left_points1 = points1
            left_points2 = points2
        elif method == "NORM_EIGHT":
            # Obtain transformation matrix T for sampled keypoints to perform normalized eight point
            T1 = get_transformation_matrix(left_points1)
            T2 = get_transformation_matrix(left_points2)
            # perform normalized eight_point algorithm
            F, F_hat = normalized_eight_point_algorithm(left_points1, left_points2, T1, T2, denormalize=True)
            # return all keypoints not just sampled ones
            left_points1 = points1
            left_points2 = points2
    return F, F_hat, left_points1, left_points2


def draw_epipolar_lines(im1, im2, F, points1, points2):
    """
    We followed the official opencv tutorial:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
    Draws the epipolar lines for the best fundamental matrix F and the keypoints for `img1` and `img2`.
    :param im1:
    :param im2:
    :param F:
    :param points1:
    :param points2:
    :return:
    """
    # transform to grayscale
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
    # cv2.circle needs int points and not float -> type conversion

    points1 = points1.astype(int)
    points2 = points2.astype(int)
    # get epilines for first image that correspond to second image and draw them on the left
    lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
    # from list of lists to list
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = plot_epipolar_line(im1, im2, lines1, points1, points2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = plot_epipolar_line(im2, im1, lines2, points2, points1)
    # plot the two images with the epipolar lines
    plt.subplot(121)
    plt.imshow(img5)
    plt.subplot(122)
    plt.imshow(img3)
    plt.show()


def plot_epipolar_line(im1, im2, lines, points1, points2):
    """
    :param im1: grayscale image
    :param im2: grayscale image
    :param lines:
    :param points1: keypoint coordinates after matching in im1
    :param points2: keypoint coordinates after matching in im2
    :return:
    """
    # shape of left image
    m, n, _ = im1.shape
    for line, point1, point2 in zip(lines, points1, points2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -line[2]/line[1] ])
        x1, y1 = map(int, [n, -(line[2]+line[0]*n)/line[1] ])
        img1 = cv2.line(im1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1, tuple(point1), 5, color, -1)
        img2 = cv2.circle(im2, tuple(point2), 5, color, -1)

    return img1, img2

def plot_keypoints_and_matches(im1, im2, filter_ratio, gauss_blur, contrast, edge):
    """
    For a given filter ratio and two images once pliot the respective keypoints
    and secondly plot the matched keypoints depending on the filter ratio and the
    blurring that is dependent on the gaussian parameter and the contrast and edge parameters of the detector
    :param im1: RGB image
    :param im2: RGB image
    :param filter_ratio: for matching
    :param gauss_blur: window size of gaussian filter
    :return: None, plots are created in the meantime
    """
    # example plot for keypoints
    img1 = transform_image(im1, blur=True, gauss_params=(gauss_blur, gauss_blur))
    img2 = transform_image(im2, blur=True, gauss_params=(gauss_blur, gauss_blur))
    # get keypoints
    key_points1, descriptors1 = get_keypoints_and_descriptors(img1, contrast=contrast, edge=edge)
    key_points2, descriptors2 = get_keypoints_and_descriptors(img2, contrast=contrast, edge=edge)
    # draw keypoints
    kp1 = cv2.drawKeypoints(img1, key_points1, cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    kp2 = cv2.drawKeypoints(img2, key_points2, cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    # draw keypoints of im1 and im2
    plt.subplot(1,2,1)
    plt.imshow(kp1)
    plt.subplot(1, 2, 2)
    plt.imshow(kp2)
    plt.show()
    # cv2.drawMatchesKnn expects list of lists as matches.
    # initialize matcher
    matcher = cv2.BFMatcher()
    # get matches
    matches = get_matches(matcher, descriptors1, descriptors2)
    print("number of matches: " + str(len(matches)))
    # filter matches
    matches = filter_matches(matches, filter_ratio)
    im3 = cv2.drawMatchesKnn(img1, key_points1, img2, key_points2, matches, None, flags=2)
    # plot ,atches
    plt.imshow(im3)
    plt.show()



############################
## Test area
if __name__ == '__main__':
    np.random.seed(42)

    folder_path = "./CV-2-Assignments/assignment_2"
    os.chdir("C:\\Users\\lintl\\Documents\\GitHub\\CV-2-Assignments\\assignment_2")

    # image directory
    imgs_dir = os.path.join("Data")

    # get all image paths
    images_path = []
    for img_path in glob.glob(os.path.join(imgs_dir, '*.png')):
        images_path.append(img_path)
    # sort names
    images_path = sorted(images_path)

    im1_path = images_path[0]
    im2_path = images_path[10]
    # read two images
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)

    # demo keypoints and matches for filterratio
    plot_keypoints_and_matches(im1, im2, filter_ratio=0.3, gauss_blur=3, contrast=0.06, edge=10)

    # demo RANSAC epipolar lines
    F, F_hat, points1, points2 = get_fundamental_matrix(im1, im2, method="RANSAC", filter_ratio=0.2, n_iteration=1000)
    draw_epipolar_lines(im1, im2, F_hat, points1, points2)
    draw_epipolar_lines(im1, im2, F, points1, points2)

    # demo Eight point algorithm epipolar lines
    F, F_hat, points1, points2 = get_fundamental_matrix(im1, im2, method="EIGHT", filter_ratio=0.2)
    draw_epipolar_lines(im1, im2, F_hat, points1, points2)

    # demo Normalized Eight point algorithm epipolar lines
    F, F_hat, points1, points2 = get_fundamental_matrix(im1, im2, method="NORM_EIGHT", filter_ratio=0.2)
    draw_epipolar_lines(im1, im2, F, points1, points2)
