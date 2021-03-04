import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd

from fundamental_matrix import *


def get_PVM(matcher, keypoints, descriptors, images):
    """Summary
    
    Args:
        matcher (BFmatcher): Description
        images (list): list of "cv2.imread(im)"-images
    
    Returns:
        pvm (2D-np.array): Patch-view matrix
    """
    match_list = []

    image_count = len(images)

    for i in range(0, image_count):
        if i%9==0:
            print("Working on image {}".format(i))
        matches = get_matches(matcher, descriptors[i], descriptors[(i+1)%image_count])
        matches = filter_matches(matches, 0.5)

        for match in matches:
            m = match[0]
            contained = False
            for match_obj in match_list:
                if match_obj.contains(i, m.queryIdx):
                    contained = True
                    match_obj.add((i+1)%image_count, m.trainIdx)
            if not contained:
                match_list.append(Match(i, ((i+1)%image_count), m.queryIdx, m.trainIdx))


    pvm = np.zeros((image_count, len(match_list)))
    for idx, match_obj in enumerate(match_list):
        for im_id in match_obj.im_ids:
            pvm[im_id,idx] = 1

    return pvm, match_list


class Match():
    def __init__(self, im_id1, im_id2, descr_id1, descr_id2):
        """
        An object of type Match() represents matches of different image-pair matchings
        
        Args:
            im_id1 (int): image id of image 1
            im_id2 (int): the descriptorID corresponding to the matched descriptor in image 1
            descr_id1 (int): image id of image 2
            descr_id2 (int): the descriptorID corresponding to the matched descriptor in image 2
        """
        # list of images, which got matched with the patch that initialized this Match-obj
        self.im_ids = [im_id1, im_id2]

        # list of the corresponding descriptors (i-th element of self.im_ids corresponds to 
        # i-th element of self.descr_ids)
        self.descr_ids = [descr_id1, descr_id2]


    def contains(self, im_id, descr_id):
        """
        Tests, if the image with id 'im_id' is contained in the self.im_ids with the
        correspondingand the descriptor.
        
        Args:
            im_id (int): 
            descr_id (int): 
        
        Returns:
            boolean: True if it is contained, false otherwise.
        """
        return (im_id in self.im_ids) and (self.descr_ids[self.im_ids.index(im_id)] == descr_id)

    def add(self, im_id, descr_id):
        """
        Adds the image and the descriptor to the match object.
        
        Args:
            im_id (int):
            descr_id (int):
        """
        self.im_ids.append(im_id)
        self.descr_ids.append(descr_id)


if __name__ == '__main__':


    # image directory
    imgs_dir = os.path.join("Data")
    # get all image paths
    images_path = []
    images = []
    for img_path in glob.glob(os.path.join(imgs_dir, '*.png')):
        images_path.append(img_path)
    # sort names
    images_path = sorted(images_path)


    for idx, im in enumerate(images_path):
        images.append(cv2.imread(im))

    matcher = cv2.BFMatcher()

    keypoints = []
    descriptors = []

    for idx, im in enumerate(images):
        kp, d = get_keypoints_and_descriptors(im)
        keypoints.append(kp)
        descriptors.append(d)

    pvm, match_list = get_PVM(matcher, keypoints, descriptors, images)

    im_match_count = [0 for i in range(49)]
    for match_obj in match_list:
        if 0 in match_obj.im_ids:
            for i in range(49):
                if len(match_obj.im_ids) > i:
                    im_match_count[i] += 1

    print(im_match_count)
    plt.hist(im_match_count)

    example_list = [(i,im_match_count[i]) for i in range(49)] 

    df = pd.DataFrame(example_list, columns=['Number of Images', 'Number of Matches'])
    df.plot(kind='bar', x='Number of Images', y='Number of Matches', title='Number of consecutive Matches for consecutive Image Pairs')


    plt.figure(num=1,figsize=(20,20))    
    plt.matshow(pvm,aspect="auto",fignum=1)
    plt.title("Point-View-Matrix for 49 house images", fontsize=20)
    plt.xlabel("Matches", fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel("Images", fontsize=18)
    plt.yticks(fontsize=16)
    plt.show()