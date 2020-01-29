from math import sqrt

import cv2
import numpy as np
from numpy import ndarray

from features import getK, extract_keypoints_gftt, extract_keypoints_orb, featureTrackingORB

# from OxfordWrapper import OxfordWrapper

# Settings
VISUALIZE_CUR_IMAGE: bool = False
VISUALIZE_TRACKING: bool = True
VISUALIZE_STEREO_FEATURES: bool = False
# Use Good features to track or ORB
USE_GFTT: bool = True
# Maximum amount of features Orb should find in each image
MAX_FEATURES: int = 4000
# The number of matches to be considered. Only uses matches with best hamming distance
MAX_MATCHES: int = 400
# Image index from which to start
START_INDEX: int = 0
# Image index after which to stop
STOP_INDEX: int = 5000
# Select KITTI sequence
KITTI_SEQUENCE: str = "05"


def getKepoints():
    file = '../data/keypoints.txt'
    return np.genfromtxt(file, delimiter=' ', dtype=None)


def getTruePose():
    file = '/Users/HJK-BD//Downloads/kitti/poses/00.txt'
    return np.genfromtxt(file, delimiter=' ', dtype=None)


def getLandMarks():
    file = '../data/p_W_landmarks.txt'
    return np.genfromtxt(file, delimiter=' ', dtype=None)


def getLeftImage(i):
    # return cv2.imread('/Users/HJK-BD//Downloads/kitti/00/image_0/{0:06d}.png'.format(i), 0)
    return cv2.imread("/run/media/rudiger/RobotCar/sequences/" + KITTI_SEQUENCE + "/image_2/{0:06d}.png".format(i), 0)


def getRightImage(i):
    # return cv2.imread('/Users/HJK-BD//Downloads/kitti/00/image_1/{0:06d}.png'.format(i), 0)
    return cv2.imread("/run/media/rudiger/RobotCar/sequences/" + KITTI_SEQUENCE + "/image_3/{0:06d}.png".format(i), 0)


def calcRelativeTranslation(first_vector: ndarray, second_vector: ndarray) -> ndarray:
    return np.ndarray([second_vector[i] - first_vector[i] for i in range(2)])


def euclidDistance(first_vector: ndarray, second_vector: ndarray) -> float:
    return sqrt((first_vector[0] - second_vector[0]) ** 2 + (first_vector[1] - second_vector[1]) ** 2 + (
            first_vector[2] - second_vector[2]) ** 2)


def playImageSequence(left_img, right_img, K):
    '''
        different ways to initialize the query points and landmark points
        you can specify the keypoints and landmarks
        or you can inilize_3D with FAST corner points, then stere match and then generate 3D points, but not so accurate
        or you can use the OPENCV feature extraction and matching functions
    '''
    reference_3D, reference_2D, reference_desc = extract_keypoints_gftt(left_img, right_img, K, baseline)
    # points, p1 = extract_keypoints_orb(left_img, right_img, K, baseline)
    reference_2D = reference_2D.astype('float32')

    # reference
    reference_img = left_img
    reference_2D: ndarray
    reference_desc: ndarray
    reference_3D = []
    cur_2D: ndarray
    cur_desc = []
    trajectory_image = np.zeros((600, 600, 3), dtype=np.uint8)
    maxError = 0
    prev_rotation_vector: ndarray
    prev_translation_vector: ndarray
    for i in range(START_INDEX, STOP_INDEX):
        print('image: ', i)
        left_img = getLeftImage(i)
        right_img = getRightImage(i)
        cur_3D, cur_2D, cur_desc = extract_keypoints_gftt(left_img, right_img, K, baseline)
        cur_2D = reference_2D.astype('float32')
        if i == 0:
            translation_vector = np.asarray([0, 0, 0])
            rotation_vector = np.asarray([0, 0, 0])
            reference_3D = cur_3D
        else:
            reference_3D, reference_2D, tracked_2Dpoints, tracked_desc = featureTrackingORB(reference_img, reference_2D,
                                                                                            reference_desc,
                                                                                            cur_2D, cur_desc,
                                                                                            reference_3D)
            # featureTracking(reference_img, left_img, reference_2D, reference_3D)

            pnp_3D_points = np.expand_dims(reference_3D, axis=2)  # 3D points
            pnp_2D_points = np.expand_dims(tracked_2Dpoints, axis=2).astype(float)  # corresponding 2D points
            rotation_vector: ndarray  # rotation angles between two camera poses
            translation_vector: ndarray  # translation between two camera poses
            inliers: ndarray  # output vector containing indices of inliers :in pnp_3D_points and pnp_2D_points
            _, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pnp_3D_points, pnp_2D_points, K, None,
                                                                                 useExtrinsicGuess=False,
                                                                                 rvec=prev_rotation_vector,
                                                                                 tvec=prev_translation_vector)

            # TODO what should happen when Ransac does not find any inliers
            # TODO check projection matrix again to see if that caused the error

            # update the new reference_2D
            reference_2D = tracked_2Dpoints[inliers[:, 0], :]
            reference_3D = reference_3D[inliers[:, 0], :]

            # retrieve the rotation matrix
            rot, _ = cv2.Rodrigues(rotation_vector)  # converts rotation vector to rotation matrix
            translation_vector = -rot.T.dot(translation_vector)  # coordinate transformation, from camera to world

            inv_transform = np.hstack((rot.T, translation_vector))  # inverse transform
            inliers_ratio = len(inliers) / len(pnp_3D_points)  # the inlier ratio

            print('inliers ratio: ', inliers_ratio)

            # re-obtain the 3 D points if the conditions satisfied
            if inliers_ratio < 0.9 or len(reference_2D) < 50:
                # initialization new landmarks
                curImage_R = getRightImage(i)
                # reference_3D, reference_2D = initialize_3D_points(left_img, curImage_R, K, baseline)
                # reference_2D = np.fliplr(reference_2D).astype('float32')
                landmark_3D_new, reference_2D_new = extract_keypoints_gftt(left_img, curImage_R, K, baseline,
                                                                           cur_2D)
                reference_2D_new = reference_2D_new.astype('float32')
                landmark_3D_new = inv_transform.dot(
                    np.vstack((landmark_3D_new.T, np.ones((1, landmark_3D_new.shape[0])))))
                valid_matches = landmark_3D_new[2, :] > 0
                landmark_3D_new = landmark_3D_new[:, valid_matches]

                reference_2D = np.vstack((reference_2D, reference_2D_new[valid_matches, :]))
                reference_3D = np.vstack((reference_3D, landmark_3D_new.T))
            else:
                reference_desc = tracked_desc

        # set reference values
        reference_img = left_img


        if VISUALIZE_CUR_IMAGE:
            cv2.imshow("Current Image", left_img)

        # draw images
        draw_x, draw_y = int(translation_vector[0]) + 300, int(translation_vector[2]) + 100

        text = "Coordinates: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(translation_vector[0]),
                                                                           float(translation_vector[1]),
                                                                           float(translation_vector[2]))
        scaling: float = 1
        try:
            cv2.circle(trajectory_image, (int(draw_x * scaling), int(draw_y * scaling)), 1, (0, 0, 255), 2)
        except:
            print("Something went wrong while drawing trajectory.")

        # cv2.circle(trajectory_image, (int(draw_x*scaling), int(draw_y*scaling)), 1, (0, 0, 255), 2);
        # cv2.circle(trajectory_image, (true_x, true_y), 1, (255, 0, 0), 2);
        cv2.rectangle(trajectory_image, (10, 30), (550, 50), (0, 0, 0), cv2.FILLED);
        cv2.putText(trajectory_image, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8);
        cv2.imshow("Trajectory", trajectory_image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: break

    # cv2.waitKey(0)
    print('Maximum Error: ', maxError)
    cv2.imwrite('map2.png', trajectory_image)


#  imgpts, jac = cv2.projectPoints(pnp_objP, rvec, tvec, K, None)


if __name__ == '__main__':
    left_img = getLeftImage(START_INDEX)
    right_img = getRightImage(START_INDEX)

    # baseline is the distance between both cameras
    baseline = 0.54
    K = getK()
    # for i in range(START_INDEX, STOP_INDEX):
    #     left_img = getLeftImage(i)
    #     right_img = getRightImage(i)
    #     keypoints = extract_keypoints_gftt(left_img, right_img, K, baseline)
    playImageSequence(left_img, right_img, K)
