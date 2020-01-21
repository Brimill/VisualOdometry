import sys

import numpy as np
from numpy import ndarray
import cv2
from numpy.linalg import inv

# from OxfordWrapper import OxfordWrapper

# Settings
VISUALIZE: bool = True
# Maximum amount of features Orb should find in each image
MAX_FEATURES: int = 1000
# Threshold for hamming distance between matched features
HAMMING_THRESHOLD: int = 30


# def getK():
#     return np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
#                      [0, 7.188560000000e+02, 1.852157000000e+02],
#                      [0, 0, 1]])

def getK():
    return np.array([[7.215377000000e+02, 0, 6.095593000000e+02],
                     [0, 7.215377000000e+02, 1.728540000000e+02],
                     [0, 0, 1]])


def getMRight():
    return np.array([[7.215377000000e+02, 0, 6.095593000000e+02, -3.875744000000e+02],
                     [0, 7.215377000000e+02, 1.728540000000e+02, 0],
                     [0, 0, 1, 0]])


def getMLeft():
    return np.array([[7.215377000000e+02, 0, 6.095593000000e+02, 4.485728000000e+01],
                     [0, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01],
                     [0, 0, 1, 2.745884000000e-03]])


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
    return cv2.imread("/run/media/rudiger/RobotCar/sequences/03/image_2/{0:06d}.png".format(i), 0)


def getRightImage(i):
    # return cv2.imread('/Users/HJK-BD//Downloads/kitti/00/image_1/{0:06d}.png'.format(i), 0)
    return cv2.imread("/run/media/rudiger/RobotCar/sequences/03/image_3/{0:06d}.png".format(i), 0)


def featureDetection(img, numCorners):
    h, w = img.shape
    thresh = dict(threshold=24, nonmaxSuppression=True)
    fast = cv2.FastFeatureDetector_create(**thresh)
    kp1 = fast.detect(img)
    kp1 = sorted(kp1, key=lambda x: x.response, reverse=True)[:numCorners]

    p1 = np.array([ele.pt for ele in kp1], dtype='int')
    # img3 = cv2.drawKeypoints(img, kp1, None, color=(255,0,0))
    # cv2.imshow('fast',img3)
    # cv2.waitKey(0) & 0xFF
    return p1


def featureTracking(img_1, img_2, p1, world_points):
    ##use KLT tracker
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p2: ndarray # output vector of 2D points containing calculated new positions of input features in second image
    status: ndarray # output status vector: if flow for feature has been found the corresponding status is set to 1
    error: ndarray # output error vector
    # Calculates for two images and a set of features from the first image the corresponding pixel positions
    # in the second image -> Optical Flow for a Sparse feature set
    p2, status, error = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
    status = status.reshape(status.shape[0])
    # find good one
    # only select features where a the new pixel position has been found
    p1 = p1[status == 1]
    p2 = p2[status == 1]
    w_points = world_points[status == 1]
    return w_points, p1, p2


def stereo_match_feature(left_img, right_img, patch_radius, keypoints, min_disp, max_disp):
    # in case you want to find stereo match by yourself
    h, w = left_img.shape
    num_points = keypoints.shape[0]

    # Depth (or disparity) map
    depth = np.zeros(left_img.shape, np.uint8)
    output = np.zeros(keypoints.shape, dtype='int')
    all_index = np.zeros((keypoints.shape[0], 1), dtype='int').reshape(-1)

    r = patch_radius
    # patch_size = 2*patch_radius + 1;

    for i in range(num_points):

        row, col = keypoints[i, 0], keypoints[i, 1]
        # print(row, col)
        best_offset = 0;
        best_score = float('inf');
        if row - r < 0 or row + r >= h or col - r < 0 or col + r >= w: continue
        left_patch = left_img[(row - r):(row + r + 1), (col - r):(col + r + 1)]  # left imag patch
        all_index[i] = 1

        for offset in range(min_disp, max_disp + 1):

            if (row - r) < 0 or row + r >= h or (col - r - offset) < 0 or (col + r - offset) >= w: continue

            diff = left_patch - right_img[(row - r):(row + r + 1), (col - r - offset):(col + r - offset + 1)]
            sum_s = np.sum(diff ** 2)

            if sum_s < best_score:
                best_score = sum_s
                best_offset = offset

        output[i, 0], output[i, 1] = row, col - best_offset

    return output, all_index


def generate3D(featureL, featureR, K, baseline):
    # points should be 3xN and intensities 1xN, where N is the amount of pixels
    # which have a valid disparity. I.e., only return points and intensities
    # for pixels of left_img which have a valid disparity estimate! The i-th
    # intensity should correspond to the i-th point.

    temp = featureL - featureR
    temp = temp[:, 1]

    print(featureL.shape, featureR.shape)

    px_left = np.vstack((featureL.T, np.ones((1, featureL.shape[0]))))
    # Switch from (row, col, 1) to (u, v, 1)
    px_left[0:2, :] = np.flipud(px_left[0:2, :])

    bv_left = inv(K).dot(px_left)

    f = K[0, 0]

    z = f * baseline / temp
    points = bv_left * z

    # intensities = left_img.reshape(-1)[disp_im > 0]

    return points


def removeDuplicate(queryPoints, refPoints, radius=5):
    # remove duplicate points from new query points,
    for i in range(len(queryPoints)):
        query = queryPoints[i]
        xliml, xlimh = query[0] - radius, query[0] + radius
        yliml, ylimh = query[1] - radius, query[1] + radius
        inside_x_lim_mask = (refPoints[:, 0] > xliml) & (refPoints[:, 0] < xlimh)
        curr_kps_in_x_lim = refPoints[inside_x_lim_mask]

        if curr_kps_in_x_lim.shape[0] != 0:
            inside_y_lim_mask = (curr_kps_in_x_lim[:, 1] > yliml) & (curr_kps_in_x_lim[:, 1] < ylimh)
            curr_kps_in_x_lim_and_y_lim = curr_kps_in_x_lim[inside_y_lim_mask, :]
            if curr_kps_in_x_lim_and_y_lim.shape[0] != 0:
                queryPoints[i] = np.array([0, 0])
    return (queryPoints[:, 0] != 0)


def initialize_3D_points(left_img, right_img, K, baseline):
    p1 = featureDetection(left_img, 500)
    p1 = np.fliplr(p1)
    # img_show  = cv2.imread('../data/left/{0:06d}.png'.format(0))
    p2, all_index = stereo_match_feature(left_img, right_img, 5, p1, 5, 50)

    p1 = p1[all_index > 0, :]
    p2 = p2[all_index > 0, :]

    M_left = K.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    M_rght = K.dot(np.hstack((np.eye(3), np.array([[-baseline, 0, 0]]).T)))

    p1_flip = np.vstack((np.flipud(p1.T), np.ones((1, p1.shape[0]))))
    p2_flip = np.vstack((np.flipud(p2.T), np.ones((1, p1.shape[0]))))

    # for p in p1:
    #     cv2.circle(img_show, (p[1], p[0]) ,1, (0,0,255), 2);

    P = cv2.triangulatePoints(M_left, M_rght, p1_flip[:2], p2_flip[:2])

    P = P / P[3]
    points = P[:3]

    # for p in p1:
    #     cv2.circle(left_img, (p[0], p[1]) ,1, (0,0,255), 2);

    # cv2.imshow('images', img_show)
    # k = cv2.waitKey(0) & 0xFF
    # print(points.T)
    return points.T, p1


def extract_keypoints_orb(left_image, right_image, K, baseline, refPoints=None):
    # detector = cv2.xfeatures2d.SURF_create(400)
    detector = cv2.ORB_create(MAX_FEATURES)
    left_features, left_descriptors = detector.detectAndCompute(left_image, None)
    right_features, right_descriptors = detector.detectAndCompute(right_image, None)

    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_matcher.match(left_descriptors, right_descriptors)
    matches = [match for match in matches if match.distance < HAMMING_THRESHOLD]
    print("Matches left over: " + str(len(matches)))

    # ratio test as per Lowe's paper
    match_points1, match_points2 = [], []
    for i, match in enumerate(matches):
        match_points1.append(left_features[match.queryIdx].pt)
        match_points2.append(right_features[match.trainIdx].pt)

    print('old lengthL', len(match_points1))

    p1 = np.array(match_points1).astype(float)
    p2 = np.array(match_points2).astype(float)
    mask: ndarray = np.empty((0, 0))
    # removes points encountered before... Why would someone do that? This makes tracking a feature impossible
    if refPoints is not None:
        mask = removeDuplicate(p1, refPoints)
        p1 = p1[mask, :]
        p2 = p2[mask, :]

    print('new lengthL ', len(p1))

    if VISUALIZE:
        # iterate over matches and remove all features which are not in match or have duplicates
        visualization: ndarray = cv2.drawMatches(left_image, left_features,
                                                 right_image, right_features,
                                                 matches, None, matchesMask=mask.tolist())
        height, width, depth = visualization.shape
        imgScale = 1980 / width
        new_height, new_width = visualization.shape[1] * imgScale, visualization.shape[0] * imgScale
        visualization = cv2.resize(visualization, (int(new_height), int(new_width)))
        cv2.imshow("Matches", visualization)
        cv2.waitKey(1)

    M_left = K.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    M_right = K.dot(np.hstack((np.eye(3), np.array([[-baseline, 0, 0]]).T)))

    p1_flip = np.vstack((p1.T, np.ones((1, p1.shape[0]))))
    p2_flip = np.vstack((p2.T, np.ones((1, p2.shape[0]))))

    P = cv2.triangulatePoints(M_left, M_right, p1_flip[:2], p2_flip[:2])

    P = P / P[3]
    land_points = P[:3]

    return land_points.T, p1


def playImageSequence(left_img, right_img, K):
    '''
        different ways to initialize the query points and landmark points
        you can specify the keypoints and landmarks
        or you can inilize_3D with FAST corner points, then stere match and then generate 3D points, but not so accurate
        or you can use the OPENCV feature extraction and matching functions
    '''

    # p1 = getKepoints().astype('float32')

    # print(p1)

    # points = getLandMarks()

    # points, p1 = initialize_3D_points(left_img, right_img, K, baseline)
    # points = points.T
    # p1 = np.fliplr(p1).astype('float32')
    # print(points.shape)
    # print(p1.shape)

    points, p1 = extract_keypoints_orb(left_img, right_img, K, baseline)
    p1 = p1.astype('float32')

    pnp_3D_points = np.expand_dims(points, axis=2)
    pnp_p1 = np.expand_dims(p1, axis=2).astype(float)

    # reference
    reference_img = left_img
    reference_2D = p1
    landmark_3D = points
    # _, rotation_vector, translation_vector = cv2.solvePnP(pnp_3D_points, pnp_p1, K, None)
    # truePose = getTruePose()
    trajectory_image = np.zeros((600, 600, 3), dtype=np.uint8);
    maxError = 0

    for i in range(0, 2000):
        print('image: ', i)
        curImage = getLeftImage(i)
        landmark_3D, reference_2D, tracked_2Dpoints = featureTracking(reference_img, curImage, reference_2D,
                                                                      landmark_3D)

        # print(len(landmark_3D), len(valid_land_mark))
        pnp_3D_points = np.expand_dims(landmark_3D, axis=2) # 3D points
        pnp_2D_points = np.expand_dims(tracked_2Dpoints, axis=2).astype(float) # corresponding 2D points
        rotation_vector: ndarray # rotation angles between two camera poses
        translation_vector: ndarray # translation between two camera poses
        inliers: ndarray # output vector containing indices of inliers in pnp_3D_points and pnp_2D_points
        _, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pnp_3D_points, pnp_2D_points, K, None)

        # update the new reference_2D
        reference_2D = tracked_2Dpoints[inliers[:, 0], :]
        landmark_3D = landmark_3D[inliers[:, 0], :]

        # retrieve the rotation matrix
        rot, _ = cv2.Rodrigues(rotation_vector) # converts rotation vector to rotation matrix
        translation_vector = -rot.T.dot(translation_vector)  # coordinate transformation, from camera to world

        inv_transform = np.hstack((rot.T, translation_vector))  # inverse transform
        inliers_ratio = len(inliers) / len(pnp_3D_points)  # the inlier ratio

        print('inliers ratio: ', inliers_ratio)

        # re-obtain the 3 D points if the conditions satisfied
        if inliers_ratio < 0.9 or len(reference_2D) < 50:
            # initialization new landmarks
            curImage_R = getRightImage(i)
            # landmark_3D, reference_2D = initialize_3D_points(curImage, curImage_R, K, baseline)
            # reference_2D = np.fliplr(reference_2D).astype('float32')
            landmark_3D_new, reference_2D_new = extract_keypoints_orb(curImage, curImage_R, K, baseline, reference_2D)
            reference_2D_new = reference_2D_new.astype('float32')
            landmark_3D_new = inv_transform.dot(np.vstack((landmark_3D_new.T, np.ones((1, landmark_3D_new.shape[0])))))
            valid_matches = landmark_3D_new[2, :] > 0
            landmark_3D_new = landmark_3D_new[:, valid_matches]

            reference_2D = np.vstack((reference_2D, reference_2D_new[valid_matches, :]))
            landmark_3D = np.vstack((landmark_3D, landmark_3D_new.T))

        reference_img = curImage
        cv2.imshow("Current Image", reference_img)

        # draw images
        draw_x, draw_y = int(translation_vector[0]) + 300, int(translation_vector[2]) + 100
        # true_x, true_y = int(truePose[i][3]) + 300, int(truePose[i][11]) + 100

        # curError = np.sqrt(
        #     (translation_vector[0] - truePose[i][3]) ** 2 + (translation_vector[1] - truePose[i][7]) ** 2 + (translation_vector[2] - truePose[i][11]) ** 2)
        # print('Current Error: ', curError)
        # if (curError > maxError):
        #     maxError = curError

        # print([truePose[i][3], truePose[i][7], truePose[i][11]])

        text = "Coordinates: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(translation_vector[0]), float(translation_vector[1]),
                                                                           float(translation_vector[2]));
        scaling: float = 1
        try:
            cv2.circle(trajectory_image, (int(draw_x * scaling), int(draw_y * scaling)), 1, (0, 0, 255), 2);
        except:
            print("Something went wrong while drawing trajectory.");
        finally:
            print("x: " + str(translation_vector[0]))
            print("y: " + str(translation_vector[2]))

        # cv2.circle(trajectory_image, (int(draw_x*scaling), int(draw_y*scaling)), 1, (0, 0, 255), 2);
        # cv2.circle(trajectory_image, (true_x, true_y), 1, (255, 0, 0), 2);
        cv2.rectangle(trajectory_image, (10, 30), (550, 50), (0, 0, 0), cv2.FILLED);
        cv2.putText(trajectory_image, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8);
        cv2.imshow("Trajectory", trajectory_image);
        k = cv2.waitKey(1) & 0xFF
        if k == 27: break

    # cv2.waitKey(0)
    print('Maximum Error: ', maxError)
    cv2.imwrite('map2.png', trajectory_image);


#  imgpts, jac = cv2.projectPoints(pnp_objP, rvec, tvec, K, None)


if __name__ == '__main__':
    # oxfordWrapper: OxfordWrapper = OxfordWrapper("/run/media/rudiger/RobotCar", "2015-11-10", "14-15-57", 400)
    # left_img, right_img = oxfordWrapper.getNextImages()
    # left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    # right_img = cv2. cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    # for i in range (0,60000):
    #     print('/Users/HJK-BD//Downloads/kitti/00/image_0/{0:06d}.png'.format(i))
    left_img = getLeftImage(0)
    right_img = getRightImage(0)

    # baseline is the distance between both cameras
    baseline = 0.54;
    K = getK()

    playImageSequence(left_img, right_img, K)