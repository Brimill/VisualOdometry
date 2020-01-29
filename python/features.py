import cv2
import numpy as np
from numpy import ndarray

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
    return queryPoints[:, 0] != 0


def extract_keypoints_gftt(left_image, right_image, K, baseline, refPoints=None):
    left_corners: ndarray = cv2.goodFeaturesToTrack(left_image, 400, 0.1, 1)
    left_kps, left_descriptors = calc_descriptors_gftt(left_corners, left_image)
    right_corners: ndarray = cv2.goodFeaturesToTrack(right_image, 400, 0.1, 1)
    right_kps, right_descriptors = calc_descriptors_gftt(right_corners, right_image)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_matcher.match(left_descriptors, right_descriptors)
    matches = [match for match in matches if match.distance < 25]
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:MAX_MATCHES]
    # print("Max Hamming Distance: " + str(matches[MAX_MATCHES - 1].distance))

    # ratio test as per Lowe's paper
    match_points1, match_points2 = [], []
    match_desc1, match_desc2 = [], []
    for i, match in enumerate(matches):
        match_points1.append(left_kps[match.queryIdx].pt)
        match_points2.append(right_kps[match.trainIdx].pt)
        match_desc1.append(left_descriptors[match.queryIdx])
        match_desc2.append(right_descriptors[match.trainIdx])

    match_desc1 = np.asarray(match_desc1)
    match_desc2 = np.asarray(match_desc2)
    # print('old lengthL', len(match_points1))

    clean_left_points: ndarray = np.array(match_points1).astype(float)
    clean_right_points: ndarray = np.array(match_points2).astype(float)
    clean_left_desc: ndarray
    mask: ndarray = np.empty((0, 0))

    # removes points encountered before... Why would someone do that? This makes tracking a feature impossible
    if refPoints is not None:
        mask = removeDuplicate(clean_left_points, refPoints)
        clean_left_points = clean_left_points[mask, :]
        clean_right_points = clean_right_points[mask, :]
        match_desc1 = match_desc1[mask, :]
        match_desc2 = match_desc2[mask, :]

    # print('new lengthL ', len(clean_left_points))

    if VISUALIZE_STEREO_FEATURES:
        # iterate over matches and remove all features which are not in match or have duplicates
        visualization: ndarray = cv2.drawMatches(left_image, left_kps,
                                                 right_image, right_kps,
                                                 matches, None, matchesMask=mask.tolist())
        height, width, depth = visualization.shape
        imgScale = 1980 / width
        new_height, new_width = visualization.shape[1] * imgScale, visualization.shape[0] * imgScale
        visualization = cv2.resize(visualization, (int(new_height), int(new_width)))
        cv2.imshow("Matches", visualization)
        cv2.waitKey(1)

    # M_left = K.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    # M_right = K.dot(np.hstack((np.eye(3), np.array([[-baseline, 0, 0]]).T)))
    M_left = getMLeft()
    M_right = getMRight()

    flipped_clean_left_points = np.vstack((clean_left_points.T, np.ones((1, clean_left_points.shape[0]))))
    flipped_clean_right_points = np.vstack((clean_right_points.T, np.ones((1, clean_right_points.shape[0]))))

    P = cv2.triangulatePoints(M_left, M_right, flipped_clean_left_points[:2], flipped_clean_right_points[:2])

    P = P / P[3]
    land_points = P[:3]

    return land_points.T, clean_left_points, match_desc1


def extract_keypoints_orb(left_image, right_image, K, baseline, refPoints=None):
    # detector = cv2.xfeatures2d.SURF_create(400)
    detector = cv2.ORB_create(MAX_FEATURES)
    left_features, left_descriptors = detector.detectAndCompute(left_image, None)
    right_features, right_descriptors = detector.detectAndCompute(right_image, None)

    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_matcher.match(left_descriptors, right_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:MAX_MATCHES]
    print("Max Hamming Distance: " + str(matches[MAX_MATCHES - 1].distance))

    # ratio test as per Lowe's paper
    match_points1, match_points2 = [], []
    for i, match in enumerate(matches):
        match_points1.append(left_features[match.queryIdx].pt)
        match_points2.append(right_features[match.trainIdx].pt)

    # print('old lengthL', len(match_points1))

    clean_left_points: ndarray = np.array(match_points1).astype(float)
    clean_right_points: ndarray = np.array(match_points2).astype(float)
    mask: ndarray = np.empty((0, 0))

    # removes points encountered before... Why would someone do that? This makes tracking a feature impossible
    if refPoints is not None:
        mask = removeDuplicate(clean_left_points, refPoints)
        clean_left_points = clean_left_points[mask, :]
        clean_right_points = clean_right_points[mask, :]

    # print('new lengthL ', len(clean_left_points))

    if VISUALIZE_STEREO_FEATURES:
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

    flipped_clean_left_points = np.vstack((clean_left_points.T, np.ones((1, clean_left_points.shape[0]))))
    flipped_clean_right_points = np.vstack((clean_right_points.T, np.ones((1, clean_right_points.shape[0]))))

    P = cv2.triangulatePoints(M_left, M_right, flipped_clean_left_points[:2], flipped_clean_right_points[:2])

    P = P / P[3]
    land_points = P[:3]

    return land_points.T, clean_left_points


def calc_descriptors_gftt(corners: ndarray, img: ndarray) -> (ndarray, ndarray):
    detector = cv2.ORB_create(400)
    keypoints: [cv2.KeyPoint] = []
    for corner_index in range(len(corners)):
        corner = corners[corner_index]
        x, y = corner.ravel()
        keypoints.append(cv2.KeyPoint(x, y, 1))
    return detector.compute(img, np.asarray(keypoints))


def featureTracking(img_t1, img_t2, points_t1, world_points):
    # use KLT tracker
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    points_t2: ndarray  # output vector of 2D points containing calculated new positions of input features in second image
    status: ndarray  # output status vector: if flow for feature has been found the corresponding status is set to 1
    error: ndarray  # output error vector
    # Calculates for two images and a set of features from the first image the corresponding pixel positions
    # in the second image -> Optical Flow for a Sparse feature set
    points_t2, status, error = cv2.calcOpticalFlowPyrLK(img_t1, img_t2, points_t1, None, **lk_params)
    status = status.reshape(status.shape[0])
    # find good one
    # only select features where a the new pixel position has been found
    points_t1 = points_t1[status == 1]
    points_t2 = points_t2[status == 1]
    w_points = world_points[status == 1]
    if VISUALIZE_TRACKING:
        # draw lines between prev points and next points
        # draw the tracks
        frame: ndarray = cv2.cvtColor(img_t1, cv2.COLOR_GRAY2RGB)
        for i, (new, old) in enumerate(zip(points_t1, points_t2)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv2.line(frame, (a, b), (c, d), [19, 252, 3], 2)
            frame = cv2.circle(frame, (a, b), 1, [252, 3, 61], -1)
        cv2.imshow("Tracking", frame)
        cv2.waitKey(1)
    return w_points, points_t1, points_t2


def featureTrackingORB(img_t1: ndarray, kp_t1: ndarray, desc_t1: ndarray, kp_t2: ndarray, desc_t2: ndarray, world_points):
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    track_matches = bf_matcher.match(desc_t1, desc_t2)
    track_matches = [match for match in track_matches if match.distance < 25]
    track_matches = sorted(track_matches, key=lambda x: x.distance)
    track_matches = track_matches[:MAX_MATCHES]
    clean_kp_t1: [cv2.KeyPoint] = []
    clean_desc_t1: [ndarray] = []
    clean_kp_t2: [cv2.KeyPoint] = []
    clean_desc_t2: [ndarray] = []
    w_points = []
    for track_index in range(len(track_matches)):
        match = track_matches[track_index]
        clean_kp_t1.append(kp_t1[match.trainIdx])
        clean_desc_t1.append(desc_t1[match.trainIdx])
        clean_kp_t2.append(kp_t2[match.queryIdx])
        clean_desc_t2.append(desc_t2[match.queryIdx])
        w_points.append(world_points[match.trainIdx])

    if VISUALIZE_TRACKING:
        # draw lines between prev points and next points
        # draw the tracks
        frame: ndarray = cv2.cvtColor(img_t1, cv2.COLOR_GRAY2RGB)
        for i, (new, old) in enumerate(zip(kp_t1, kp_t2)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv2.line(frame, (a, b), (c, d), [19, 252, 3], 2)
            frame = cv2.circle(frame, (a, b), 1, [252, 3, 61], -1)
        cv2.imshow("Tracking", frame)
        cv2.waitKey(1)
    return w_points, clean_kp_t1, clean_kp_t2, clean_desc_t2


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
