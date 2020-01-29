import numpy as np
from numpy import ndarray
from math import sqrt


def calcRelativeTranslation(first_vector: ndarray, second_vector: ndarray) -> ndarray:
    # translation: [float] = [second_vector[i][0] - first_vector[i][0] for i in range(3)]
    translation: [float] = [second_component - first_component for first_component, second_component in zip(first_vector, second_vector)]
    return np.asarray(translation)


def euclidDistance(first_pose: ndarray, second_pose: ndarray) -> float:
    sqrtcontent = (first_pose[0][0] - second_pose[0][0]) ** 2 + (first_pose[1][0] - second_pose[1][0]) ** 2 + (
                first_pose[2][0] - second_pose[2][0]) ** 2
    return sqrt(sqrtcontent)


def containsNaN(vector: ndarray) -> bool:
    for i in range(0, len(vector) - 1):
        if not vector[i] == vector[i]:
            return True
    return False


def getKepoints():
    file = '../data/keypoints.txt'
    return np.genfromtxt(file, delimiter=' ', dtype=None)


def getTruePose():
    file = '/Users/HJK-BD//Downloads/kitti/poses/00.txt'
    return np.genfromtxt(file, delimiter=' ', dtype=None)


def getLandMarks():
    file = '../data/p_W_landmarks.txt'
    return np.genfromtxt(file, delimiter=' ', dtype=None)
