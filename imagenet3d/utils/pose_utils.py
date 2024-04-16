import math

import numpy as np
from scipy.linalg import logm


def getRz(azimuth):
    return np.array([
        [math.cos(azimuth), -math.sin(azimuth), 0],
        [math.sin(azimuth), math.cos(azimuth), 0],
        [0, 0, 1],
    ])


def getRx(elevation):
    return np.array([
        [1, 0, 0],
        [0, math.cos(elevation), -math.sin(elevation)],
        [0, math.sin(elevation), math.cos(elevation)],
    ])


def getRy(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])


def get_transformation_matrix(azimuth, elevation, distance):
    # camera center
    C = np.zeros((3, 1))
    C[0] = distance * math.cos(elevation) * math.sin(azimuth)
    C[1] = -distance * math.cos(elevation) * math.cos(azimuth)
    C[2] = distance * math.sin(elevation)

    # rotate coordinate system by theta is equal to rotating the model by theta
    azimuth = -azimuth
    elevation = -(math.pi / 2 - elevation)

    # rotation matrix
    Rz = getRz(azimuth)
    Rx = getRx(elevation)
    R_rot = np.dot(Rx, Rz)
    R = np.hstack((R_rot, np.dot(-R_rot, C)))
    R = np.vstack((R, [0, 0, 0, 1]))

    return R


def cal_rotation_matrix(theta, elevation, azimuth):
    azimuth = -azimuth
    elevation = -(math.pi / 2 - elevation)
    return getRy(theta) @ np.dot(getRx(elevation), getRz(azimuth))


def pose_error(pose1, pose2):
    if isinstance(pose1, dict):
        azimuth1, elevation1, theta1 = (
            float(pose1["azimuth"]),
            float(pose1["elevation"]),
            float(pose1["theta"]))
    else:
        azimuth1, elevation1, theta1 = pose1
    if isinstance(pose2, dict):
        azimuth2, elevation2, theta2 = (
            float(pose2["azimuth"]),
            float(pose2["elevation"]),
            float(pose2["theta"]))
    else:
        azimuth2, elevation2, theta2 = pose2
    anno_matrix = cal_rotation_matrix(theta1, elevation1, azimuth1)
    pred_matrix = cal_rotation_matrix(theta2, elevation2, azimuth2)
    if (
        np.any(np.isnan(anno_matrix))
        or np.any(np.isnan(pred_matrix))
        or np.any(np.isinf(anno_matrix))
        or np.any(np.isinf(pred_matrix))
    ):
        error_ = np.pi
    else:
        error_ = (
            (logm(np.dot(np.transpose(pred_matrix), anno_matrix)) ** 2).sum()
        ) ** 0.5 / (2.0 ** 0.5)
    return error_
