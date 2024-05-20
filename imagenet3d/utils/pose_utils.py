import math

import numpy as np
from scipy.linalg import logm


def getRz(azimuth):
    return np.array([
        [math.cos(azimuth), -math.sin(azimuth), 0],
        [math.sin(azimuth), math.cos(azimuth), 0],
        [0, 0, 1],
    ])


def batch_getRz(azimuth):
    return np.stack([
        np.stack([np.cos(azimuth), -np.sin(azimuth), np.zeros_like(azimuth)], axis=-1),
        np.stack([np.sin(azimuth), np.cos(azimuth), np.zeros_like(azimuth)], axis=-1),
        np.stack([np.zeros_like(azimuth), np.zeros_like(azimuth), np.ones_like(azimuth)], axis=-1)
    ], axis=1)


def getRx(elevation):
    return np.array([
        [1, 0, 0],
        [0, np.cos(elevation), -np.sin(elevation)],
        [0, np.sin(elevation), np.cos(elevation)],
    ])


def batch_getRx(elevation):
    return np.stack([
        np.stack([np.ones_like(elevation), np.zeros_like(elevation), np.zeros_like(elevation)], axis=-1),
        np.stack([np.zeros_like(elevation), np.cos(elevation), -np.sin(elevation)], axis=-1),
        np.stack([np.zeros_like(elevation), np.sin(elevation), np.cos(elevation)], axis=-1)
    ], axis=1)


def getRy(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])


def batch_getRy(theta):
    return np.stack([
        np.stack([np.cos(theta), -np.sin(theta), np.zeros_like(theta)], axis=-1),
        np.stack([np.sin(theta), np.cos(theta), np.zeros_like(theta)], axis=-1),
        np.stack([np.zeros_like(theta), np.zeros_like(theta), np.ones_like(theta)], axis=-1)
    ], axis=1)


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


def batch_cal_rotation_matrix(theta, elevation, azimuth):
    azimuth = -azimuth
    elevation = -(math.pi / 2 - elevation)
    return batch_getRy(theta) @ (batch_getRx(elevation) @ batch_getRz(azimuth))


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


def batch_pose_error(pose1, pose2):
    if isinstance(pose1, dict):
        azimuth1, elevation1, theta1 = (
            pose1["azimuth"],
            pose1["elevation"],
            pose1["theta"])
    else:
        azimuth1, elevation1, theta1 = pose1
    if isinstance(pose2, dict):
        azimuth2, elevation2, theta2 = (
            pose2["azimuth"],
            pose2["elevation"],
            pose2["theta"])
    else:
        azimuth2, elevation2, theta2 = pose2
    batch_anno_matrix1 = batch_cal_rotation_matrix(theta1, elevation1, azimuth1)
    batch_pred_matrix2 = batch_cal_rotation_matrix(theta2, elevation2, azimuth2)

    m = np.transpose(batch_pred_matrix2, (0, 2, 1)) @ batch_anno_matrix1
    m = np.array([logm(m[i]) for i in range(len(m))])
    error_ = (
        (m ** 2).sum(axis=2).sum(axis=1)
    ) ** 0.5 / (2.0 ** 0.5)

    return error_


if __name__ == '__main__':
    a, e, t = 0.0, 0.0, 0.0
    a1, e1, t1 = 1.0, 2.0, 3.0
    print(pose_error((a, e, t), (a1, e1, t1)))

    a, e, t = np.array([a, a, a, a]), np.array([e, e, e, e]), np.array([t, t, t, t])
    a1, e1, t1 = np.array([a1, a1, a1, a1]), np.array([e1, e1, e1, e1]), np.array([t1, t1, t1, t1])
    print(batch_pose_error((a, e, t), (a1, e1, t1)))

    for i in range(1000):
        a1 = np.random.rand(10) * 2 * np.pi
        e1 = np.random.rand(10) * 2 * np.pi
        t1 = np.random.rand(10) * 2 * np.pi
        a2 = np.random.rand(10) * 2 * np.pi
        e2 = np.random.rand(10) * 2 * np.pi
        t2 = np.random.rand(10) * 2 * np.pi

        errors1 = np.array([pose_error((a1[j], e1[j], t1[j]), (a2[j], e2[j], t2[j])) for j in range(10)])
        errors2 = batch_pose_error((a1, e1, t1), (a2, e2, t2))
        assert np.max(np.abs(errors2 - errors1)) < 1e-6
