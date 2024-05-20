# https://github.com/Richard-Guofeng-Zhang/OmniNeMo (May 13, 2024)

from itertools import combinations
import math
import os

import BboxTools as bbt
import numpy as np
import torch

from imagenet3d.utils import bbox_tools as bbt


def cal_point_weight(direct_dict, vert, anno):
    cam_3d = CameraTransformer(anno).get_camera_position()
    vec_ = cam_3d.reshape((1, -1)) - vert
    vec_ = vec_ / (np.sum(vec_ ** 2, axis=1, keepdims=True) ** 0.5)
    matrix_dict = np.array([direct_dict[k] for k in direct_dict.keys()])
    return np.sum(vec_ * matrix_dict, axis=1)


def circle_circonscrit(T):
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4) = T
    A = np.array(
        [
            [x4 - x1, y4 - y1, z4 - z1],
            [x4 - x2, y4 - y2, z4 - z2],
            [x4 - x3, y4 - y3, z4 - z3],
        ]
    )
    Y = np.array(
        [
            (x4 ** 2 + y4 ** 2 + z4 ** 2 - x1 ** 2 - y1 ** 2 - z1 ** 2),
            (x4 ** 2 + y4 ** 2 + z4 ** 2 - x2 ** 2 - y2 ** 2 - z2 ** 2),
            (x4 ** 2 + y4 ** 2 + z4 ** 2 - x3 ** 2 - y3 ** 2 - z3 ** 2),
        ]
    )
    if np.linalg.det(A) == 0:
        return None, 0
    Ainv = np.linalg.inv(A)
    X = 0.5 * np.dot(Ainv, Y)
    x, y, z = X[0], X[1], X[2]
    r = ((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) ** 0.5
    return (x, y, z), r


def l2norm(x, axis=1):
    return x / np.sum(x ** 2, axis=axis, keepdims=True) ** 0.5


def ransac_one(target, points, non_linear_foo=lambda x: x > 0.01):
    # non_linear_foo = lambda x: x
    non_linear_foo = lambda x: np.exp(x)
    all_combinations = np.array(list(combinations(range(points.shape[0]), 3)))

    distances = np.ones(all_combinations.shape[0]) * 100
    centers = np.zeros((all_combinations.shape[0], 3))
    radius = np.zeros(all_combinations.shape[0])
    for i, selection in enumerate(all_combinations):
        selected_points = points[selection]
        center, r = circle_circonscrit(
            np.concatenate((selected_points, np.expand_dims(target, axis=0)), axis=0)
        )
        if center is None:
            continue
        dis_caled = np.sum(
            non_linear_foo(
                np.abs(np.sum((points - np.array([center])) ** 2, axis=1) ** 0.5 - r)
            )
        )

        centers[i] = np.array(center)
        radius[i] = r
        distances[i] = dis_caled
    min_idx = np.argmin(distances)
    center_ = centers[min_idx]
    return l2norm(center_ - target, axis=0)


def direction_calculator(verts, faces):
    out_dict = {i: set() for i in range(verts.shape[0])}

    for t in faces:
        for k in t:
            out_dict[k] = out_dict[k].union(set(t) - {k})

    direct_dict = {}
    for k in out_dict.keys():
        if len(list(out_dict[k])) <= 2:
            direct_dict[k] = np.array([1, 0, 0])
            continue
        # direct_dict[k] = l2norm(np.mean(l2norm(verts[np.array(list(out_dict[k]))] - np.expand_dims(verts[k], axis=0)), axis=0), axis=0)
        direct_dict[k] = ransac_one(verts[k], verts[np.array(list(out_dict[k]))])

    return direct_dict


def get_anno(record, *args, idx=0):
    out = []
    for key_ in args:
        if key_ in ["azimuth", "elevation", "distance", "focal", "theta", "viewport"]:
            out.append(float(record[key_]))
        else:
            out.append(record[key_])
    if len(out) == 1:
        return out[0]
    return tuple(out)


class CameraTransformer:
    parameters_3d_to_2d = (
        "azimuth, elevation, distance, focal, theta, principal, viewport"
    )
    parameters_2d_to_3d = "theta, focal, principal, viewport"
    parameters_transformation_matrix = "azimuth, elevation, distance"
    parameters_camera_polygon = "height, width, theta, focal, principal, viewport"

    def __init__(self, record):
        self.record = record

    def project_points_3d_to_2d(self, x):
        return project_points_3d_to_2d(
            x, *get_anno(self.record, *self.parameters_3d_to_2d.split(", "))
        )

    def project_points_2d_to_3d(self, x):
        return project_points_2d_to_3d(
            x, *get_anno(self.record, *self.parameters_2d_to_3d.split(", "))
        )

    def get_camera_polygon(self):
        return get_camera_polygon(
            *get_anno(self.record, *self.parameters_camera_polygon.split(", "))
        )

    def get_transformation_matrix(self):
        return get_transformation_matrix(
            *get_anno(self.record, *self.parameters_transformation_matrix.split(", "))
        )

    def get_camera_position(self):
        return get_camera_position(self.get_transformation_matrix())


class Projector3Dto2D(CameraTransformer):
    def __call__(self, x):
        return self.project_points_3d_to_2d(x)


class Projector2Dto3D(CameraTransformer):
    def __call__(self, x):
        return self.project_points_2d_to_3d(x)


def project_points_3d_to_2d(
    x3d,
    azimuth,
    elevation,
    distance,
    focal,
    theta,
    principal,
    viewport,
):
    R = get_transformation_matrix(azimuth, elevation, distance)
    if R is None:
        return np.empty(0)

    # perspective project matrix
    # however, we set the viewport to 3000, which makes the camera similar to
    # an affine-camera.
    # Exploring a real perspective camera can be a future work.
    M = viewport
    P = np.array([[M * focal, 0, 0], [0, M * focal, 0], [0, 0, -1]]).dot(R[:3, :4])

    # project
    x3d_ = np.hstack((x3d, np.ones((len(x3d), 1)))).T
    x2d = np.dot(P, x3d_)
    x2d[0, :] = x2d[0, :] / x2d[2, :]
    x2d[1, :] = x2d[1, :] / x2d[2, :]
    x2d = x2d[0:2, :]

    # rotation matrix 2D
    R2d = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    x2d = np.dot(R2d, x2d).T

    # transform to image coordinate
    x2d[:, 1] *= -1
    x2d = x2d + np.repeat(principal[np.newaxis, :], len(x2d), axis=0)

    return x2d


def get_transformation_matrix(azimuth, elevation, distance):
    if distance == 0:
        # return None
        distance = 0.1

    # camera center
    C = np.zeros((3, 1))
    C[0] = distance * math.cos(elevation) * math.sin(azimuth)
    C[1] = -distance * math.cos(elevation) * math.cos(azimuth)
    C[2] = distance * math.sin(elevation)

    # rotate coordinate system by theta is equal to rotating the model by theta
    azimuth = -azimuth
    elevation = -(math.pi / 2 - elevation)

    # rotation matrix
    Rz = np.array(
        [
            [math.cos(azimuth), -math.sin(azimuth), 0],
            [math.sin(azimuth), math.cos(azimuth), 0],
            [0, 0, 1],
        ]
    )  # rotation by azimuth
    Rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(elevation), -math.sin(elevation)],
            [0, math.sin(elevation), math.cos(elevation)],
        ]
    )  # rotation by elevation
    R_rot = np.dot(Rx, Rz)
    R = np.hstack((R_rot, np.dot(-R_rot, C)))
    R = np.vstack((R, [0, 0, 0, 1]))

    return R


def project_points_2d_to_3d(x2d, theta, focal, principal, viewport):
    x2d = x2d.copy()
    # rotate the camera model
    R2d = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    # projection matrix
    M = viewport
    P = np.array(
        [
            [M * focal, 0, 0],
            [0, M * focal, 0],
            [0, 0, -1],
        ]
    )
    x2d -= principal
    x2d[:, 1] *= -1
    x2d = np.dot(np.linalg.inv(R2d), x2d.T).T
    x2d = np.hstack((x2d, np.ones((len(x2d), 1), dtype=np.float64)))
    x2d = np.dot(np.linalg.inv(P), x2d.T).T
    return x2d


def get_camera_polygon(height, width, theta, focal, principal, viewport):
    x0 = np.array([0, 0, 0], dtype=np.float64)

    # project the 3D points
    x = np.array(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ],
        dtype=np.float64,
    )
    x = project_points_2d_to_3d(x, theta, focal, principal, viewport)

    x = np.vstack((x0, x))

    return x


def get_camera_position(projection_matrix):
    pro_ = projection_matrix[0:3, 0:3]
    pro_ = np.linalg.pinv(pro_)

    f_ = projection_matrix[0:3, 3:]

    return np.matmul(pro_, f_).ravel()


def area_triangle(p0, p1, p2):
    return (
        np.abs(
            p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1])
        )
        / 2
    )


def generate_mask_kernel(p0, p1, p2, mask_size, positions, eps=3):
    A = area_triangle(p0, p1, p2)

    A1 = area_triangle(p0, p1, positions.T)
    A2 = area_triangle(p1, p2, positions.T)
    A3 = area_triangle(p0, p2, positions.T)

    return (np.abs(A - (A1 + A2 + A3)) < eps).reshape(mask_size)


def linear_space_solve(posi_, depth_):
    posi_ = np.concatenate([np.transpose(posi_), np.ones((1, posi_.shape[0]))])
    get = np.matmul(depth_.reshape((1, 3)), np.linalg.inv(posi_))
    return lambda x, get=get: np.matmul(x, get[:, 0:2].T) + get[0, 2]


def generate_depth_map_one_triangle(points, depth):
    box = bbt.contain_points(points)
    if box.size < 3 or area_triangle(*points) < 1e-2:
        return np.ones(box.shape, dtype=np.bool_), None
    # box = box.pad(1)
    points -= np.array([box.lu])

    mask_size = box.shape
    x_range = (
        np.ones(mask_size, dtype=np.float32) * np.arange(mask_size[1]).reshape(1, -1)
    ).ravel()
    y_range = (
        np.ones(mask_size, dtype=np.float32) * np.arange(mask_size[0]).reshape(-1, 1)
    ).ravel()
    positions = np.concatenate([y_range.reshape(-1, 1), x_range.reshape(-1, 1)], axis=1)

    mask_ = generate_mask_kernel(*points.tolist(), mask_size, positions)

    depth_map = linear_space_solve(points, depth)(positions).reshape(mask_size)
    depth_map = depth_map * mask_ + 1e10 * np.logical_not(mask_)
    # assert tuple(depth_map.shape) == tuple(box.shape), 'map size: ' + str(tuple(depth_map.shape)) + ' box size: ' + str(tuple(box.shape))
    return depth_map, box


def cal_occ_one_image(points_2d, distance, triangles, image_size, inf_=1e10, eps=1e-3):
    out_depth = np.ones(image_size, dtype=np.float32) * inf_

    # handle the case that points are out of boundary of the image
    points_2d = np.max([np.zeros_like(points_2d), points_2d], axis=0)
    points_2d = np.min(
        [np.ones_like(points_2d) * (np.array([image_size]) - 1), points_2d], axis=0
    )

    for tri_ in triangles:
        points = points_2d[tri_]
        depths = distance[tri_]

        get_map, get_box = generate_depth_map_one_triangle(points, depths)
        if not get_box:
            continue

        get_box.set_boundary(out_depth.shape)

        # assert tem_box.size == get_box.size, str(get_box) + '   ' + str(tem_box) + '  ' + str(points.tolist())
        get_box.assign(
            out_depth,
            np.min([get_map, get_box.apply(out_depth)], axis=0),
            auto_fit=False,
        )

    invalid_parts = out_depth > inf_ * 0.9

    out_depth[invalid_parts] = 0

    visible_distance = out_depth[tuple(points_2d.T.tolist())]
    if_visible = np.abs(distance - visible_distance) < eps
    return if_visible


def load_off(off_file_name, to_torch=False):
    file_handle = open(off_file_name)
    # n_points = int(file_handle.readlines(6)[1].split(' ')[0])
    # all_strings = ''.join(list(islice(file_handle, n_points)))

    file_list = file_handle.readlines()
    n_points = int(file_list[1].split(" ")[0])
    all_strings = "".join(file_list[2 : 2 + n_points])
    array_ = np.fromstring(all_strings, dtype=np.float32, sep="\n")

    all_strings = "".join(file_list[2 + n_points :])
    array_int = np.fromstring(all_strings, dtype=np.int32, sep="\n")

    array_ = array_.reshape((-1, 3))

    if not to_torch:
        return array_, array_int.reshape((-1, 4))[:, 1::]
    else:
        return torch.from_numpy(array_), torch.from_numpy(
            array_int.reshape((-1, 4))[:, 1::]
        )


def normalization(value):
    return (value - value.min()) / (value.max() - value.min())


def box_include_2d(self_box, other):
    return np.logical_and(
        np.logical_and(
            self_box.bbox[0][0] <= other[:, 0], other[:, 0] < self_box.bbox[0][1]
        ),
        np.logical_and(
            self_box.bbox[1][0] <= other[:, 1], other[:, 1] < self_box.bbox[1][1]
        ),
    )


class MeshLoader:
    def __init__(self, path):
        file_list = os.listdir(path)

        l = len(file_list)
        file_list = ["%02d.off" % (i + 1) for i in range(l)]

        self.mesh_points_3d = []
        self.mesh_triangles = []

        for fname in file_list:
            points_3d, triangles = load_off(os.path.join(path, fname))
            self.mesh_points_3d.append(points_3d)
            self.mesh_triangles.append(triangles)

    def __getitem__(self, item):
        return self.mesh_points_3d[item], self.mesh_triangles[item]

    def __len__(self):
        return len(self.mesh_points_3d)


class MeshConverter:
    def __init__(self, path):
        self.loader = MeshLoader(path=path)

    def get_one(self, annos, return_distance=False):
        off_idx = get_anno(annos, "cad_index")

        points_3d, triangles = self.loader[off_idx - 1]
        points_2d = Projector3Dto2D(annos)(points_3d).astype(np.int32)
        points_2d = np.flip(points_2d, axis=1)
        cam_3d = CameraTransformer(
            annos
        ).get_camera_position()  #  @ np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])

        distance = np.sum((-points_3d - cam_3d.reshape(1, -1)) ** 2, axis=1) ** 0.5
        distance_ = normalization(distance)
        h, w = get_anno(annos, "height", "width")
        map_size = (h, w)

        if_visible = cal_occ_one_image(
            points_2d=points_2d,
            distance=distance_,
            triangles=triangles,
            image_size=map_size)

        # box_ori = bbt.from_numpy(get_anno(annos, "box_ori"))
        # box_cropped = bbt.from_numpy(get_anno(annos, "box_obj").astype(np.int32))
        # box_cropped.set_boundary(
        #     get_anno(annos, "box_obj").astype(np.int32)[4::].tolist()
        # )
        # projection_foo = bbt.projection_function_by_boxes(box_ori, box_cropped)
        # pixels_2d = projection_foo(points_2d)

        # if_visible = np.logical_and(if_visible, box_include_2d(box_ori, points_2d))

        # handle the case that points are out of boundary of the image
        # pixels_2d = np.max([np.zeros_like(pixels_2d), pixels_2d], axis=0)
        # pixels_2d = np.min(
        #     [
        #         np.ones_like(pixels_2d) * (np.array([box_cropped.boundary]) - 1),
        #         pixels_2d,
        #     ],
        #     axis=0,
        # )
        pixels_2d = points_2d

        if return_distance:
            return pixels_2d, if_visible, distance_

        return pixels_2d, if_visible
