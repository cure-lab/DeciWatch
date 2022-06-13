# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch

def normalize_screen_coordinates(X, w, h):
    # Normalize pixel-wise 2d pose, so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    if X.shape[-1] == 3: #input 3d pose
        X_norm = X[..., :2]
        X_norm = X_norm / w * 2 - [1, h / w]
        X_out = np.concatenate((X_norm, X[..., 2:3] / 1000), -1)
    else:
        assert X.shape[-1] == 2
        X_out = X / w * 2 - [1, h / w]
    return X_out


def image_coordinates(X, w, h):
    # Reverse normalized 2d poses into pixel-wise 2d poses
    if X.shape[-1] == 3: #input 3d pose
        X_norm = X[..., :2]
        X_norm[..., :1] = (X_norm[..., :1] + 1) * w / 2
        X_norm[..., 1:2] = (X_norm[..., 1:2] + h / w) * w / 2
        X_out = torch.cat([X_norm, X[..., 2:3] * 1000], -1)

    else:
        assert X.shape[-1] == 2
        X_out = (X + [1, h / w]) * w / 2
    return X_out

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return (v + 2 * (q[..., :1] * uv + uuv))


def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)

      
def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
      
      
def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R)  # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t)  # Rotate and translate


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def get_intrinsic(camera_params):
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    fx, fy, cx, cy = camera_params[..., :1], camera_params[..., 1:2], camera_params[..., 2:3], camera_params[..., 3:4]
    return fx, fy, cx, cy


def infer_camera_intrinsics(points2d, points3d):
    """Infer camera instrinsics from 2D<->3D point correspondences."""
    pose2d = points2d.reshape(-1, 2)
    pose3d = points3d.reshape(-1, 3)
    x3d = np.stack([pose3d[:, 0], pose3d[:, 2]], axis=-1)
    x2d = (pose2d[:, 0] * pose3d[:, 2])
    alpha_x, x_0 = list(np.linalg.lstsq(x3d, x2d, rcond=-1)[0].flatten())
    y3d = np.stack([pose3d[:, 1], pose3d[:, 2]], axis=-1)
    y2d = (pose2d[:, 1] * pose3d[:, 2])
    alpha_y, y_0 = list(np.linalg.lstsq(y3d, y2d, rcond=-1)[0].flatten())
    return np.array([alpha_x, x_0, alpha_y, y_0])


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1,
                           keepdim=True)
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c


def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        if type(camera_params) == torch:
            camera_params = camera_params.unsqueeze(1)
        else:
            camera_params = camera_params[:, np.newaxis]

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    XX = X[..., :2] / X[..., 2:]
    # XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    if np.array(XX).any() > 1 or np.array(XX).any() < -1:
        print(np.array(XX).any() > 1 or np.array(XX).any() < -1)
        print('Attention for this pose!!!')
    return f * XX + c


def reprojection(pose_3d, abs_depth, camera):
    """
    :param pose_3d: predicted 3d or normed 3d with pixel unit
    :param abs_depth: absolute depth root Z in the camera coordinate
    :param camera: camera intrinsic parameters
    :return: 3d pose in the camera cooridinate with millimeter unit, root joint: zero-center
    """
    camera = camera.unsqueeze(dim=1).unsqueeze(dim=1)
    cx, cy, fx, fy = camera[:,:,:,2:3], camera[:,:,:,3:4], camera[:,:,:,0:1], camera[:,:,:,1:2]
    final_3d = torch.zeros_like(pose_3d)
    final_3d_x = (pose_3d[:, :, :, 0:1] - cx) / fx
    final_3d_y = (pose_3d[:, :, :, 1:2] - cy) / fy
    final_3d[:, :, :, 0:1] = final_3d_x * abs_depth
    final_3d[:, :, :, 1:2] = final_3d_y * abs_depth
    final_3d[:, :, :, 2:3] = abs_depth
    return final_3d
