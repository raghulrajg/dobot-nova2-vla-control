"""
From a 2D detection to the object's 3D pose in the robot base frame, using
the wrist camera's aligned depth and the twin's camera FK:

    p_base = T_base_camera(q_detect) @ backproject(u, v, depth)

The object is what sticks out of the table inside its box (heights above
table_z, so the table itself and shadows drop out). Its frame:

    origin  centre of its top face (footprint centre, highest 10 % of points)
    z       straight up
    x       across the narrow side -- the direction the fingers close along
            when grasp_yaw_offset_deg is 0 (see the twin's README)
    y       along the long side

size = (length along y, width along x, height above the table).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ..twin import pose


class LocalizationError(RuntimeError):
    pass


@dataclass
class ObjectPose:
    T_base_object: np.ndarray
    length: float
    width: float
    height: float
    n_points: int

    @property
    def top_z(self):
        return float(self.T_base_object[2, 3])

    @property
    def xy(self):
        return self.T_base_object[:2, 3].copy()


def backproject(depth, K, T_base_camera, region=None):
    """Base-frame points (N, 3) and their pixel (v, u) for valid depth
    pixels; region = (x0, y0, x1, y1) limits it to a box."""
    h, w = depth.shape
    x0, y0, x1, y1 = (0, 0, w, h) if region is None else [int(round(v)) for v in region]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 3)), np.zeros((0, 2), int)
    d = depth[y0:y1, x0:x1]
    vv, uu = np.mgrid[y0:y1, x0:x1]
    ok = np.isfinite(d) & (d > 0)
    d, uu, vv = d[ok], uu[ok], vv[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.stack([(uu - cx) * d / fx, (vv - cy) * d / fy, d], axis=1)
    pts = pts @ T_base_camera[:3, :3].T + T_base_camera[:3, 3]
    return pts, np.stack([vv, uu], axis=1)


def height_map(depth, K, T_base_camera, table_z):
    """Height of every pixel above the table [m] (NaN where no depth)."""
    h, w = depth.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    pts, vu = backproject(depth, K, T_base_camera)
    out[vu[:, 0], vu[:, 1]] = pts[:, 2] - table_z
    return out


def localize(box, depth, K, T_base_camera, table_z, mask=None, min_height=0.006, min_points=30):
    """ObjectPose of the object in box (x0, y0, x1, y1), optionally limited
    to mask (HxW bool). Raises LocalizationError when depth can't place it."""
    if depth is None or K is None:
        raise LocalizationError("the wrist camera gives no depth: it is needed to place the object in 3D")
    pts, vu = backproject(depth, K, T_base_camera, box)
    if mask is not None and len(pts):
        keep = mask[vu[:, 0], vu[:, 1]]
        pts = pts[keep]
    heights = pts[:, 2] - table_z if len(pts) else np.zeros(0)
    pts = pts[heights > min_height]
    if len(pts) < min_points:
        raise LocalizationError(f"only {len(pts)} depth points stand above the table inside the detection "
                                "(object too flat, too shiny for the depth camera, or table_z wrong?)")
    h = pts[:, 2] - table_z
    height = float(np.percentile(h, 95))
    # footprint from the upper part: side walls seen at an angle drop out
    upper = pts[h > 0.5 * height]
    if len(upper) < min_points:
        upper = pts
    xy = upper[:, :2]
    # trim stray depth pixels along the principal axes, then fit the
    # minimum-area rectangle: its sides follow the object's edges (PCA alone
    # turns 45 deg on a square)
    mean = xy.mean(axis=0)
    _, evecs = np.linalg.eigh(np.cov((xy - mean).T) if len(xy) > 2 else np.eye(2))
    proj = (xy - mean) @ evecs
    lo, hi = np.percentile(proj, [1, 99], axis=0)
    xy = xy[np.all((proj >= lo) & (proj <= hi), axis=1)]
    (cx, cy), _, _ = rect = cv2.minAreaRect((xy * 1000).astype(np.float32))
    corners = cv2.boxPoints(rect) / 1000.0
    e1, e2 = corners[1] - corners[0], corners[2] - corners[1]
    major, minor_len, major_len = (e1, np.linalg.norm(e2), np.linalg.norm(e1)) \
        if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, np.linalg.norm(e1), np.linalg.norm(e2))
    major = major / np.linalg.norm(major)
    minor = np.array([major[1], -major[0]])          # minor x major = +z (right-handed)
    center = np.array([cx, cy]) / 1000.0
    top_z = table_z + height

    x_axis = np.array([minor[0], minor[1], 0.0])
    y_axis = np.array([major[0], major[1], 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    T = pose(R, [center[0], center[1], top_z])
    return ObjectPose(T, float(major_len), float(minor_len), height, int(len(pts)))
