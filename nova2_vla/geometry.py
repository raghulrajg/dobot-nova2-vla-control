"""Small pose helpers on top of the twin's kinematics (metres, radians
unless a name says _deg; joint vectors are servo degrees)."""

import math

import numpy as np

from .twin import fk, joint_map, pose, rot_x, rot_z


def tool_pose(q_servo_deg):
    """Tool tip pose in the base frame at these servo angles."""
    return fk(joint_map.servo_to_visual(q_servo_deg), "tool")


def camera_pose(q_servo_deg):
    """Wrist camera pose (optical frame: z = view axis) in the base frame."""
    return fk(joint_map.servo_to_visual(q_servo_deg), "camera")


def tool_down(x, y, z, yaw_rad):
    """Tool tip at (x, y, z) pointing straight down, tool x turned yaw_rad
    about the base z. Same as Dobot rx=180, ry=0, rz=yaw."""
    return pose(rot_z(yaw_rad) @ rot_x(math.pi), [x, y, z])


def yaw_of(T):
    """Heading of a frame's x axis about the base z [rad]."""
    return math.atan2(T[1, 0], T[0, 0])


def tilt_deg(T):
    """Angle between the frame's z axis and straight down [deg]."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(T[2, 2])))))


def home_servo_deg():
    return np.asarray(joint_map.home_servo_deg, dtype=float).copy()
