"""
A simulated cell for --dry-run and the tests: boxes on a table, a wrist
camera rendered (RGB + depth) from the fake controller's joints through the
twin's camera FK, a gripper that holds a box when its fingers close around
one, and a stand-in for a trained VLA.

The fake controller (robot/fake.py) is the real TCP protocol, so a dry run
drives the same code that drives the Nova 2.
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .geometry import camera_pose, tool_down, tool_pose, yaw_of
from .perception.cameras import Frame, intrinsics_from_fov
from .twin import GripResult, IKError, pose, rot_z, solve_ik


@dataclass
class SimBox:
    name: str
    x: float
    y: float
    length: float          # along the box's local y
    width: float           # along local x
    height: float
    yaw_deg: float
    color: tuple           # RGB 0..255
    bottom: float = 0.0    # set by the world
    held_offset: Optional[np.ndarray] = None   # box pose in the tool frame while held

    def T(self):
        return pose(rot_z(math.radians(self.yaw_deg)), [self.x, self.y, self.bottom])


DEMO_BOXES = [
    SimBox("red_block", -0.46, 0.06, 0.045, 0.030, 0.030, 20.0, (200, 30, 30)),
    SimBox("blue_block", -0.36, 0.18, 0.050, 0.050, 0.025, 0.0, (30, 60, 200)),
    SimBox("green_bar", -0.50, 0.20, 0.080, 0.022, 0.020, -35.0, (30, 170, 60)),
]


@dataclass
class SimWorld:
    table_z: float
    boxes: List[SimBox] = field(default_factory=list)
    joints: callable = None          # () -> current servo deg of the (fake) robot

    def __post_init__(self):
        for b in self.boxes:
            b.bottom = self.table_z

    def box(self, name):
        return next(b for b in self.boxes if b.name == name)

    def update_held(self):
        T_tool = tool_pose(self.joints())
        for b in self.boxes:
            if b.held_offset is not None:
                T = T_tool @ b.held_offset
                b.x, b.y, b.bottom = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
                b.yaw_deg = math.degrees(yaw_of(T))

    # ------------------------------------------------------------ rendering
    def render(self, T_base_camera, K, width, height, rng=None):
        self.update_held()
        vv, uu = np.mgrid[0:height, 0:width].astype(np.float64)
        d_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
        R, o = T_base_camera[:3, :3], T_base_camera[:3, 3]
        d = d_cam @ R.T                                     # world ray per pixel; camera z component = 1
        depth = np.full((height, width), np.inf)
        rgb = np.zeros((height, width, 3), np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.table_z - o[2]) / d[..., 2]
        hit = np.isfinite(t) & (t > 0)
        p = o + t[..., None] * d
        checker = ((np.floor(p[..., 0] / 0.03) + np.floor(p[..., 1] / 0.03)) % 2)[..., None]
        table_rgb = 150 + 25 * checker
        rgb = np.where(hit[..., None], table_rgb, rgb)
        depth = np.where(hit, t, depth)

        for b in sorted(self.boxes, key=lambda b: b.bottom):
            top = b.bottom + b.height
            with np.errstate(divide="ignore", invalid="ignore"):
                tb = (top - o[2]) / d[..., 2]
            p = o + tb[..., None] * d
            c, s = math.cos(math.radians(b.yaw_deg)), math.sin(math.radians(b.yaw_deg))
            lx = (p[..., 0] - b.x) * c + (p[..., 1] - b.y) * s
            ly = -(p[..., 0] - b.x) * s + (p[..., 1] - b.y) * c
            inside = (np.abs(lx) <= b.width / 2) & (np.abs(ly) <= b.length / 2) & (tb > 0) & (tb < depth)
            stripes = 0.85 + 0.15 * ((np.floor(ly / 0.008) % 2)[..., None])
            rgb = np.where(inside[..., None], np.array(b.color) * stripes, rgb)
            depth = np.where(inside, tb, depth)

        rng = rng or np.random.default_rng(0)
        depth = np.where(np.isfinite(depth), depth + rng.normal(0, 0.0007, depth.shape), 0.0)
        rgb = np.clip(rgb + rng.normal(0, 3, rgb.shape), 0, 255).astype(np.uint8)
        return rgb, depth.astype(np.float32)

    def reference_image(self, box: SimBox, size=160):
        """A top-down picture of one box alone, for its objects/<name>/ folder."""
        img = np.full((size, size, 3), 128, np.uint8)     # as capture-refs saves them: background greyed out
        px = size / 0.1                                      # 10 cm field
        c, s = math.cos(math.radians(box.yaw_deg)), math.sin(math.radians(box.yaw_deg))
        vv, uu = np.mgrid[0:size, 0:size]
        x = (uu - size / 2) / px
        y = (vv - size / 2) / px
        lx, ly = x * c + y * s, -x * s + y * c
        inside = (np.abs(lx) <= box.width / 2) & (np.abs(ly) <= box.length / 2)
        stripes = 0.85 + 0.15 * ((np.floor(ly / 0.008) % 2)[..., None])
        img = np.where(inside[..., None], (np.array(box.color) * stripes).astype(np.uint8), img)
        return img.astype(np.uint8)


class SimCamera:
    """Renders on every read (no thread): what the wrist camera would see now."""

    def __init__(self, name, cfg, world: SimWorld):
        self.name, self.cfg, self.world = name, cfg, world
        self.K = intrinsics_from_fov(cfg.width, cfg.height)
        self.rng = np.random.default_rng(1)

    def start(self):
        return self

    def stop(self):
        pass

    def read(self, newer_than=None, timeout=2.0):
        if self.cfg.mount == "wrist":
            T = camera_pose(self.world.joints())
        else:   # a fixed camera above the table, looking down
            T = pose(np.diag([1.0, -1.0, -1.0]), [-0.42, 0.12, self.world.table_z + 0.9])
        rgb, depth = self.world.render(T, self.K, self.cfg.width, self.cfg.height, self.rng)
        return Frame(rgb, depth if self.cfg.depth else None, self.K.copy(), time.monotonic())


class SimGripper:
    """Feetech-like gripper in the sim world: closing around a box holds it."""

    def __init__(self, world: SimWorld, max_opening=0.08):
        self.world, self.max_opening = world, max_opening
        self.held: Optional[SimBox] = None
        self.events = []

    def open(self, speed=100, wait=True):
        if self.held is not None:
            self.world.update_held()
            self.held.held_offset = None
            self.held.bottom = max(self.world.table_z, self.held.bottom)   # rests where it is let go
            self.held = None
        self.events.append("open")

    def close(self, speed=50):
        self.events.append("close")
        T = tool_pose(self.world.joints())
        tip = T[:3, 3]
        for b in self.world.boxes:
            c, s = math.cos(math.radians(b.yaw_deg)), math.sin(math.radians(b.yaw_deg))
            lx = (tip[0] - b.x) * c + (tip[1] - b.y) * s
            ly = -(tip[0] - b.x) * s + (tip[1] - b.y) * c
            # finger closing axis = tool x; it must lie across the box's narrow side within 20 deg
            ang = math.degrees(abs(((yaw_of(T) - math.radians(b.yaw_deg)) + math.pi / 2) % math.pi - math.pi / 2))
            span = b.width if ang < 20 else (b.length if abs(ang - 90) < 20 else None)
            if (span is not None and span < self.max_opening and abs(lx) < b.width / 2 and abs(ly) < b.length / 2
                    and b.bottom < tip[2] < b.bottom + b.height):
                self.held = b
                b.held_offset = np.linalg.inv(T) @ b.T()
                pos = int(255 * (1 - span / self.max_opening))
                return GripResult(holding=True, position=pos, raw=2048 + pos * 4, load=250)
        return GripResult(holding=False, position=255, raw=3052, load=20)

    def disconnect(self):
        pass


class MockGraspPolicy:
    """Stands in for a trained VLA in dry runs. It reads the true pose of
    the box under the tool from the sim world, steps the tool down to it in
    1 cm moves, closes, and lifts -- emitting joint targets exactly like a
    VLA, so the cell's control loop, safety filter and ServoJ streaming are
    exercised for real."""

    def __init__(self, world: SimWorld, horizon=15, step_m=0.01):
        self.world, self.horizon, self.step_m = world, horizon, step_m

    def reset(self):
        pass

    def predict_chunk(self, images, state, task):
        q = np.asarray(state[:6], float)
        T = tool_pose(q)
        free = [b for b in self.world.boxes if b.held_offset is None]
        b = min(free, key=lambda b: math.hypot(b.x - T[0, 3], b.y - T[1, 3]))
        holding = state[6] > 0.05
        # fingers across the narrow side, the turn nearest the current wrist
        yaw = math.radians(b.yaw_deg)
        yaw = min((yaw, yaw + math.pi), key=lambda a: abs((a - yaw_of(T) + math.pi) % (2 * math.pi) - math.pi))
        grasp = np.array([b.x, b.y, b.bottom + b.height - min(0.02, b.height / 2)])
        out, q_ref, p = [], q, T[:3, 3].copy()
        for _ in range(self.horizon):
            if holding:
                p = p + np.array([0, 0, self.step_m])
                g = 1.0
            else:
                d = grasp - p
                n = np.linalg.norm(d)
                if n < 0.002:
                    out.append(np.r_[q_ref, 1.0])     # there: close
                    continue
                p = p + d * min(1.0, self.step_m / n)
                g = 0.0
            try:
                q_ref = solve_ik(tool_down(p[0], p[1], p[2], yaw), q_ref)
            except IKError:
                pass
            out.append(np.r_[q_ref, g])
        return np.array(out)


def write_demo_objects(objects_dir, world: SimWorld):
    """objects/<box>/ref_*.png reference pictures (taken from above, turned)
    for the sim boxes, as an operator would add a new product."""
    from PIL import Image
    from pathlib import Path
    for b in world.boxes:
        folder = Path(objects_dir) / b.name
        folder.mkdir(parents=True, exist_ok=True)
        yaw0 = b.yaw_deg
        for i, extra in enumerate((0, 50, 110)):
            b.yaw_deg = yaw0 + extra
            Image.fromarray(world.reference_image(b)).save(folder / f"ref_{i}.png")
        b.yaw_deg = yaw0
