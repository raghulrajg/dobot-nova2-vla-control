"""
SafetyFilter: every joint target a learned policy proposes passes through
here before it reaches the robot. A VLA can output anything, so nothing
it says is trusted:

  1. not finite                          -> rejected (hold)
  2. outside the joint limits            -> clamped to them
  3. more than max_step_deg from the
     last safe target on any joint       -> shortened (whole vector scaled)
  4. at the target, any of
       - part of the arm below the table (the twin's TableConstraint:
         links, camera 5 cm, gripper tip 5 mm)
       - tool tip outside the workspace box
       - tool tip outside the pick zone (a cylinder around the object
         being picked, from the table up to just above the hover pose)
       - tool tilted more than max_tilt_deg from straight down
                                         -> moved back along the step to the
                                            farthest point that passes all of
                                            them; if none, rejected (hold)

The caller holds the arm on rejection and stops the policy after too many
rejections in a row.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..geometry import tilt_deg, tool_pose
from ..twin import SERVO_LIMITS_DEG, TableConstraint, joint_map


@dataclass
class PickZone:
    center_xy: Tuple[float, float]
    radius: float
    z_max: float


@dataclass
class SafeTarget:
    q: np.ndarray                 # what to command
    status: str                   # "ok" | "limited" | "rejected"
    reasons: List[str] = field(default_factory=list)

    @property
    def rejected(self):
        return self.status == "rejected"


class SafetyFilter:
    def __init__(self, table: TableConstraint, max_step_deg=2.0, max_tilt_deg=45.0,
                 workspace_x=(-0.65, 0.65), workspace_y=(-0.65, 0.65), workspace_z_max=0.70,
                 limits=SERVO_LIMITS_DEG, joint_margin_deg=1.0):
        self.table = table
        self.max_step_deg = float(max_step_deg)
        self.max_tilt_deg = float(max_tilt_deg)
        self.workspace_x, self.workspace_y = workspace_x, workspace_y
        self.workspace_z_max = workspace_z_max
        self.lo = np.asarray(limits)[:, 0] + joint_margin_deg
        self.hi = np.asarray(limits)[:, 1] - joint_margin_deg
        self.zone: Optional[PickZone] = None

    def set_zone(self, zone: Optional[PickZone]):
        self.zone = zone

    def problems(self, q) -> List[str]:
        """Why joint target q (servo deg) is unsafe; [] = safe."""
        out = list(self.table.violations(joint_map.servo_to_visual(q)))
        T = tool_pose(q)
        x, y, z = T[:3, 3]
        if not (self.workspace_x[0] <= x <= self.workspace_x[1] and self.workspace_y[0] <= y <= self.workspace_y[1]
                and z <= self.workspace_z_max):
            out.append(f"tool tip ({x:.3f}, {y:.3f}, {z:.3f}) outside the workspace")
        if self.zone is not None:
            r = math.hypot(x - self.zone.center_xy[0], y - self.zone.center_xy[1])
            if r > self.zone.radius:
                out.append(f"tool tip {r * 100:.1f} cm from the object (zone {self.zone.radius * 100:.0f} cm)")
            if z > self.zone.z_max:
                out.append(f"tool tip {z:.3f} m above the pick zone ({self.zone.z_max:.3f} m)")
        tilt = tilt_deg(T)
        if tilt > self.max_tilt_deg:
            out.append(f"tool tilted {tilt:.0f} deg (max {self.max_tilt_deg:.0f})")
        return out

    def filter(self, q_ref, q_target) -> SafeTarget:
        """q_ref: the last target that was sent (safe); q_target: the
        policy's proposal. Both servo degrees."""
        q_ref = np.asarray(q_ref, dtype=float)
        q_target = np.asarray(q_target, dtype=float).reshape(-1)[:6]
        if q_target.shape != (6,) or not np.all(np.isfinite(q_target)):
            return SafeTarget(q_ref.copy(), "rejected", ["non-finite or malformed action"])

        reasons = []
        q = np.clip(q_target, self.lo, self.hi)
        if not np.allclose(q, q_target):
            reasons.append("clamped to the joint limits")
        step = q - q_ref
        biggest = float(np.abs(step).max())
        if biggest > self.max_step_deg:
            q = q_ref + step * (self.max_step_deg / biggest)
            reasons.append(f"step {biggest:.1f} deg shortened to {self.max_step_deg:.1f}")

        bad = self.problems(q)
        if not bad:
            return SafeTarget(q, "limited" if reasons else "ok", reasons)
        # back off along the step to the farthest safe point
        lo_t, hi_t, best = 0.0, 1.0, None
        if not self.problems(q_ref):
            for _ in range(8):
                mid = (lo_t + hi_t) / 2
                if self.problems(q_ref + mid * (q - q_ref)):
                    hi_t = mid
                else:
                    lo_t, best = mid, q_ref + mid * (q - q_ref)
        if best is None or lo_t < 0.05:
            return SafeTarget(q_ref.copy(), "rejected", reasons + bad)
        return SafeTarget(best, "limited", reasons + [f"stopped at {lo_t:.0%} of the step: " + "; ".join(bad)])
