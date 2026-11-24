"""
Bridge to the Nova 2 digital twin (github.com/raghulrajg/manipulator-agent).

The kinematics (DH tables, closed-form IK, table check), the top-down pick
planner, the Dobot TCP client and the Feetech gripper all live in that repo
and are used here unchanged, so there is one source of truth for the arm
model: keep editing the DH rows in manipulator_kinematics.py there.

It is a git submodule at third_party/manipulator-agent. Point
MANIPULATOR_AGENT_PATH at another checkout to use that one instead.
"""

import os
import sys
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent.parent / "third_party" / "manipulator-agent"
TWIN_DIR = Path(os.environ.get("MANIPULATOR_AGENT_PATH", _DEFAULT)).resolve()

if not (TWIN_DIR / "manipulator_kinematics.py").is_file():
    raise ImportError(
        f"the Nova 2 digital twin is missing at {TWIN_DIR}: run "
        "`git submodule update --init` or set MANIPULATOR_AGENT_PATH")
if str(TWIN_DIR) not in sys.path:
    sys.path.insert(0, str(TWIN_DIR))

from manipulator_kinematics import (  # noqa: E402
    SERVO_LIMITS_DEG, T_FLANGE_CAMERA, T_FLANGE_TOOL, IKError, TableConstraint, fk,
    ik_solutions, joint_map, pose, rot_x, rot_z, solve_ik,
)
from manipulator_agent import ManipulatorAgent, plan_place  # noqa: E402
from dobot_tcp import (  # noqa: E402
    MODE_COLLISION, MODE_DISABLED, MODE_ERROR, MODE_IDLE, MODE_NAMES, MODE_POWER_OFF,
    MODE_RUNNING, DobotError, DobotTCP,
)
from gripper import LEFT, RIGHT, FakeGripper, FeetechGripper, GripperError, GripResult  # noqa: E402

__all__ = [
    "TWIN_DIR", "SERVO_LIMITS_DEG", "T_FLANGE_CAMERA", "T_FLANGE_TOOL", "IKError",
    "TableConstraint", "fk", "ik_solutions", "joint_map", "pose", "rot_x", "rot_z", "solve_ik",
    "ManipulatorAgent", "plan_place", "MODE_COLLISION", "MODE_DISABLED", "MODE_ERROR",
    "MODE_IDLE", "MODE_NAMES", "MODE_POWER_OFF", "MODE_RUNNING", "DobotError", "DobotTCP",
    "LEFT", "RIGHT", "FakeGripper", "FeetechGripper", "GripperError", "GripResult",
]
