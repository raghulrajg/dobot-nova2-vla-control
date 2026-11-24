"""
The contract between the cell and any grasp policy.

Observation: one RGB image per policy camera (HxWx3 uint8) and the state
    [j1 .. j6 (servo deg, as GetAngle reports), gripper closure 0 open .. 1 closed]
Action, per control step: the same 7 numbers as the target for the next step
    [j1 .. j6 target (servo deg), gripper command (> 0.6 close, < 0.4 open)]

Recorded datasets use exactly these names, so a policy trained on them
needs no adapters.
"""

from typing import Dict, Protocol

import numpy as np

JOINT_NAMES = [f"j{i}.pos" for i in range(1, 7)]
STATE_NAMES = JOINT_NAMES + ["gripper.pos"]
ACTION_NAMES = STATE_NAMES
ROBOT_TYPE = "dobot_nova2"


class GraspPolicy(Protocol):
    def reset(self) -> None: ...

    def predict_chunk(self, images: Dict[str, np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        """(H, 7) actions for the next H control steps."""
        ...


class PolicyError(RuntimeError):
    pass
