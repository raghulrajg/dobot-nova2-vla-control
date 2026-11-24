from .base import ACTION_NAMES, JOINT_NAMES, ROBOT_TYPE, STATE_NAMES, GraspPolicy, PolicyError


def load_policy(path, device="cuda", cameras=None):
    """A LeRobot checkpoint (local dir or Hub id) as a GraspPolicy."""
    from .lerobot_policy import LeRobotVLA
    return LeRobotVLA(path, device, cameras)


__all__ = ["ACTION_NAMES", "JOINT_NAMES", "ROBOT_TYPE", "STATE_NAMES", "GraspPolicy", "PolicyError", "load_policy"]
