"""
Cell configuration: one YAML file (config/cell.yaml) describing the robot,
gripper, cameras, perception, VLA policy, pick strategy and safety limits.
Every key has a default here; the YAML only overrides what differs.
Unknown keys are an error, so a typo can't silently fall back to a default.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class RobotConfig:
    ip: str = "192.168.5.1"
    home_servo_deg: Optional[List[float]] = None    # detection pose; None = twin's home
    table_z_m: Optional[float] = None               # table surface, base frame [m]; required on the robot
    table_x_range_m: Optional[Tuple[float, float]] = None   # table footprint; None = endless floor
    table_y_range_m: Optional[Tuple[float, float]] = None
    speed: int = 20                  # MovJ speed / acceleration %
    line_speed: int = 10             # MovL speed / acceleration %
    speed_factor: Optional[int] = None   # controller global speed %, None = leave as on the pendant
    reach_tol_mm: float = 2.0
    model_tol_mm: float = 3.0
    realtime_feedback: bool = True   # read joints from port 30004 (8 ms) during ServoJ streaming


@dataclass
class GripperConfig:
    type: str = "feetech"            # feetech | fake
    port: str = "/dev/ttyUSB1"
    servo_id: int = 21
    torque_limit: int = 300
    max_opening_m: float = 0.08      # widest object the fingers can take


@dataclass
class CameraConfig:
    type: str = "realsense"          # realsense | opencv | sim
    serial: Optional[str] = None     # realsense serial (None = first found)
    index: int = 0                   # opencv device index
    width: int = 640
    height: int = 480
    fps: int = 30
    depth: bool = True               # realsense: also stream depth aligned to colour
    mount: str = "wrist"             # wrist (moves with the flange, twin's camera frame) | fixed
    optical_axis_rotation_deg: float = 0.0   # wrist: image rotated about the view axis vs the twin's camera frame


@dataclass
class PerceptionConfig:
    proposer: str = "depth+owlv2"    # any of depth, owlv2 joined with +
    embedder: str = "dinov2"         # dinov2 | color_hist
    owl_model: str = "google/owlv2-base-patch16-ensemble"
    embed_model: str = "facebook/dinov2-base"
    device: str = "cuda"
    min_similarity: float = 0.55     # cosine similarity to the reference images to accept a match
    ambiguity_margin: float = 0.03   # warn if the runner-up is this close
    min_object_height_m: float = 0.006   # depth proposer: pixels this far above the table are objects
    min_area_px: int = 150
    detect_frames: int = 3           # frames to read at the detection pose (latest is used)


@dataclass
class VLAConfig:
    policy_path: Optional[str] = None    # LeRobot checkpoint dir (…/pretrained_model) or Hub id
    device: str = "cuda"
    fps: int = 15                    # control / recording rate (must match the training dataset)
    execute_steps: int = 10          # actions run from each predicted chunk before predicting again
    max_steps: int = 450             # give up after this many control steps (30 s at 15 Hz)
    task_template: str = "pick up the {description}"
    cameras: Optional[List[str]] = None  # rig cameras fed to the policy; None = all
    gripper_close_above: float = 0.6     # action[6] above this closes the gripper
    gripper_open_below: float = 0.4
    post_grasp_steps: int = 10       # steps the policy may keep lifting after a good grip
    max_missed_grips: int = 2


@dataclass
class PickConfig:
    strategy: str = "vla_with_fallback"  # vla | scripted | vla_with_fallback
    hover_height_m: float = 0.15     # tool tip above the grasp point where the VLA takes over; also the lift
    grasp_depth_m: float = 0.02      # fingertips this far below the object's top (capped at half its height)
    grasp_yaw_offset_deg: float = 0.0   # 90 if the fingers close along the flange's y axis
    place_approach_m: float = 0.10
    place_clearance_m: float = 0.005    # object's bottom this far above the place surface at release
    try_flipped_place: bool = True   # if w is unreachable, try w + 180 (parallel gripper)


@dataclass
class SafetyConfig:
    max_joint_step_deg: float = 2.0       # per control step (2 deg at 15 Hz = 30 deg/s)
    max_joint_speed_deg_s: float = 45.0   # ServoJ streamer's rate limit
    servo_period_s: float = 0.03          # ServoJ every 30 ms (Dobot's recommendation)
    max_tilt_deg: float = 45.0            # tool z within this of straight down during the VLA grasp
    pick_zone_radius_m: float = 0.08      # VLA may move the tool tip this far from the object's axis
    pick_zone_above_hover_m: float = 0.05
    workspace_x_m: Tuple[float, float] = (-0.65, 0.65)
    workspace_y_m: Tuple[float, float] = (-0.65, 0.65)
    workspace_z_max_m: float = 0.70
    max_rejected_steps: int = 15          # consecutive unsafe actions before the VLA is stopped
    link_clearance_m: float = 0.05
    tip_clearance_m: float = 0.005


@dataclass
class PathsConfig:
    objects_dir: str = "objects"
    locations_file: str = "config/locations.json"
    runs_dir: str = "runs"
    dataset_root: str = "data/nova2_picks"
    dataset_repo_id: str = "local/nova2_picks"


@dataclass
class CellConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    cameras: Dict[str, CameraConfig] = field(default_factory=lambda: {"wrist": CameraConfig()})
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    vla: VLAConfig = field(default_factory=VLAConfig)
    pick: PickConfig = field(default_factory=PickConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    base_dir: Path = field(default_factory=Path.cwd)   # relative paths resolve against this

    def path(self, p):
        p = Path(p).expanduser()
        return p if p.is_absolute() else self.base_dir / p

    @property
    def wrist_camera(self) -> str:
        for name, cam in self.cameras.items():
            if cam.mount == "wrist":
                return name
        raise ConfigError("no camera with mount: wrist (the object is located with the wrist camera)")


def _build(cls, data, where):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; valid: {sorted(names)}")
    kwargs = {}
    for key, value in data.items():
        if key.endswith(("_range_m", "_x_m", "_y_m")) and value is not None:
            value = tuple(float(v) for v in value)
            if len(value) != 2 or value[0] >= value[1]:
                raise ConfigError(f"{where}.{key}: expected [low, high], got {list(value)}")
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path=None, overrides: Optional[dict] = None) -> CellConfig:
    """Read a cell YAML (None = all defaults); `overrides` is a nested dict
    applied on top (used by the command line)."""
    data, base = {}, Path.cwd()
    if path is not None:
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        base = path.resolve().parent.parent if path.resolve().parent.name == "config" else path.resolve().parent
    for section, values in (overrides or {}).items():
        data.setdefault(section, {})
        data[section] = {**(data[section] or {}), **values}

    top = {f.name for f in dataclasses.fields(CellConfig)} - {"base_dir"}
    unknown = sorted(set(data) - top)
    if unknown:
        raise ConfigError(f"unknown section(s) {unknown}; valid: {sorted(top)}")

    cameras_raw = data.get("cameras") or {"wrist": {}}
    cameras = {name: _build(CameraConfig, c, f"cameras.{name}") for name, c in cameras_raw.items()}
    cfg = CellConfig(
        robot=_build(RobotConfig, data.get("robot"), "robot"),
        gripper=_build(GripperConfig, data.get("gripper"), "gripper"),
        cameras=cameras,
        perception=_build(PerceptionConfig, data.get("perception"), "perception"),
        vla=_build(VLAConfig, data.get("vla"), "vla"),
        pick=_build(PickConfig, data.get("pick"), "pick"),
        safety=_build(SafetyConfig, data.get("safety"), "safety"),
        paths=_build(PathsConfig, data.get("paths"), "paths"),
        base_dir=base,
    )
    validate(cfg)
    return cfg


def validate(cfg: CellConfig):
    if cfg.pick.strategy not in ("vla", "scripted", "vla_with_fallback"):
        raise ConfigError(f"pick.strategy must be vla, scripted or vla_with_fallback, not {cfg.pick.strategy!r}")
    for name in ("speed", "line_speed", "speed_factor"):
        v = getattr(cfg.robot, name)
        if v is not None and not 1 <= v <= 100:
            raise ConfigError(f"robot.{name} must be 1..100")
    if cfg.robot.home_servo_deg is not None and len(cfg.robot.home_servo_deg) != 6:
        raise ConfigError("robot.home_servo_deg needs 6 joint angles")
    if cfg.robot.table_z_m is not None and not -1.0 < cfg.robot.table_z_m < 0.4:
        raise ConfigError(f"robot.table_z_m = {cfg.robot.table_z_m} is not a height in metres "
                          "(it must be below the tool tip at home, 0.218 m)")
    if cfg.gripper.type not in ("feetech", "fake"):
        raise ConfigError(f"gripper.type must be feetech or fake, not {cfg.gripper.type!r}")
    for name, cam in cfg.cameras.items():
        if cam.type not in ("realsense", "opencv", "sim"):
            raise ConfigError(f"cameras.{name}.type must be realsense, opencv or sim")
        if cam.mount not in ("wrist", "fixed"):
            raise ConfigError(f"cameras.{name}.mount must be wrist or fixed")
    cfg.wrist_camera  # noqa: B018 -- raises if there is none
    if not 0 < cfg.vla.gripper_open_below < cfg.vla.gripper_close_above < 1:
        raise ConfigError("need 0 < vla.gripper_open_below < vla.gripper_close_above < 1")
    if cfg.vla.cameras is not None:
        missing = [c for c in cfg.vla.cameras if c not in cfg.cameras]
        if missing:
            raise ConfigError(f"vla.cameras {missing} are not in cameras")
    if cfg.safety.servo_period_s < 0.004:
        raise ConfigError("safety.servo_period_s must be at least 0.004 s (ServoJ's minimum t)")
    for part in cfg.perception.proposer.split("+"):
        if part not in ("depth", "owlv2"):
            raise ConfigError(f"perception.proposer: unknown proposer {part!r} (depth, owlv2)")
    if cfg.perception.embedder not in ("dinov2", "color_hist"):
        raise ConfigError("perception.embedder must be dinov2 or color_hist")
