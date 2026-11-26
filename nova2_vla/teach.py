"""
Teaching by touching: read where the gripper tip is straight from the
robot's joints (twin FK), so no pendant arithmetic (tool 0 vs tool 1,
flange vs fingertips, mm vs m) can go wrong.

  measure-table      touch the table with the closed fingertips at a few
                     spots -> robot.table_z_m
  teach-location     touch the spot where a part should be set down ->
                     an {x, y, z, w} entry in the locations file

The arm is only read, never moved or enabled: jog it from the pendant (or
hand-guide it) between readings. Each reading opens and closes its own
connection, so the pendant stays free.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .geometry import tilt_deg, tool_pose, yaw_of
from .task import LocationLibrary, TaskError
from .twin import fk, joint_map

MAX_TILT_DEG = 5.0


@dataclass
class TipReading:
    xyz: np.ndarray      # tool tip in the base frame [m]
    yaw_deg: float       # tool x about the vertical (= the pendant's Rz with the tool pointing down)
    tilt_deg: float      # tool z from straight down
    model_err_mm: float  # twin FK flange vs the controller's own GetPose
    hint: str = ""       # why they might differ, if the controller knows


def read_tip(connect) -> TipReading:
    """connect() -> a connected robot (context manager)."""
    with connect() as robot:
        q = robot.get_angle()
        flange_ctrl = robot.get_pose()[:3] / 1000.0
        hint = getattr(robot, "pose_hint", "")
    T = tool_pose(q)
    err = float(np.linalg.norm(fk(joint_map.servo_to_visual(q), "flange")[:3, 3] - flange_ctrl) * 1000)
    return TipReading(T[:3, 3].copy(), math.degrees(yaw_of(T)), tilt_deg(T), err, hint)


def _check_reading(r: TipReading, model_tol_mm, log):
    if r.model_err_mm > model_tol_mm:
        raise TaskError(f"the twin's model is {r.model_err_mm:.1f} mm off the controller's pose, so this reading "
                        "would be wrong: fix the DH table first (see the twin README)" + r.hint)
    if r.tilt_deg > MAX_TILT_DEG:
        log(f"  warning: the tool is tilted {r.tilt_deg:.1f} deg; point it straight down (Rx 180, Ry 0) "
            "for an exact reading")


@dataclass
class TableMeasurement:
    z: float                 # what to use as robot.table_z_m [m]
    mean: float
    spread: float            # highest - lowest reading [m]
    readings: List[TipReading]


def measure_table(connect, points=3, ask=input, log=print, model_tol_mm=3.0) -> TableMeasurement:
    readings = []
    for i in range(points):
        ask(f"[{i + 1}/{points}] close the gripper, jog the fingertips down until they just touch the table "
            f"(a sheet of paper under them should just drag){', at another spot' if i else ''}; then press Enter ")
        r = read_tip(connect)
        _check_reading(r, model_tol_mm, log)
        log(f"  tip at x {r.xyz[0] * 1000:7.1f}  y {r.xyz[1] * 1000:7.1f}  z {r.xyz[2] * 1000:7.1f} mm")
        readings.append(r)
    zs = np.array([r.xyz[2] for r in readings])
    spread = float(zs.max() - zs.min())
    # uneven table: plan against its highest point, so nothing dips into it
    z = float(zs.max()) if spread > 0.003 else float(zs.mean())
    if spread > 0.003:
        log(f"  warning: the readings differ by {spread * 1000:.1f} mm (table not level, or a reading was off); "
            "using the highest")
    return TableMeasurement(z, float(zs.mean()), spread, readings)


def write_table_z(config_path, z):
    """Set robot.table_z_m in the cell YAML, keeping its comments."""
    path = Path(config_path)
    text = path.read_text()
    new, n = re.subn(r"(?m)^(\s*)table_z_m:[^\n]*$", lambda m: f"{m.group(1)}table_z_m: {z:.4f}", text, count=1)
    if n == 0:
        raise TaskError(f"no table_z_m line in {path}: add `table_z_m: {z:.4f}` under robot:")
    path.write_text(new)


def teach_location(connect, locations_file, name, z_reference="surface", table_z=None, ask=input, log=print,
                   model_tol_mm=3.0):
    """Store where the tip is now as location `name`. surface: the tip
    touches the surface the part is set down on; tool: the tip is where
    it should be when the gripper opens."""
    what = ("touches the surface where the part should be set down" if z_reference == "surface"
            else "is where it should be when the gripper opens")
    ask(f"jog the gripper (pointing straight down, turned as the part should be) until its tip {what}; "
        "then press Enter ")
    r = read_tip(connect)
    _check_reading(r, model_tol_mm, log)

    path = Path(locations_file)
    data = json.loads(path.read_text()) if path.exists() else {"units": "mm", "z_reference": "surface",
                                                               "locations": {}}
    scale = 1000.0 if data.get("units", "mm") == "mm" else 1.0
    entries = data.setdefault("locations", {}) if "locations" in data or not path.exists() else data
    x, y, z = (round(float(v) * scale, 1 if scale > 1 else 4) for v in r.xyz)
    entry = {"x": x, "y": y, "z": z, "w": round(r.yaw_deg, 1)}
    if z_reference == "surface" and table_z is not None and abs(r.xyz[2] - table_z) < 0.002:
        entry["z"] = "table"          # it's the table itself: follow table_z_m from now on
    if z_reference != data.get("z_reference", "surface"):
        entry["z_reference"] = z_reference
    replaced = name in entries
    entries[name] = entry
    LocationLibrary.parse(data, str(path), table_z)    # refuse to write a file that won't load
    path.write_text(json.dumps(data, indent=2) + "\n")
    log(f"{'updated' if replaced else 'added'} {name!r} in {path}: {json.dumps(entry)}")
    return entry
