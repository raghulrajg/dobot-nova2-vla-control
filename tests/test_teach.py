import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from nova2_vla.config import load_config
from nova2_vla.geometry import home_servo_deg, tool_down
from nova2_vla.robot.fake import FakeNova2Controller
from nova2_vla.robot.nova2 import Nova2
from nova2_vla.task import LocationLibrary, TaskError
from nova2_vla.teach import measure_table, teach_location, write_table_z
from nova2_vla.twin import solve_ik

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def ctrl():
    c = FakeNova2Controller().start()
    yield c
    c.stop()


def _connect(c):
    return lambda: Nova2("127.0.0.1", c.port, realtime_feedback=False)


def _jog(c, x, y, z, yaw_deg=0.0, tilt=False):
    q = solve_ik(tool_down(x, y, z, math.radians(yaw_deg)), home_servo_deg())
    if tilt:
        q = q + np.array([0, 0, 0, 0, 8.0, 0])
    with c.lock:
        c.joints = q


def test_measure_table_level(ctrl):
    spots = iter([(-0.40, 0.10), (-0.30, 0.25), (-0.45, -0.05)])
    m = measure_table(_connect(ctrl), 3, ask=lambda _: _jog(ctrl, *next(spots), 0.018), log=lambda _: None)
    assert m.z == pytest.approx(0.018, abs=1e-4) and m.spread < 1e-4


def test_measure_table_uneven_uses_highest(ctrl):
    heights = iter([0.018, 0.024, 0.016])
    logs = []
    m = measure_table(_connect(ctrl), 3, ask=lambda _: _jog(ctrl, -0.4, 0.1, next(heights)), log=logs.append)
    assert m.z == pytest.approx(0.024, abs=1e-4) and m.spread == pytest.approx(0.008, abs=1e-4)
    assert any("differ by 8.0 mm" in line for line in logs)


def test_measure_table_warns_on_tilt(ctrl):
    logs = []
    measure_table(_connect(ctrl), 1, ask=lambda _: _jog(ctrl, -0.4, 0.1, 0.02, tilt=True), log=logs.append)
    assert any("tilted" in line for line in logs)


def test_write_table_z_keeps_comments(tmp_path):
    cfg = tmp_path / "config" / "cell.yaml"
    cfg.parent.mkdir()
    shutil.copy(REPO / "config" / "cell.yaml", cfg)
    write_table_z(cfg, 0.01834)
    assert load_config(cfg).robot.table_z_m == pytest.approx(0.0183)
    assert "# The table surface in the base frame" in cfg.read_text()
    bare = tmp_path / "bare.yaml"
    bare.write_text("robot:\n  ip: 1.2.3.4\n")
    with pytest.raises(TaskError, match="add `table_z_m"):
        write_table_z(bare, 0.02)


def test_teach_location(ctrl, tmp_path):
    path = tmp_path / "locations.json"
    shutil.copy(REPO / "config" / "locations.json", path)
    quiet = dict(log=lambda _: None)
    # on a shelf 80 mm above the table, gripper turned 30 deg
    entry = teach_location(_connect(ctrl), path, "shelf_E", table_z=0.018,
                           ask=lambda _: _jog(ctrl, -0.30, 0.35, 0.098, 30.0), **quiet)
    assert entry == {"x": pytest.approx(-300.0), "y": pytest.approx(350.0), "z": pytest.approx(98.0),
                     "w": pytest.approx(30.0)}
    # touching the table itself -> follows table_z_m
    entry = teach_location(_connect(ctrl), path, "bin_A", table_z=0.018,
                           ask=lambda _: _jog(ctrl, -0.26, 0.41, 0.0185), **quiet)
    assert entry["z"] == "table"
    # release height of the tip, for a fixture
    entry = teach_location(_connect(ctrl), path, "jig", "tool", table_z=0.018,
                           ask=lambda _: _jog(ctrl, -0.35, 0.20, 0.10), **quiet)
    assert entry["z_reference"] == "tool"
    locs = LocationLibrary.load(path, 0.018)
    assert locs.get("bin_A").x == pytest.approx(-0.26) and locs.get("bin_A").z == pytest.approx(0.018)
    assert locs.get("shelf_E").w_deg == pytest.approx(30.0)
    assert locs.get("tray_C").x == pytest.approx(-0.42)            # the others are kept
    assert json.loads(path.read_text())["units"] == "mm"


def test_teach_location_new_file(ctrl, tmp_path):
    path = tmp_path / "new.json"
    teach_location(_connect(ctrl), path, "drop", ask=lambda _: _jog(ctrl, -0.3, 0.3, 0.05), log=lambda _: None)
    assert LocationLibrary.load(path).get("drop").z == pytest.approx(0.05)
