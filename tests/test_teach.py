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


@pytest.fixture
def robot(ctrl):
    r = Nova2("127.0.0.1", ctrl.port).connect()      # one connection for the whole test
    yield r
    r.close()


def _jog(c, x, y, z, yaw_deg=0.0, tilt=False):
    q = solve_ik(tool_down(x, y, z, math.radians(yaw_deg)), home_servo_deg())
    if tilt:
        q = q + np.array([0, 0, 0, 0, 8.0, 0])
    with c.lock:
        c.joints = q


def test_measure_table_level(ctrl, robot):
    spots = iter([(-0.40, 0.10), (-0.30, 0.25), (-0.45, -0.05)])
    m = measure_table(robot, 3, ask=lambda _: _jog(ctrl, *next(spots), 0.018), log=lambda _: None)
    assert m.z == pytest.approx(0.018, abs=1e-4) and m.spread < 1e-4


def test_measure_table_uneven_uses_highest(ctrl, robot):
    heights = iter([0.018, 0.024, 0.016])
    logs = []
    m = measure_table(robot, 3, ask=lambda _: _jog(ctrl, -0.4, 0.1, next(heights)), log=logs.append)
    assert m.z == pytest.approx(0.024, abs=1e-4) and m.spread == pytest.approx(0.008, abs=1e-4)
    assert any("differ by 8.0 mm" in line for line in logs)


def test_measure_table_warns_on_tilt(ctrl, robot):
    logs = []
    measure_table(robot, 1, ask=lambda _: _jog(ctrl, -0.4, 0.1, 0.02, tilt=True), log=logs.append)
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


def test_teach_location(ctrl, robot, tmp_path):
    path = tmp_path / "locations.json"
    shutil.copy(REPO / "config" / "locations.json", path)
    quiet = dict(log=lambda _: None)
    # on a shelf 80 mm above the table, gripper turned 30 deg
    entry = teach_location(robot, path, "shelf_E", table_z=0.018,
                           ask=lambda _: _jog(ctrl, -0.30, 0.35, 0.098, 30.0), **quiet)
    assert entry == {"x": pytest.approx(-300.0), "y": pytest.approx(350.0), "z": pytest.approx(98.0),
                     "w": pytest.approx(30.0)}
    # touching the table itself -> follows table_z_m
    entry = teach_location(robot, path, "bin_A", table_z=0.018,
                           ask=lambda _: _jog(ctrl, -0.26, 0.41, 0.0185), **quiet)
    assert entry["z"] == "table"
    # release height of the tip, for a fixture
    entry = teach_location(robot, path, "jig", "tool", table_z=0.018,
                           ask=lambda _: _jog(ctrl, -0.35, 0.20, 0.10), **quiet)
    assert entry["z_reference"] == "tool"
    locs = LocationLibrary.load(path, 0.018)
    assert locs.get("bin_A").x == pytest.approx(-0.26) and locs.get("bin_A").z == pytest.approx(0.018)
    assert locs.get("shelf_E").w_deg == pytest.approx(30.0)
    assert locs.get("tray_C").x == pytest.approx(-0.42)            # the others are kept
    assert json.loads(path.read_text())["units"] == "mm"


def test_teach_location_new_file(ctrl, robot, tmp_path):
    path = tmp_path / "new.json"
    teach_location(robot, path, "drop", ask=lambda _: _jog(ctrl, -0.3, 0.3, 0.05), log=lambda _: None)
    assert LocationLibrary.load(path).get("drop").z == pytest.approx(0.05)


def test_one_connection_for_all_readings(ctrl, robot):
    spots = iter([(-0.40, 0.10), (-0.30, 0.25), (-0.45, -0.05)])
    measure_table(robot, 3, ask=lambda _: _jog(ctrl, *next(spots), 0.018), log=lambda _: None)
    assert len(ctrl.connections) == 1 and not ctrl.feedback_connections


def test_busy_controller_message():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)                       # accepts, never answers: like a controller serving another client
    from nova2_vla.twin import DobotError
    try:
        with pytest.raises(DobotError, match="one TCP connection at a time"):
            Nova2("127.0.0.1", s.getsockname()[1], timeout=0.5).connect()
    finally:
        s.close()


# ---------------------------------------------------------------- jogging without a pendant

class Keys:
    def __init__(self, seq):
        self.seq = list(seq)

    def get(self):
        return self.seq.pop(0) if self.seq else "x"


def _jogger(robot, keys, table_z=0.018, tip_clearance=-0.003):
    from nova2_vla.jog import Jogger
    from nova2_vla.twin import TableConstraint
    robot.enable()
    table = TableConstraint(table_z, 0.05, tip_clearance) if table_z is not None else None
    return Jogger(robot, Keys(keys), table=table, log=lambda _: None)


def test_jog_steps_the_tip(ctrl, robot):
    from nova2_vla.geometry import tool_pose
    z0 = tool_pose(robot.get_angle())[2, 3]
    j = _jogger(robot, ["3", "f", "f", "1", "w", "2", "e", "\n"])
    j.position("")
    T = tool_pose(robot.get_angle())
    assert T[2, 3] == pytest.approx(z0 - 0.020, abs=1e-4)          # 2 x 10 mm down
    assert T[0, 3] == pytest.approx(-0.345 + 0.001, abs=1e-4)      # 1 mm +X
    assert math.degrees(math.atan2(T[1, 0], T[0, 0])) == pytest.approx(-90 - 5, abs=0.1)   # e: turned 5 deg


def test_jog_stops_at_the_table(ctrl, robot):
    from nova2_vla.geometry import tool_pose
    _jog(ctrl, -0.40, 0.10, 0.030)
    j = _jogger(robot, ["3", "f", "f", "f", "\n"])               # 10 mm steps from 12 mm above the table
    j.position("")
    assert tool_pose(robot.get_angle())[2, 3] >= 0.018 - 0.003 - 1e-6


def test_jog_x_aborts_and_drag_mode(ctrl, robot):
    from nova2_vla.jog import JogAborted
    from nova2_vla.robot.nova2 import MODE_DRAG
    j = _jogger(robot, ["g"])
    with pytest.raises(JogAborted):
        j.position("")                                             # g, then the keys run out -> x
    assert j.dragging and robot.robot_mode() == MODE_DRAG
    j.close()
    assert robot.robot_mode() == 5
