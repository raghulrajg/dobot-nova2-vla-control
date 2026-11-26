import socket
import threading
import time

import numpy as np
import pytest

from nova2_vla.geometry import home_servo_deg, tool_down, tool_pose
from nova2_vla.robot.fake import FakeNova2Controller
from nova2_vla.robot.feedback import PACKET_SIZE, RealtimeFeedback, build_packet, parse_packet
from nova2_vla.robot.nova2 import MODE_DRAG, Nova2
from nova2_vla.robot.safety import PickZone, SafetyFilter
from nova2_vla.robot.servo import ServoStreamer
from nova2_vla.twin import DobotError, TableConstraint, solve_ik

HOME = home_servo_deg()
TABLE = TableConstraint(-0.05)


# ---------------------------------------------------------------- feedback

def test_feedback_packet_roundtrip():
    q = [-180.0, 1.5, -90.25, 0.0, 90.0, 89.5]
    fb = parse_packet(build_packet(q, robot_mode=7, stamp_ms=123, tool_vector=(1, 2, 3, 4, 5, 6)))
    assert fb.robot_mode == 7 and fb.stamp_ms == 123
    assert fb.q_actual.tolist() == q and fb.tool_vector.tolist() == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        parse_packet(b"\0" * PACKET_SIZE)


def test_feedback_reader_resyncs_after_garbage():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def serve():
        conn, _ = server.accept()
        with conn:
            conn.sendall(b"junk!" + build_packet(HOME, 5))
            time.sleep(0.05)
            conn.sendall(build_packet(HOME + 1, 7)[:700])   # split packet
            time.sleep(0.05)
            conn.sendall(build_packet(HOME + 1, 7)[700:])
            time.sleep(0.3)

    threading.Thread(target=serve, daemon=True).start()
    fb = RealtimeFeedback("127.0.0.1", server.getsockname()[1]).start()
    time.sleep(0.25)
    assert fb.latest.robot_mode == 7
    assert fb.latest.q_actual == pytest.approx(HOME + 1)
    fb.stop()
    server.close()


# ---------------------------------------------------------------- ServoJ and drag on the fake controller

@pytest.fixture
def robot():
    ctrl = FakeNova2Controller(servo_deg_per_s=300).start()
    r = Nova2("127.0.0.1", ctrl.port, feedback_port=ctrl.feedback_port, poll_s=0.01).connect()
    r.enable()
    yield r, ctrl
    r.close()
    ctrl.stop()


def test_servo_streamer_rate_limits_and_reaches_goal(robot):
    r, ctrl = robot
    s = ServoStreamer(r, period=0.01, max_speed_deg_s=200).start(r.joints())
    goal = HOME + np.array([10, -5, 5, 0, 0, 20])
    s.set_goal(goal)
    t0 = time.monotonic()
    assert s.settle(tol_deg=0.05, timeout=3)
    assert time.monotonic() - t0 >= 20 / 200 * 0.9            # 20 deg at 200 deg/s, not a jump
    s.stop()
    r.wait_idle()
    assert r.get_angle() == pytest.approx(goal, abs=1e-3)
    assert ctrl.servo_count >= 10


def test_servo_streamer_reports_collision(robot):
    r, ctrl = robot
    s = ServoStreamer(r, period=0.01, mode_check_every=2).start(r.joints())
    with ctrl.lock:
        ctrl.mode = 11       # collision detected
    time.sleep(0.2)
    with pytest.raises(DobotError, match="collision|-2"):
        s.set_goal(HOME)
    s.stop()


def test_servoj_refuses_joint_limits(robot):
    r, _ = robot
    with pytest.raises(DobotError, match="joint limits"):
        r.servo_j(HOME + np.array([0, 200, 0, 0, 0, 0]))
    with pytest.raises(DobotError, match="joint limits"):
        r.servo_j(HOME * np.nan)


def test_drag_mode(robot):
    r, ctrl = robot
    r.start_drag()
    assert r.robot_mode() == MODE_DRAG
    ctrl.drag_to(HOME + 3)
    time.sleep(0.05)
    assert r.joints() == pytest.approx(HOME + 3)
    r.stop_drag()
    r.wait_idle()


# ---------------------------------------------------------------- safety filter

def _q(x, y, z, yaw=0.0):
    return solve_ik(tool_down(x, y, z, yaw), HOME)


def test_safety_passes_small_safe_steps():
    f = SafetyFilter(TABLE)
    q0 = _q(-0.40, 0.10, 0.10)
    q1 = _q(-0.40, 0.10, 0.095)
    out = f.filter(q0, q1)
    assert out.status == "ok" and out.q == pytest.approx(q1)


def test_safety_rejects_nan_and_shortens_big_steps():
    f = SafetyFilter(TABLE, max_step_deg=2.0)
    q0 = _q(-0.40, 0.10, 0.10)
    assert f.filter(q0, q0 * np.nan).rejected
    out = f.filter(q0, q0 + np.array([10.0, 0, 0, 0, 0, 0]))
    assert out.status == "limited" and np.abs(out.q - q0).max() == pytest.approx(2.0)


def test_safety_never_goes_below_the_table():
    f = SafetyFilter(TABLE, max_step_deg=30.0)
    q0 = _q(-0.40, 0.10, -0.03)                       # tip 2 cm above the table
    q_bad = _q(-0.40, 0.10, -0.07)                    # tip 2 cm into it
    out = f.filter(q0, q_bad)
    assert not out.rejected and out.status == "limited"
    assert tool_pose(out.q)[2, 3] >= -0.05 + 0.005 - 1e-4      # stopped at the tip clearance
    assert f.filter(_q(-0.40, 0.10, -0.044), q_bad).rejected   # already at the limit: hold


def test_safety_pick_zone_and_tilt():
    f = SafetyFilter(TABLE, max_step_deg=90.0, max_tilt_deg=30.0)
    f.set_zone(PickZone((-0.40, 0.10), 0.05, 0.20))
    q0 = _q(-0.40, 0.10, 0.10)
    assert "from the object" in " ".join(f.problems(_q(-0.30, 0.10, 0.10)))
    assert "above the pick zone" in " ".join(f.problems(_q(-0.40, 0.10, 0.25)))
    out = f.filter(q0, _q(-0.25, 0.10, 0.10))
    assert np.hypot(*(tool_pose(out.q)[:2, 3] - [-0.40, 0.10])) <= 0.05 + 1e-3
    tilted = q0 + np.array([0, 0, 0, 0, 40.0, 0])
    assert "tilted" in " ".join(f.problems(tilted))


# ---------------------------------------------------------------- GetPose on older V4 firmware

@pytest.mark.parametrize("style, cmd", [("keyword", "GetPose(user=0,tool=0)"), ("positional", "GetPose(0,0)"),
                                        ("bare", "GetPose()")])
def test_get_pose_finds_the_firmware_form(style, cmd):
    from nova2_vla.twin import fk, joint_map
    ctrl = FakeNova2Controller(get_pose_style=style).start()
    try:
        with Nova2("127.0.0.1", ctrl.port, realtime_feedback=False) as r:
            p = r.get_pose()
            assert r._pose_cmd == cmd
            assert p[:3] / 1000 == pytest.approx(fk(joint_map.servo_to_visual(HOME), "flange")[:3, 3], abs=1e-6)
            assert r.get_pose()[:3] == pytest.approx(p[:3])             # cached form, no more probing
            assert r.pose_frame == ("selected" if style == "bare" else "tool0") and r.pose_hint == ""
        sent = [(n, a) for n, a, _ in ctrl.log if n in ("GetPose", "User", "Tool")]
        probes = {"keyword": [], "positional": [("GetPose", "user=0,tool=0")],
                  "bare": [("GetPose", "user=0,tool=0"), ("GetPose", "0,0")]}[style]
        arg = cmd[len("GetPose("):-1]
        frames = [("GetPose", ""), ("User", "0"), ("Tool", "0")] if style == "bare" else []
        assert sent == probes + frames + [("GetPose", arg)] * 2      # probed once, then cached
    finally:
        ctrl.stop()


def test_bare_get_pose_selects_the_base_frames():
    """Only GetPose() works and a gripper tool is selected: it selects
    user 0 / tool 0 itself, or explains why the reading is off."""
    from nova2_vla.teach import measure_table
    from nova2_vla.task import TaskError
    # firmware that refuses User/Tool: the mismatch is explained
    ctrl = FakeNova2Controller(get_pose_style="bare", selected_tool=1, frame_commands=False).start()
    try:
        with Nova2("127.0.0.1", ctrl.port) as r, pytest.raises(TaskError, match=r"User\(0\) / Tool\(0\) were refused"):
            measure_table(r, 1, ask=lambda _: None, log=lambda _: None)
    finally:
        ctrl.stop()
    # firmware that accepts them: tool 0 is selected and the reading is the flange
    ctrl = FakeNova2Controller(get_pose_style="bare", selected_tool=1).start()
    try:
        with Nova2("127.0.0.1", ctrl.port) as r:
            m = measure_table(r, 1, ask=lambda _: None, log=lambda _: None)
            assert ctrl.selected_tool == 0 and m.readings[0].model_err_mm < 0.01
    finally:
        ctrl.stop()
