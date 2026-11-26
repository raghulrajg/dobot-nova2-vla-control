"""End-to-end dry runs: the real cell code over the fake Nova 2 controller
(TCP), the rendered wrist camera and the simulated gripper."""

import argparse
import json
import math
import time

import numpy as np
import pytest

from nova2_vla.cli import build
from nova2_vla.config import load_config
from nova2_vla.data.recorder import EpisodeRecorder, Sampler
from nova2_vla.geometry import tool_pose

LOCATIONS = {
    "units": "mm",
    "locations": {
        "bin_A": {"x": -250, "y": 400, "z": -50, "w": 0},
        "tray_C": {"x": -420, "y": -150, "z": -50, "w": 45},
        "far_away": {"x": 2000, "y": 0, "z": 0, "w": 0},
    },
}


@pytest.fixture
def make_cell(tmp_path):
    (tmp_path / "loc.json").write_text(json.dumps(LOCATIONS))
    sessions = []

    def make(strategy="vla", **cfg_overrides):
        overrides = {"pick": {"strategy": strategy},
                     "paths": {"runs_dir": str(tmp_path / "runs"), "locations_file": str(tmp_path / "loc.json")},
                     "vla": {"fps": 30, "max_steps": 200}}
        for section, values in cfg_overrides.items():
            overrides.setdefault(section, {}).update(values)
        cfg = load_config(None, overrides)
        args = argparse.Namespace(dry_run=True, strategy=strategy, policy=None, record=False)
        s = build(cfg, args, log=lambda m: None)
        sessions.append(s)
        return s
    yield make
    for s in sessions:
        s.close()


def _at_home(s):
    return np.abs(s.robot.get_angle() - s.cell.home).max() < 0.5


def _placed_at(s, name, loc_name):
    loc = s.cell.locations.get(loc_name)
    box = s.world.box(name)
    return math.hypot(box.x - loc.x, box.y - loc.y) < 0.005 and box.held_offset is None


def test_vla_pick_and_place(make_cell):
    s = make_cell("vla")
    r = s.cell.run("pick the object 'red_block' place the 'bin_A'")
    assert r.ok, r.detail
    assert r.grasped_by == "vla"
    assert _placed_at(s, "red_block", "bin_A")
    assert _at_home(s)
    assert s.world.box("blue_block").x == pytest.approx(-0.36)    # the others weren't touched
    assert (s.cell.cfg.path(r.run_dir) / "attempt1.jpg").exists()


def test_scripted_pick_and_place_with_turned_place(make_cell):
    s = make_cell("scripted")
    r = s.cell.run("pick green_bar and place it on tray_C")
    assert r.ok, r.detail
    assert r.grasped_by == "scripted"
    assert _placed_at(s, "green_bar", "tray_C")
    # placed with the gripper turned to w (or w + 180): the bar's narrow side lies along w
    yaw = s.world.box("green_bar").yaw_deg
    assert abs((yaw - 45 + 90) % 180 - 90) < 3


class StubbornPolicy:
    """Never closes the gripper: the VLA attempt times out."""

    def reset(self):
        pass

    def predict_chunk(self, images, state, task):
        return np.tile(np.r_[state[:6], 0.0], (10, 1))


class ReckessPolicy:
    """Always tries to drive the tool 10 cm into the table."""

    def reset(self):
        pass

    def predict_chunk(self, images, state, task):
        q = state[:6] + np.array([0, 25.0, 25.0, 0, 0, 0])
        return np.tile(np.r_[q, 0.0], (10, 1))


@pytest.mark.parametrize("policy, why", [(StubbornPolicy(), "no grasp within"),
                                         (ReckessPolicy(), "unsafe")])
def test_vla_failure_falls_back_to_scripted(make_cell, policy, why):
    s = make_cell("vla_with_fallback", vla={"max_steps": 40})
    s.cell.policy = policy
    r = s.cell.run("pick the object 'blue_block' place the 'bin_A'")
    assert r.ok, r.detail
    assert r.grasped_by == "scripted"
    assert any(why in line for line in r.log), r.log
    assert _placed_at(s, "blue_block", "bin_A")


def test_reckless_policy_never_breaks_the_table(make_cell):
    s = make_cell("vla", vla={"max_steps": 60})
    s.cell.policy = ReckessPolicy()
    lowest = []
    orig = s.robot.servo_j

    def spy(q, **kw):
        lowest.append(tool_pose(q)[2, 3])
        return orig(q, **kw)
    s.robot.servo_j = spy
    r = s.cell.run("pick the object 'red_block' place the 'bin_A'")
    assert not r.ok and r.state == "grasp_failed"
    assert min(lowest) >= -0.05 + 0.005 - 1e-4       # tool tip never below table + 5 mm
    assert _at_home(s)


def test_object_not_found(make_cell, tmp_path):
    s = make_cell("vla")
    folder = s.cell.objects.root / "yellow_block"
    folder.mkdir()
    from PIL import Image
    Image.fromarray(np.full((64, 64, 3), (230, 220, 30), np.uint8)).save(folder / "ref.png")
    r = s.cell.run("pick the object 'yellow_block' place the 'bin_A'")
    assert r.state == "not_found" and _at_home(s)


def _motions(s):
    return [name for name, _, _ in s.ctrl.log if name in ("MovJ", "MovL", "ServoJ")]


def test_unreachable_location_is_refused_before_moving(make_cell):
    s = make_cell("vla")
    before = len(_motions(s))
    r = s.cell.run("pick the object 'red_block' place the 'far_away'")
    assert r.state == "rejected" and "can't be reached" in r.detail and "out of the arm's reach" in r.detail
    assert len(_motions(s)) == before          # the arm never moved
    assert s.world.box("red_block").held_offset is None


def test_unknown_names_are_rejected(make_cell):
    s = make_cell("vla")
    r = s.cell.run("pick the object 'red_blok' place the 'bin_A'")
    assert r.state == "rejected" and "did you mean 'red_block'" in r.detail
    r = s.cell.run("pick the object 'red_block' place the 'bin_Z'")
    assert r.state == "rejected" and "unknown location" in r.detail


class FakeDataset:
    def __init__(self):
        self.frames, self.episodes, self.cleared = [], [], 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.episodes.append(self.frames)
        self.frames = []

    def clear_episode_buffer(self):
        self.cleared += 1
        self.frames = []

    def finalize(self):
        pass


def test_recording_a_scripted_demo(make_cell):
    s = make_cell("scripted", vla={"fps": 20})
    s.ctrl.deg_per_s = 150.0           # real-robot pace, so a grasp spans many frames
    ds = FakeDataset()
    rec = EpisodeRecorder(None, None, 20, s.cell.rig.shapes(), dataset=ds)
    s.cell.recorder = Sampler(rec, s.cell.rig, s.robot, s.cell.gripper, log=lambda m: None)
    s.cell.hover_jitter = (0.02, 15.0)
    r = s.cell.run("pick the object 'red_block' place the 'bin_A'")
    assert r.ok, r.detail
    assert len(ds.episodes) == 1
    ep = ds.episodes[0]
    assert len(ep) >= 10
    states = np.array([f["observation.state"] for f in ep])
    actions = np.array([f["action"] for f in ep])
    assert states.shape[1] == actions.shape[1] == 7
    np.testing.assert_allclose(actions[:-1, :6], states[1:, :6])     # action = next joints
    assert actions[0, 6] == 0.0 and actions[-1, 6] == 1.0              # gripper: open ... closed
    assert ep[0]["task"] == "pick up the red block"
    assert ep[0]["observation.images.wrist"].shape == (480, 640, 3)
    assert tool_pose(states[-1, :6])[2, 3] > tool_pose(states[np.argmax(actions[:, 6])][:6])[2, 3] + 0.1  # lifted


def test_recorder_drops_failed_and_short_episodes():
    ds = FakeDataset()
    rec = EpisodeRecorder(None, None, 10, {"wrist": (4, 4, 3)}, dataset=ds, min_frames=3)
    img = {"wrist": np.zeros((4, 4, 3), np.uint8)}
    rec.begin("t")
    for i in range(5):
        rec.add(img, np.r_[np.full(6, float(i)), 0.0], 0.0)
    assert rec.end(keep=False) is False and ds.cleared == 1 and not ds.episodes
    rec.begin("t")
    rec.add(img, np.zeros(7), 1.0)
    assert rec.end(keep=True) is False          # 1 frame < min_frames
    rec.begin("t")
    for i in range(4):
        rec.add(img, np.r_[np.full(6, float(i)), 0.0], float(i >= 2))
    assert rec.end(keep=True) is True
    acts = np.array([f["action"] for f in ds.episodes[0]])
    assert acts[:, 0].tolist() == [1, 2, 3, 3] and acts[:, 6].tolist() == [0, 0, 1, 1]


def test_hand_guided_demo(make_cell):
    """Drag mode: the 'operator' moves the arm onto the block, closes, lifts."""
    from nova2_vla.geometry import tool_down
    from nova2_vla.twin import solve_ik
    s = make_cell("scripted", vla={"fps": 20})
    ds = FakeDataset()
    # the simulated hand moves for ~1 s (a real demo takes several): a short episode is enough here
    rec = EpisodeRecorder(None, None, 20, s.cell.rig.shapes(), dataset=ds, min_frames=3)
    s.cell.recorder = Sampler(rec, s.cell.rig, s.robot, s.cell.gripper, log=lambda m: None)
    box = s.world.box("green_bar")
    script = iter(["", "c", "", "y"])
    prompts = []

    def operator(prompt):
        prompts.append(prompt)
        step = len(prompts)
        q = s.ctrl.joints.copy()
        if step == 2:   # hand-guide down onto the bar, then close
            yaw = math.radians(box.yaw_deg)
            for z in np.linspace(tool_pose(q)[2, 3], -0.05 + box.height - 0.01, 20):
                q = solve_ik(tool_down(box.x, box.y, z, yaw), q)
                s.ctrl.drag_to(q)
                time.sleep(0.03)
        if step == 3:   # lift it
            T = tool_pose(q)
            for z in np.linspace(T[2, 3], T[2, 3] + 0.12, 20):
                q = solve_ik(tool_down(T[0, 3], T[1, 3], z, math.atan2(T[1, 0], T[0, 0])), q)
                s.ctrl.drag_to(q)
                time.sleep(0.03)
        return next(script)

    s.cell.operator = operator
    r = s.cell.pick_and_place(s.cell.objects.get("green_bar"), s.cell.locations.get("bin_A"), strategy="human")
    assert r.ok, r.detail
    assert r.grasped_by == "human"
    assert _placed_at(s, "green_bar", "bin_A")
    assert len(ds.episodes) == 1 and len(ds.episodes[0]) >= 3
    assert [n for n, _, _ in s.ctrl.log].count("StartDrag") == 1
    assert len(s.ctrl.connections) == 1 and not s.ctrl.feedback_connections     # one TCP connection


def test_collect_resets_itself(make_cell):
    from nova2_vla.collect import collect
    s = make_cell("scripted", vla={"fps": 10})
    s.ctrl.deg_per_s = 150.0
    ds = FakeDataset()
    rec = EpisodeRecorder(None, None, 10, s.cell.rig.shapes(), dataset=ds)
    s.cell.recorder = Sampler(rec, s.cell.rig, s.robot, s.cell.gripper, log=lambda m: None)
    s.cell.hover_jitter = (0.015, 10.0)
    spec = s.cell.objects.get("red_block")
    start = (s.world.box("red_block").x, s.world.box("red_block").y)
    assert collect(s.cell, spec, 2, "scripted", log=lambda m: None) == 2
    assert len(ds.episodes) == 2
    end = (s.world.box("red_block").x, s.world.box("red_block").y)
    assert math.hypot(end[0] - start[0], end[1] - start[1]) > 0.01      # moved to a new spot
    for other in ("blue_block", "green_bar"):                           # never dropped onto another
        b = s.world.box(other)
        assert math.hypot(end[0] - b.x, end[1] - b.y) >= 0.08 - 1e-6


def test_cli_dry_run(tmp_path, capsys):
    from nova2_vla.cli import main
    (tmp_path / "loc.json").write_text(json.dumps(LOCATIONS))
    cfg = tmp_path / "cell.yaml"
    cfg.write_text(f"paths:\n  runs_dir: {tmp_path / 'runs'}\n  locations_file: {tmp_path / 'loc.json'}\n"
                   "vla:\n  fps: 30\n")
    assert main(["--config", str(cfg), "--dry-run", "run", "pick the object 'red_block' place the 'tray_C'"]) == 0
    assert "DONE" in capsys.readouterr().out
    assert main(["--config", str(cfg), "--dry-run", "run", "pick the object 'nothing' place the 'tray_C'"]) == 2
    assert main(["--config", str(cfg), "list"]) == 0


def test_place_below_the_table_is_explained():
    from nova2_vla.pipeline import PickPlaceCell
    from nova2_vla.task import Location, TaskError
    cfg = load_config(None, {"robot": {"table_z_m": 0.018}})
    cell = PickPlaceCell(cfg, None, None, None, None, None, None, log=lambda m: None)   # planning only
    with pytest.raises(TaskError, match=r"68 mm below the table .*\"z\": \"table\""):
        cell.precheck_location(Location("bin_A", -0.25, 0.40, -0.05, 0.0, "surface"))
    cell.precheck_location(Location("bin_A", -0.25, 0.40, 0.018, 0.0, "surface"))      # on the table: fine
    cfg = load_config(None, {"robot": {"table_z_m": 0.018, "table_x_range_m": [-0.7, -0.3]}})
    beside = PickPlaceCell(cfg, None, None, None, None, None, None, log=lambda m: None)
    beside.precheck_location(Location("bin_low", -0.25, 0.40, -0.05, 0.0, "surface"))  # off the table: allowed
