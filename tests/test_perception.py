import copy
import math

import numpy as np
import pytest

from nova2_vla.config import CameraConfig
from nova2_vla.geometry import camera_pose, home_servo_deg
from nova2_vla.perception.grounding import (
    ColorHistEmbedder, DepthProposer, ReferenceObjectFinder, SceneContext, box_iou, merge_candidates, Candidate,
)
from nova2_vla.perception.localize import LocalizationError, localize
from nova2_vla.sim import DEMO_BOXES, SimCamera, SimWorld, write_demo_objects
from nova2_vla.task import ObjectLibrary

TABLE_Z = -0.05


@pytest.fixture
def scene(tmp_path):
    world = SimWorld(TABLE_Z, copy.deepcopy(DEMO_BOXES), joints=home_servo_deg)
    cam = SimCamera("wrist", CameraConfig(type="sim"), world)
    frame = cam.read()
    ctx = SceneContext(frame.K, camera_pose(home_servo_deg()), TABLE_Z)
    write_demo_objects(tmp_path, world)
    finder = ReferenceObjectFinder([DepthProposer()], ColorHistEmbedder(), min_similarity=0.5)
    return world, frame, ctx, ObjectLibrary(tmp_path), finder


def _yaw_diff_mod180(a_deg, b_deg):
    return abs((a_deg - b_deg + 90) % 180 - 90)


@pytest.mark.parametrize("name", [b.name for b in DEMO_BOXES])
def test_find_and_localize_each_object(scene, name):
    world, frame, ctx, objects, finder = scene
    res = finder.find(objects.get(name), frame, ctx)
    assert res.target is not None, res.reason
    assert len(res.candidates) == 3
    obj = localize(res.target.box, frame.depth, frame.K, ctx.T_base_camera, TABLE_Z, res.target.mask)
    truth = world.box(name)
    assert obj.xy == pytest.approx([truth.x, truth.y], abs=0.004)
    assert obj.top_z == pytest.approx(TABLE_Z + truth.height, abs=0.003)
    assert obj.height == pytest.approx(truth.height, abs=0.003)
    assert obj.width == pytest.approx(truth.width, abs=0.005)
    assert obj.length == pytest.approx(truth.length, abs=0.006)
    yaw = math.degrees(math.atan2(obj.T_base_object[1, 0], obj.T_base_object[0, 0]))
    if truth.length - truth.width > 0.01:
        assert _yaw_diff_mod180(yaw, truth.yaw_deg) < 4
    else:   # square: either pair of sides, but along the edges, not the diagonal
        assert abs((yaw - truth.yaw_deg + 45) % 90 - 45) < 4
    R = obj.T_base_object[:3, :3]
    assert np.linalg.det(R) == pytest.approx(1.0) and R[:, 2] == pytest.approx([0, 0, 1])


def test_unknown_object_is_not_found(scene, tmp_path):
    world, frame, ctx, objects, finder = scene
    yellow = copy.deepcopy(DEMO_BOXES[0])
    yellow.name, yellow.color = "yellow_block", (230, 220, 30)
    write_demo_objects(tmp_path, SimWorld(TABLE_Z, [yellow]))
    res = finder.find(ObjectLibrary(tmp_path).get("yellow_block"), frame, ctx)
    assert res.target is None and "not found" in res.reason


def test_empty_table(scene):
    world, frame, ctx, objects, finder = scene
    world.boxes = []
    empty = SimCamera("wrist", CameraConfig(type="sim"), world).read()
    res = finder.find(objects.get("red_block"), empty, ctx)
    assert res.target is None and "nothing" in res.reason


def test_localize_needs_depth(scene):
    _, frame, ctx, _, _ = scene
    with pytest.raises(LocalizationError, match="depth"):
        localize([0, 0, 50, 50], None, frame.K, ctx.T_base_camera, TABLE_Z)
    with pytest.raises(LocalizationError, match="above the table"):
        localize([0, 0, 30, 30], frame.depth, frame.K, ctx.T_base_camera, TABLE_Z)   # bare table


def test_merge_prefers_masks():
    a = Candidate(np.array([0, 0, 10, 10.0]), 0.9, "owlv2")
    b = Candidate(np.array([1, 1, 10, 10.0]), 0.2, "depth", mask=np.ones((2, 2), bool))
    c = Candidate(np.array([50, 50, 60, 60.0]), 0.5, "owlv2")
    kept = merge_candidates([a, b, c])
    assert b in kept and a not in kept and c in kept
    assert box_iou(a.box, a.box) == pytest.approx(1.0)
