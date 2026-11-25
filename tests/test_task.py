import json

import numpy as np
import pytest
from PIL import Image

from nova2_vla.task import LocationLibrary, ObjectLibrary, TaskError, parse_command


@pytest.mark.parametrize("text, obj, loc", [
    ("pick the object 'gear_housing' place the 'bin_A'", "gear_housing", "bin_A"),
    ('Pick the object "gear housing" place the "bin A"', "gear housing", "bin A"),
    ("pick gear_housing place bin_A", "gear_housing", "bin_A"),
    ("pick up the gear_housing and place it in bin_A.", "gear_housing", "bin_A"),
    ("please grab 'm6 bolt', then put it on 'tray 3'", "m6 bolt", "tray 3"),
    ("pick the object 'x' place the location 'y'", "x", "y"),
])
def test_parse_command(text, obj, loc):
    cmd = parse_command(text)
    assert (cmd.object_name, cmd.location_name) == (obj, loc)


@pytest.mark.parametrize("text", ["place bin_A", "pick gear", "", "move the arm home"])
def test_parse_command_rejects(text):
    with pytest.raises(TaskError):
        parse_command(text)


def _image(path, colour):
    Image.fromarray(np.full((32, 32, 3), colour, np.uint8)).save(path)


def test_object_library(tmp_path):
    (tmp_path / "gear_housing").mkdir()
    _image(tmp_path / "gear_housing" / "a.png", (10, 20, 200))
    _image(tmp_path / "gear_housing" / "b.JPG", (10, 20, 200))
    (tmp_path / "gear_housing" / "object.json").write_text(json.dumps({"description": "blue gear housing",
                                                                        "grip_torque": 250}))
    (tmp_path / "empty").mkdir()
    lib = ObjectLibrary(tmp_path)
    assert lib.names() == ["gear_housing"]
    spec = lib.get("Gear Housing")          # case, spaces and underscores don't matter
    assert spec.name == "gear_housing" and len(spec.images) == 2
    assert spec.instruction("pick up the {description}") == "pick up the blue gear housing"
    assert spec.meta["grip_torque"] == 250
    assert spec.load_images()[0].shape == (32, 32, 3)
    with pytest.raises(TaskError, match="did you mean 'gear_housing'"):
        lib.get("gear_housin")
    with pytest.raises(TaskError, match="no reference images"):
        lib.get("empty")


def test_object_without_meta_uses_folder_name(tmp_path):
    (tmp_path / "m6-bolt").mkdir()
    _image(tmp_path / "m6-bolt" / "a.png", (90, 90, 90))
    assert ObjectLibrary(tmp_path).get("m6-bolt").description == "m6 bolt"


def test_locations_mm_and_formats():
    locs = LocationLibrary.parse({
        "units": "mm",
        "locations": {
            "bin_A": {"x": -250, "y": 400, "z": -50, "w": 30},
            "tray": [-300, 100, 20, -90],
            "jig": {"x": 100, "y": -400, "z": 80, "rz": 45, "z_reference": "tool"},
        }})
    a = locs["bin_A"]
    assert (a.x, a.y, a.z, a.w_deg, a.z_reference) == (-0.25, 0.4, -0.05, 30.0, "surface")
    assert locs["tray"].xyz == pytest.approx((-0.3, 0.1, 0.02))
    assert locs["jig"].w_deg == 45.0 and locs["jig"].z_reference == "tool"


def test_locations_flat_file_in_metres(tmp_path):
    p = tmp_path / "loc.json"
    p.write_text(json.dumps({"units": "m", "bin": {"x": -0.25, "y": 0.4, "z": -0.05, "w": 0}}))
    lib = LocationLibrary.load(p)
    assert lib.get("BIN").x == -0.25


@pytest.mark.parametrize("data, match", [
    ({"units": "mm", "locations": {"a": {"x": -0.25, "y": 0.4, "z": 0.0, "w": 0}}}, "looks like metres"),
    ({"units": "m", "locations": {"a": {"x": -250, "y": 400, "z": 0, "w": 0}}}, "looks like millimetres"),
    ({"locations": {"a": {"x": 1, "y": 2, "z": 3}}}, "missing 'w'"),
    ({"locations": {"a": {"x": 1, "y": 2, "z": 3, "qx": 0, "qy": 0, "qz": 0, "qw": 1}}}, "quaternions"),
    ({"locations": {"a": [1, 2, 3]}}, r"\[x, y, z, w\]"),
    ({"units": "inch", "locations": {"a": [1, 2, 3, 4]}}, "units"),
    ({"locations": {"a": {"x": "left", "y": 2, "z": 3, "w": 0}}}, "numbers"),
    ({"locations": {}}, "no locations"),
])
def test_location_errors(data, match):
    with pytest.raises(TaskError, match=match):
        LocationLibrary.parse(data)


def test_camera_rename_map():
    from nova2_vla.train import camera_rename_map
    base = ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]
    w, s = "observation.images.wrist", "observation.images.scene"
    assert camera_rename_map([w, s], base) == {w: base[0], s: base[1]}
    assert camera_rename_map([w], [w]) == {}                       # already the policy's names
    assert camera_rename_map([w, s], [w]) == {}                    # policy uses a subset
    assert camera_rename_map([w], []) == {}
    with pytest.raises(ValueError, match="more"):
        camera_rename_map([w, s], ["observation.images.top"])      # 2 cameras, model takes 1


def test_location_on_the_table():
    locs = LocationLibrary.parse({"units": "mm", "locations": {
        "bin": {"x": -250, "y": 400, "z": "table", "w": 0},
        "jig": {"x": -300, "y": 100, "z": 120, "w": 0, "z_reference": "tool"}}}, table_z=0.018)
    assert locs["bin"].z == pytest.approx(0.018) and locs["bin"].z_reference == "surface"
    assert locs["jig"].z == pytest.approx(0.12)
    with pytest.raises(TaskError, match="table_z_m"):
        LocationLibrary.parse({"locations": {"bin": {"x": -250, "y": 400, "z": "table", "w": 0}}})
    with pytest.raises(TaskError, match="z_reference 'surface'"):
        LocationLibrary.parse({"locations": {"bin": {"x": -250, "y": 400, "z": "TABLE", "w": 0,
                                                     "z_reference": "tool"}}}, table_z=0.018)
