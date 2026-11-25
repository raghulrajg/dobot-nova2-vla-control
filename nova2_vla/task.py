"""
The operator's request and what it refers to.

    pick the object 'gear_housing' place the 'bin_A'

* the object name is a folder in objects/ holding reference photos of the
  product (any number, any angles; .jpg/.png/.bmp/.webp) and, optionally,
  object.json with hints:
      {"description": "blue plastic gear housing",   # used in the VLA instruction
       "grasp_depth_m": 0.015, "grasp_yaw_offset_deg": 0, "grip_torque": 250,
       "min_similarity": 0.6}
  A new product needs only a new folder: no retraining, no reprogramming.

* the location name is a key in the locations JSON file:
      {"units": "mm", "z_reference": "surface",
       "locations": {"bin_A": {"x": -250, "y": 400, "z": -50, "w": 0}}}
  x, y, z: where the object goes, in the robot base frame (the frame the
  pendant shows with user 0). w: gripper yaw about the vertical [deg], the
  pendant's Rz while the tool points straight down (Rx 180, Ry 0).
  z_reference "surface" (default): z is the surface the object is put on,
  and the release height follows from where the object was gripped;
  "tool": z is the gripper tip height at release.
  "z": "table" puts it on the table itself, at robot.table_z_m from the
  cell config, so re-measuring the table moves every such place with it.
  A location may also be a list [x, y, z, w], and may override z_reference.
"""

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TaskError(ValueError):
    pass


# ---------------------------------------------------------------- command

_NAME = r"""('[^']+'|"[^"]+"|`[^`]+`|[^\s,'"`]*[^\s,'"`.!?])"""   # bare names don't end in . ! ?
_COMMAND = re.compile(
    r"^\s*(?:please\s+)?(?:pick|grab|take)(?:\s+up)?\s+(?:the\s+)?(?:object\s+|item\s+|part\s+|product\s+)?"
    + _NAME
    + r"\s*,?\s*(?:and\s+|then\s+)*(?:place|put|drop|move)\s+(?:it\s+)?(?:(?:in|on|at|to|into|onto|in\s+to)\s+)?"
    r"(?:the\s+)?(?:location\s+|place\s+)?" + _NAME + r"\s*[.!]?\s*$",
    re.IGNORECASE,
)


@dataclass
class Command:
    object_name: str
    location_name: str


def parse_command(text: str) -> Command:
    """'pick the object 'x' place the 'y'' (and looser variants such as
    'pick x and place it in y') -> Command(x, y)."""
    m = _COMMAND.match(text)
    if not m:
        raise TaskError(f"cannot read {text!r}: say e.g. pick the object 'gear' place the 'bin_A'")
    return Command(*(g.strip("'\"`").strip() for g in m.groups()))


def _key(name: str) -> str:
    return re.sub(r"[\s_\-]+", "_", name.strip().lower())


def _lookup(name, table: dict, what):
    keys = {_key(k): k for k in table}
    k = keys.get(_key(name))
    if k is not None:
        return table[k]
    close = difflib.get_close_matches(_key(name), list(keys), n=3, cutoff=0.6)
    hint = f" -- did you mean {', '.join(repr(keys[c]) for c in close)}?" if close else ""
    known = ", ".join(sorted(table)) or "none"
    raise TaskError(f"unknown {what} {name!r}{hint} (known: {known})")


# ---------------------------------------------------------------- objects

@dataclass
class ObjectSpec:
    name: str
    folder: Path
    images: List[Path]
    description: str
    meta: Dict = field(default_factory=dict)

    def instruction(self, template: str) -> str:
        return template.format(name=self.name, description=self.description)

    def load_images(self):
        """Reference images as RGB uint8 arrays."""
        from PIL import Image
        import numpy as np
        out = []
        for p in self.images:
            with Image.open(p) as im:
                out.append(np.asarray(im.convert("RGB")))
        return out

    @property
    def stamp(self):
        """Changes whenever a reference image or object.json changes (cache key)."""
        files = self.images + [self.folder / "object.json"]
        return tuple((p.name, p.stat().st_mtime_ns) for p in files if p.exists())


class ObjectLibrary:
    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise TaskError(f"objects folder {self.root} does not exist")

    def names(self) -> List[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and self._images(p))

    @staticmethod
    def _images(folder: Path):
        return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())

    def get(self, name: str) -> ObjectSpec:
        folders = {p.name: p for p in self.root.iterdir() if p.is_dir()}
        folder = _lookup(name, folders, "object")
        images = self._images(folder)
        if not images:
            raise TaskError(f"object folder {folder} has no reference images ({', '.join(sorted(IMAGE_SUFFIXES))})")
        meta = {}
        meta_file = folder / "object.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except json.JSONDecodeError as e:
                raise TaskError(f"{meta_file}: {e}") from None
            if not isinstance(meta, dict):
                raise TaskError(f"{meta_file}: expected a JSON object")
        description = str(meta.get("description") or folder.name.replace("_", " ").replace("-", " "))
        return ObjectSpec(folder.name, folder, images, description, meta)


# ---------------------------------------------------------------- locations

@dataclass
class Location:
    name: str
    x: float          # base frame [m]
    y: float
    z: float
    w_deg: float      # gripper yaw about the vertical
    z_reference: str  # "surface" | "tool"

    @property
    def xyz(self):
        return (self.x, self.y, self.z)


_ALIASES = {"x": ("x",), "y": ("y",), "z": ("z",), "w": ("w", "yaw", "r", "rz")}


class LocationLibrary:
    def __init__(self, locations: Dict[str, Location], source="(inline)"):
        self.locations = locations
        self.source = source

    def names(self):
        return sorted(self.locations)

    def get(self, name: str) -> Location:
        return _lookup(name, self.locations, f"location in {self.source}")

    @classmethod
    def load(cls, path, table_z=None) -> "LocationLibrary":
        """table_z: the table height [m], for locations with "z": "table"."""
        path = Path(path)
        if not path.exists():
            raise TaskError(f"locations file {path} does not exist")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise TaskError(f"{path}: {e}") from None
        return cls(cls.parse(data, str(path), table_z), str(path))

    @staticmethod
    def parse(data, source="locations", table_z=None) -> Dict[str, Location]:
        if not isinstance(data, dict):
            raise TaskError(f"{source}: expected a JSON object")
        units = data.get("units", "mm")
        default_ref = data.get("z_reference", "surface")
        entries = data.get("locations", None)
        if entries is None:   # flat file: {"bin_A": {...}, ...}
            entries = {k: v for k, v in data.items() if k not in ("units", "z_reference")}
        if units not in ("mm", "m"):
            raise TaskError(f"{source}: units must be 'mm' or 'm', not {units!r}")
        if not entries:
            raise TaskError(f"{source}: no locations")
        scale = 0.001 if units == "mm" else 1.0

        out = {}
        for name, v in entries.items():
            where = f"{source}: location {name!r}"
            ref = default_ref
            if isinstance(v, (list, tuple)):
                if len(v) != 4:
                    raise TaskError(f"{where}: expected [x, y, z, w], got {v}")
                x, y, z, w = v
            elif isinstance(v, dict):
                if {"qx", "qy", "qz", "qw"} <= set(v):
                    raise TaskError(f"{where}: quaternions are not supported, give w as the yaw in degrees")
                vals = {}
                for k, aliases in _ALIASES.items():
                    found = [a for a in aliases if a in v]
                    if not found:
                        raise TaskError(f"{where}: missing {k!r}")
                    vals[k] = v[found[0]]
                x, y, z, w = vals["x"], vals["y"], vals["z"], vals["w"]
                ref = v.get("z_reference", ref)
            else:
                raise TaskError(f"{where}: expected {{x, y, z, w}} or [x, y, z, w]")
            if ref not in ("surface", "tool"):
                raise TaskError(f"{where}: z_reference must be 'surface' or 'tool'")
            on_table = isinstance(z, str) and z.strip().lower() == "table"
            if on_table:
                if ref != "surface":
                    raise TaskError(f"{where}: \"z\": \"table\" needs z_reference 'surface' "
                                    "(the gripper tip can't go down to the table)")
                if table_z is None:
                    raise TaskError(f"{where}: \"z\": \"table\" needs robot.table_z_m in the cell config")
                z = 0.0
            try:
                x, y, z, w = (float(a) for a in (x, y, z, w))
            except (TypeError, ValueError):
                raise TaskError(f"{where}: x, y, w must be numbers, z a number or \"table\"") from None
            # catch the wrong unit: the Nova 2 reaches 625 mm
            biggest = max(abs(x), abs(y), abs(z))
            if units == "mm" and 0 < biggest < 5:
                raise TaskError(f"{where}: ({x}, {y}, {z}) mm is within 5 mm of the base -- "
                                "the file looks like metres: set \"units\": \"m\"")
            if units == "m" and biggest > 5:
                raise TaskError(f"{where}: ({x}, {y}, {z}) m is out of reach -- "
                                "the file looks like millimetres: set \"units\": \"mm\"")
            z = float(table_z) if on_table else z * scale
            out[str(name)] = Location(str(name), x * scale, y * scale, z, w, ref)
        return out
