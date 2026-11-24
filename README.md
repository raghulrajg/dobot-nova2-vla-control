# Dobot Nova 2 — VLA pick and place

Tell the robot what to move, not how:

```
pick the object 'gear_housing' place the 'bin_A'
```

* `gear_housing` is a folder of reference photos of the product (`objects/gear_housing/`).
* `bin_A` is an entry `{x, y, z, w}` in a JSON file of place locations (`config/locations.json`).

A new product that arrives in the morning needs a folder of 3–6 photos —
no reprogramming, no retraining. The grasp itself is done by a pretrained
vision-language-action model (SmolVLA by default) fine-tuned once on this
cell, with a classical grasp as a fallback.

It builds on the Nova 2 digital twin,
[raghulrajg/manipulator-agent](https://github.com/raghulrajg/manipulator-agent)
(a git submodule in `third_party/`): its DH model, closed-form IK, table
check, Dobot TCP client and Feetech gripper driver are used unchanged.

## How it works

```
 "pick the object 'X' place the 'Y'"
        │
        ▼
 1  check Y is reachable (IK + table check)  ── no → refused, nothing moves
 2  home: wrist RGB-D camera looks down at the table
 3  FIND X      every object on the table (depth blobs + OWLv2 objectness)
                compared with objects/X/*.png (DINOv2 features) → best match
 4  LOCATE X    aligned depth + twin camera FK → position, yaw, size in the base frame
 5  PLAN        hover pose 15 cm above X, plus a top-down grasp (twin IK, table-safe)
 6  GRASP       VLA: camera images + joints + "pick up the <description>"
                  → joint targets at 15 Hz → SafetyFilter → ServoJ at 33 Hz
                until the gripper closes on something
                (fails? back up, home, look again, scripted top-down grasp)
 7  PLACE Y     straight up, carry (the hanging object is checked against the
                table), lower to Y, open, back up, home
```

**Why the VLA does the grasp and IK does the rest.** A VLA is good at
the part that varies with every new product — how to approach, where to
close the fingers, correcting when the part is not where it looked — and
weak at what a factory needs to be exact: "put it at x, y, z, w" and "never
touch the table". So the learned policy only drives the arm inside a small
cylinder around the object, every action it proposes passes a safety
filter built on the twin's kinematics, and the carry and place are planned
with the twin's IK.

**What "pretrained" does and doesn't buy.** No public VLA has seen a
Nova 2 with this camera and gripper, so none works zero-shot here. SmolVLA
comes pretrained on thousands of manipulation episodes and learns this
cell from a few hundred demos of your own (recorded by the cell itself, see
below). After that, new products of a similar kind need only reference
photos. A product with a very different shape or material may need a few
dozen more demos and a short re-train; until then the scripted grasp
covers it (`vla_with_fallback`).

## Repository

| Path | What |
|---|---|
| `nova2_vla/task.py` | the command parser, product folders, locations JSON |
| `nova2_vla/perception/` | cameras (RealSense, OpenCV), the reference-photo finder (depth / OWLv2 proposals, DINOv2 matching), 3D localisation |
| `nova2_vla/policy/` | the policy contract and the LeRobot wrapper (SmolVLA, π0.5, GR00T … any LeRobot checkpoint) |
| `nova2_vla/robot/` | Nova 2 ServoJ / drag mode / port-30004 feedback, the ServoJ streamer, the safety filter, a fake controller |
| `nova2_vla/pipeline.py` | the pick-and-place sequence and its failure handling |
| `nova2_vla/data/`, `collect.py` | demo recording (LeRobot dataset v3) and self-resetting collection |
| `nova2_vla/train.py` | fine-tuning through `lerobot-train` |
| `nova2_vla/sim.py` | simulated cell for `--dry-run` and the tests |
| `config/cell.yaml` | the cell: robot, gripper, cameras, models, speeds, safety limits |
| `config/locations.json` | place locations |
| `objects/` | one folder of reference photos per product ([objects/README.md](objects/README.md)) |
| `third_party/manipulator-agent` | the digital twin (submodule) |

## Setup (Ubuntu, RTX 5080)

```bash
git clone --recursive https://github.com/raghulrajg/dobot-nova2-vla-control.git
cd dobot-nova2-vla-control
```

If you cloned without `--recursive`:

```bash
git submodule update --init
```

The RTX 5080 is a Blackwell GPU (sm_120): it needs NVIDIA driver 570 or
newer and a PyTorch build for CUDA 12.8 or newer; older builds import fine
but have no kernels for it. LeRobot 0.7 needs Python 3.12+ (Ubuntu 24.04's
default).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[vision,realsense,vla,dev]"
python -m nova2_vla check
```

`check` prints the GPU, whether this torch build has sm_120 kernels, the
packages, and whether every location in `config/locations.json` is
reachable. Add `--robot` to also connect to the controller (read only).

Put your user in the `dialout` group for the gripper's USB-RS485 adapter.
The Nova 2 must be on controller firmware V4 (TCP/IP guide V4.x: ports
29999 and 30004).

**Try it without hardware first** (fake controller over TCP on localhost,
rendered wrist camera, simulated gripper, a stand-in policy):

```bash
python -m nova2_vla --dry-run run "pick the object 'red_block' place the 'bin_A'"
python -m nova2_vla --dry-run shell
pytest -q
```

## Commissioning (once)

1. **Table height.** With tool 0 selected, jog the tool tip onto the table,
   read the flange Z on the pendant, subtract 197 mm and enter it in metres
   as `robot.table_z_m` in `config/cell.yaml`. Nothing runs on the robot
   without it.
2. **Model check.** `python -m nova2_vla check --robot` compares the twin's
   FK with the controller's own pose (must be under 3 mm). If not, see
   "Correction to your DH table" in the twin's README.
3. **Wrist camera.** The twin's camera frame is the colour camera's optical
   frame (z = view axis). If detections land consistently turned or offset,
   set `cameras.wrist.optical_axis_rotation_deg`, or fix the camera table in
   the twin. Check a few objects with `run` at `speed: 10` first.
4. **Gripper axis.** Grasps put the tool's x axis across the narrow side of
   the part. If your fingers close along the flange's y axis, set
   `pick.grasp_yaw_offset_deg: 90`.
5. **Places beside the table.** The table is treated as an endless floor at
   `table_z_m`. If a bin sits lower than the table, set
   `robot.table_x_range_m` / `table_y_range_m` to the table's footprint.

## Daily use

**A new product:** make its folder of reference photos (in-cell photos match best):

```bash
python -m nova2_vla capture-refs gear_housing --views 4 --description "blue plastic gear housing"
```

**Locations:** `config/locations.json`:

```json
{
  "units": "mm",
  "z_reference": "surface",
  "locations": {
    "bin_A":     {"x": -250, "y": 400, "z": -50, "w": 0},
    "fixture_D": {"x": -350, "y": 300, "z": -20, "w": 90, "z_reference": "tool"}
  }
}
```

* `x, y, z` in the robot base frame (the pendant's coordinates with user
  frame 0), in `units` (`mm` or `m`; a file that looks like the other unit
  is refused).
* `w` is the gripper's turn about the vertical in degrees — the pendant's
  Rz while the tool points straight down (Rx 180, Ry 0). If `w` is out of
  reach, `w + 180` is tried (same grip for a parallel gripper; set
  `pick.try_flipped_place: false` if the part's orientation matters).
* `z_reference: "surface"` (default): `z` is the surface the part is set
  down on; the release height follows from where the part was gripped, so
  tall and short products land the same. `"tool"`: `z` is the gripper tip
  height at release, e.g. when you teach it by jogging.
* A location may also be written `[x, y, z, w]`.

**Run tasks:**

```bash
python -m nova2_vla run "pick the object 'gear_housing' place the 'bin_A'"
python -m nova2_vla shell     # then type one task per line; also: list, home, quit
python -m nova2_vla list
```

Every task writes `runs/<time>_<object>_to_<location>/` with
`attempt1.jpg` (every candidate and its similarity; the chosen one in green)
and `result.json` (outcome, object pose, the full log). Outcomes: `done`,
`not_found`, `rejected` (unknown name, unreachable location), `unreachable`
(the object can't be grasped safely), `grasp_failed`, `place_failed`. All
of them end with the arm at home; robot faults (alarm, collision, lost
connection) stop the robot instead.

## Teaching the VLA

**1. Collect demos.** The cell resets itself: each episode picks the product
from wherever it is and puts it down at a random free spot under the camera.

```bash
# scripted top-down grasps from a randomly offset hover pose (+-2 cm, +-15 deg)
python -m nova2_vla collect --object gear_housing --episodes 50
# hand-guided grasps (drag mode) for what a top-down script can't do;
# c+Enter closes, o+Enter opens, Enter finishes; the cell carries and resets
python -m nova2_vla collect --object odd_bracket --episodes 20 --mode human
```

Episodes go to `data/nova2_picks` as a LeRobot dataset: wrist (and scene)
video, `observation.state` = 6 joints in degrees + gripper closure,
`action` = the next step's joints + gripper command, and the instruction
`"pick up the <description>"`. Only grasps that end holding the part are
kept. Aim for a few hundred episodes over 10–20 products of the kinds you
handle, in varied positions; mixing in hand-guided demos teaches grasps the
script can't. `run --record` also saves successful grasps from normal
production, so the dataset keeps growing.

**2. Fine-tune** (SmolVLA, 450 M parameters, fits the 16 GB RTX 5080):

```bash
python -m nova2_vla train                      # lerobot/smolvla_base, 20k steps, batch 32
python -m nova2_vla train --batch 16           # if CUDA runs out of memory
python -m nova2_vla train --print-only         # just show the lerobot-train command
```

It runs `lerobot-train` and maps the cell's camera names onto the ones the
base model was pretrained with. The log shows the step rate, so you can judge how long 20k steps take. Then set

```yaml
vla:
  policy_path: outputs/train/nova2_smolvla/checkpoints/last/pretrained_model
```

**Bigger models.** π0.5 (`--base lerobot/pi05_base`) runs inference on 16 GB
but fine-tuning it needs far more memory: train on a rented GPU, copy the
checkpoint back and point `policy_path` at it. Any LeRobot checkpoint with
7-dim state and action loads the same way.

**3. Run with it.** `pick.strategy: vla_with_fallback` (default) tries the
VLA and falls back to the scripted grasp; `vla` uses only the VLA,
`scripted` never loads it. Watch the first runs at `speed: 10` and keep the
e-stop within reach.

## Safety

What a learned policy cannot do here, whatever it outputs:

| Check | Where | On violation |
|---|---|---|
| Joint limits (Nova 2 URDF, 1° margin) | SafetyFilter, `Nova2.servo_j` | clamped / refused |
| ≤ 2° per joint per policy step, ≤ 45°/s in the ServoJ stream | SafetyFilter, ServoStreamer | shortened |
| Arm links and camera ≥ 5 cm, gripper tip ≥ 5 mm above the table | SafetyFilter (twin `TableConstraint`) | stopped at the last safe point on the step |
| Tool tip within 8 cm of the object's axis and below hover + 5 cm | SafetyFilter pick zone | same |
| Tool tilted ≤ 45° from straight down | SafetyFilter | same |
| 15 unsafe proposals in a row | VLA loop | VLA stopped, fallback |
| Alarm, collision, disable during streaming | ServoStreamer (RobotMode via port 30004) | Stop(), task aborted |

Everything outside the VLA grasp is planned before it moves: every
waypoint and joint move is checked against the table with the twin's IK
(as in `pick_runner.py`), the carry move with the hanging part's height
added, and the gripper only closes or opens after the joints (FK) and the
controller's own pose both confirm the tool is at the target.

## Configuration

`config/cell.yaml` is commented; every key and its default is in
`nova2_vla/config.py`. Unknown keys are an error, so a typo can't silently
fall back to a default. The pieces you'll touch most:

* `robot.table_z_m`, `robot.speed`, `robot.line_speed`
* `cameras.*` — the wrist RealSense (needed: depth places the object); an
  optional fixed `scene` camera gives the VLA more context
* `perception.proposer` (`depth+owlv2`), `perception.min_similarity`
* `vla.policy_path`, `vla.fps` (must match the dataset), `vla.task_template`
* `pick.strategy`, `pick.hover_height_m`, `pick.grasp_depth_m`
* `safety.*`

## Notes

* Detection runs at the home pose; the product must be visible to the
  wrist camera from there (move `robot.home_servo_deg` to cover another
  area).
* Depth places the object, so very flat (< 6 mm), transparent or mirror
  parts need care: OWLv2 still proposes them, but their 3D position needs
  depth readings.
* Hand-guided demos use the Nova 2's drag mode (`StartDrag`/`StopDrag`);
  ServoJ streaming follows the TCP/IP guide V4's recommendation of a
  set-point every 30 ms.
* The digital twin in `third_party/manipulator-agent` is GPL-3.0.
