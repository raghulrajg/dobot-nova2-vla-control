"""
Collecting grasp demonstrations for fine-tuning the VLA, with the cell
resetting itself: every episode picks the product from wherever it lies and
puts it down at a random free spot on the table, ready for the next one.

  scripted  the twin's top-down grasp from a hover pose jittered by a few cm
            and degrees, so the policy also learns to correct its approach
  human     you hand-guide the arm (drag mode) for grasps a top-down
            script can't do; the cell still does the carrying and resetting

Only episodes where the gripper ends up holding the object are saved.
"""

import math

import numpy as np

from .perception.grounding import SceneContext
from .perception.localize import LocalizationError, localize
from .task import Location, TaskError


def default_area(cell, margin=0.12):
    """Table area under the wrist camera at home, +-margin around its axis."""
    T = cell.rig.T_base_camera(cell.wrist, cell.home)
    x, y = T[0, 3], T[1, 3]
    return (x - margin, x + margin, y - margin, y + margin)


def object_centers(cell):
    """xy of everything on the table, seen from home."""
    q = cell.robot.get_angle()
    frame = cell.rig.read(cell.wrist)
    T = cell.rig.T_base_camera(cell.wrist, q)
    centers = []
    for c in cell.finder.propose(frame, SceneContext(frame.K, T, cell.table_z)):
        try:
            centers.append(localize(c.box, frame.depth, frame.K, T, cell.table_z, c.mask).xy)
        except LocalizationError:
            pass
    return centers


def free_spot(cell, area, min_dist=0.08, tries=300, rng=None) -> Location:
    rng = rng or np.random.default_rng()
    cell.go_home()
    centers = object_centers(cell)
    x0, x1, y0, y1 = area
    for _ in range(tries):
        xy = rng.uniform((x0, y0), (x1, y1))
        if any(math.hypot(*(xy - c)) < min_dist for c in centers):
            continue
        loc = Location("random_spot", float(xy[0]), float(xy[1]), cell.table_z, float(rng.uniform(-90, 90)),
                       "surface")
        try:
            cell.precheck_location(loc)
        except TaskError:
            continue
        return loc
    raise TaskError(f"no free, reachable spot in x {x0:.2f}..{x1:.2f}, y {y0:.2f}..{y1:.2f} m "
                    f"at least {min_dist * 100:.0f} cm from the {len(centers)} objects on the table")


def collect(cell, spec, episodes, mode="scripted", area=None, max_failures_in_row=5, log=print):
    if cell.recorder is None:
        raise TaskError("no recorder: collect needs a dataset to write to")
    area = area or default_area(cell)
    rng = np.random.default_rng()
    done = failed = in_row = 0
    while done < episodes:
        loc = free_spot(cell, area, rng=rng)
        log(f"--- episode {done + 1}/{episodes}: {spec.name} -> ({loc.x:.3f}, {loc.y:.3f}), w {loc.w_deg:.0f}")
        saved_before = cell.recorder.recorder.saved_episodes
        r = cell.pick_and_place(spec, loc, strategy=mode)
        if r.ok and cell.recorder.recorder.saved_episodes > saved_before:
            done += 1
            in_row = 0
            continue
        in_row += 1
        if r.ok:   # moved fine, but the recorder dropped it (too short, or refused by the operator)
            log("episode not kept")
        else:
            failed += 1
            log(f"episode failed ({r.state}): {r.detail}")
            if r.state == "place_failed":
                raise TaskError(f"stopped: {r.detail} (take the part out of the gripper)")
        if in_row >= max_failures_in_row:
            raise TaskError(f"stopped after {in_row} episodes in a row that were not kept; last: "
                            f"{r.detail if not r.ok else 'dropped by the recorder'}")
    log(f"collected {done} episodes ({failed} failed attempts); saved so far: {cell.recorder.recorder.saved_episodes}")
    return done
