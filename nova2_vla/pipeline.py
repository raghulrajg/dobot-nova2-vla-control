"""
PickPlaceCell: "pick the object 'X' place the 'Y'" from start to finish.

  0. check Y is reachable before anything moves
  1. home (the detection pose), wrist camera looks down at the table
  2. find X among the objects on the table with its reference photos
  3. 3D pose from depth: where it stands, how it is turned, how tall/wide
  4. plan the hover pose above it, and a classic top-down grasp as a
     fallback, with the twin's table-safe IK (nothing moves if this fails)
  5. MovJ to the hover pose
  6. grasp:
       vla       the policy drives the arm (ServoJ, every step through the
                 SafetyFilter, confined to a cylinder around X) until the
                 gripper closes on something
       scripted  MovL straight down, close, MovL up (the twin's plan)
       vla_with_fallback   vla first; if it fails, back up, home, look
                 again, scripted
  7. lift straight up, carry to Y (checked so the hanging object clears the
     table), lower, open, back up, home

Every step that can fail for a reason in the world (object not there, out
of reach, missed grip) ends with the arm home and a TaskResult saying why.
Robot faults (alarm, collision, lost connection) stop the robot and raise.
"""

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .geometry import home_servo_deg, tool_down, tool_pose, yaw_of
from .perception.grounding import SceneContext, annotate
from .perception.localize import LocalizationError, localize
from .policy.base import PolicyError
from .robot.safety import PickZone, SafetyFilter
from .robot.servo import ServoStreamer
from .task import Location, ObjectSpec, TaskError, parse_command
from .twin import (
    T_FLANGE_TOOL, DobotError, IKError, ManipulatorAgent, TableConstraint, fk, joint_map, plan_place,
    solve_ik,
)

ROT_TOL_DEG = 1.0


class GraspFailed(RuntimeError):
    pass


class ObjectNotFound(TaskError):
    pass


@dataclass
class TaskResult:
    ok: bool
    state: str                 # done | not_found | unreachable | grasp_failed | place_failed | rejected
    detail: str
    object_name: str = ""
    location_name: str = ""
    grasped_by: str = ""       # vla | scripted
    object_xyz: Optional[list] = None
    duration_s: float = 0.0
    run_dir: str = ""
    log: list = field(default_factory=list)


@dataclass
class GraspOutcome:
    ok: bool
    detail: str
    tip_z: Optional[float] = None    # tool tip height when the fingers closed
    steps: int = 0


def tool_error(q, T_target):
    T = tool_pose(q)
    R = T[:3, :3].T @ T_target[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    return float(np.linalg.norm(T[:3, 3] - T_target[:3, 3])), ang


class PickPlaceCell:
    def __init__(self, cfg, robot, gripper, rig, finder, objects, locations, policy=None,
                 recorder=None, record_vla=False, hover_jitter=(0.0, 0.0), log=print, operator=input):
        self.cfg = cfg
        self.robot, self.gripper, self.rig = robot, gripper, rig
        self.finder, self.objects, self.locations = finder, objects, locations
        self.policy = policy
        self.recorder = recorder              # nova2_vla.data.Sampler, or None
        self.record_vla = record_vla
        self.hover_jitter = hover_jitter      # (metres, degrees) random offset of the hover pose, for demos
        self._print = log
        self.operator = operator              # prompt -> reply, for hand-guided demos
        self._lines = []
        self.home = (np.asarray(cfg.robot.home_servo_deg, float) if cfg.robot.home_servo_deg
                     else home_servo_deg())
        self.table_z = cfg.robot.table_z_m
        if self.table_z is None:
            raise TaskError("robot.table_z_m is not set: measure the table height once (see README)")
        s = cfg.safety
        self.table = TableConstraint(self.table_z, s.link_clearance_m, s.tip_clearance_m,
                                     cfg.robot.table_x_range_m, cfg.robot.table_y_range_m)
        self.wrist = cfg.wrist_camera
        self.rng = np.random.default_rng()

    def log(self, msg):
        self._lines.append(msg)
        self._print(msg)

    # ================================================================ robot helpers
    def startup(self):
        try:
            self.robot.request_control()
        except DobotError as e:
            self.log(f"RequestControl: {e} (already in TCP mode?)")
        self.robot.enable()
        if self.cfg.robot.speed_factor is not None:
            self.robot.speed_factor(self.cfg.robot.speed_factor)
        self.check_model()
        self.gripper.open()
        self.go_home()

    def check_model(self):
        """The controller's flange (GetPose, tool 0) must match the twin's FK
        of its joints, or every IK target would be off."""
        q = self.robot.get_angle()
        measured = self.robot.get_pose()[:3] / 1000
        model = fk(joint_map.servo_to_visual(q), "flange")[:3, 3]
        err = float(np.linalg.norm(measured - model) * 1000)
        self.log(f"model check: controller vs twin FK {err:.2f} mm")
        if err > self.cfg.robot.model_tol_mm:
            raise DobotError(f"the twin's kinematic model is {err:.1f} mm off the controller's pose: "
                             "see 'Correction to your DH table' in the manipulator-agent README"
                             + getattr(self.robot, "pose_hint", ""))

    def move_j(self, q, what="MovJ", table=None):
        """Joint move after checking its whole path against the table."""
        q = np.asarray(q, float)
        bad = (table or self.table).path_violations(joint_map.servo_to_visual(self.robot.get_angle()),
                                                    joint_map.servo_to_visual(q))
        if bad:
            raise IKError(f"{what}: the joint move goes below the table: " + ", ".join(bad))
        r = self.cfg.robot
        return self.robot.move_j(q, v=r.speed, a=r.speed, what=what)

    def move_l(self, q, T_tool, what="MovL"):
        tol = self.cfg.robot.reach_tol_mm

        def reached(qq):
            dp, dang = tool_error(qq, T_tool)
            return dp * 1000 <= tol and dang <= ROT_TOL_DEG
        r = self.cfg.robot
        return self.robot.move_l(q, reached, v=r.line_speed, a=r.line_speed, what=what)

    def at_target(self, T_tool, what):
        """Measure before using the gripper: joints (FK) and the controller's
        own flange position must both be at the target."""
        q = self.robot.get_angle()
        dp, dang = tool_error(q, T_tool)
        flange = (T_tool @ np.linalg.inv(T_FLANGE_TOOL))[:3, 3]
        dctrl = float(np.linalg.norm(self.robot.get_pose()[:3] / 1000 - flange) * 1000)
        r = self.cfg.robot
        if dp * 1000 > r.reach_tol_mm or dang > ROT_TOL_DEG or dctrl > r.reach_tol_mm + r.model_tol_mm:
            raise DobotError(f"not at {what}: tool tip {dp * 1000:.1f} mm / {dang:.1f} deg off, "
                             f"controller flange {dctrl:.1f} mm off" + getattr(self.robot, "pose_hint", ""))

    def go_home(self):
        q = self.robot.get_angle()
        if np.abs(q - self.home).max() > 0.3:
            self.move_j(self.home, "home")

    def straight_up(self, z_min, what="lift", table=None):
        """MovL straight up from where the tool is to at least z_min."""
        q = self.robot.get_angle()
        T = tool_pose(q)
        if T[2, 3] >= z_min - 1e-4:
            return q
        T_up = T.copy()
        T_up[2, 3] = z_min
        q_up = solve_ik(T_up, q, table or self.table)
        self.move_l(q_up, T_up, what)
        return q_up

    # ================================================================ the task
    def run(self, text: str) -> TaskResult:
        try:
            cmd = parse_command(text)
            spec, loc = self.objects.get(cmd.object_name), self.locations.get(cmd.location_name)
        except TaskError as e:
            return TaskResult(False, "rejected", str(e))
        return self.pick_and_place(spec, loc)

    def pick_and_place(self, spec: ObjectSpec, loc: Location, strategy: str = None) -> TaskResult:
        """strategy: None = pick.strategy from the config; also "human"
        (hand-guided grasp in drag mode, for recording demos)."""
        t0 = time.monotonic()
        self._lines = []
        run_dir = self._run_dir(spec.name, loc.name)
        result = TaskResult(False, "", "", spec.name, loc.name, run_dir=str(run_dir))
        try:
            self._pick_and_place(spec, loc, run_dir, result, strategy or self.cfg.pick.strategy)
        except ObjectNotFound as e:
            result.state, result.detail = "not_found", str(e)
        except TaskError as e:
            result.state, result.detail = result.state or "rejected", str(e)
        except (IKError, LocalizationError) as e:
            result.state, result.detail = result.state or "unreachable", str(e)
        except GraspFailed as e:
            result.state, result.detail = "grasp_failed", str(e)
        finally:
            result.duration_s = round(time.monotonic() - t0, 2)
            result.log = list(self._lines)
            self._save(run_dir, result)
        if not result.ok:
            self.log(f"[{result.state}] {result.detail}")
            self._recover()
        return result

    def _pick_and_place(self, spec, loc, run_dir, result, strategy):
        self.log(f"task: pick {spec.name!r} ({spec.description}) -> place at {loc.name!r} "
                 f"({loc.x:.3f}, {loc.y:.3f}, {loc.z:.3f}) m, w {loc.w_deg:.1f} deg")
        self.precheck_location(loc)
        # per-product squeeze, back to the default for products that don't set one
        self.gripper.set_torque(int(spec.meta.get("grip_torque", self.cfg.gripper.torque_limit)))

        if strategy in ("vla", "vla_with_fallback") and self.policy is None:
            if strategy == "vla":
                raise TaskError("pick.strategy is vla but no policy is loaded (vla.policy_path)")
            self.log("no VLA policy loaded: using the scripted grasp")
            strategy = "scripted"
        attempts = {"vla": ["vla"], "scripted": ["scripted"], "vla_with_fallback": ["vla", "scripted"],
                    "human": ["human"]}[strategy]
        grasp = {"vla": lambda obj, plan: self.vla_grasp(spec, obj, plan.waypoints[0]),
                 "scripted": lambda obj, plan: self.scripted_grasp(spec, plan),
                 "human": lambda obj, plan: self.human_grasp(spec)}

        outcome = None
        for i, how in enumerate(attempts):
            self.go_home()
            obj, plan, agent = self.detect_and_plan(spec, run_dir, tag=f"attempt{i + 1}")
            result.object_xyz = np.round(obj.T_base_object[:3, 3], 4).tolist()
            hover = plan.waypoints[0]
            self.move_to_hover(hover)
            try:
                outcome = grasp[how](obj, plan)
            except PolicyError as e:
                outcome = GraspOutcome(False, f"policy error: {e}")
            if outcome.ok:
                result.grasped_by = how
                break
            self.log(f"{how} grasp failed: {outcome.detail}")
            self.retreat(hover)
        if not outcome.ok:
            raise GraspFailed(outcome.detail)
        self.lift_and_place(spec, loc, outcome.tip_z, result)
        result.ok, result.state = True, "done"
        result.detail = (f"{spec.name} placed at {loc.name} by the {result.grasped_by} grasp "
                         f"(object was at {result.object_xyz})")
        self.log(f"[done] {result.detail}")

    def lift_and_place(self, spec, loc, tip_z, result):
        """Holding the object gripped at tool tip height tip_z: straight up,
        carry, lower onto loc, open, back up, home."""
        cfg = self.cfg
        h_tip = tip_z - self.table_z             # the object hangs this far below the tool tip
        carry = TableConstraint(self.table_z, cfg.safety.link_clearance_m, h_tip + 0.01,
                                cfg.robot.table_x_range_m, cfg.robot.table_y_range_m)
        q_lift = self.straight_up(tip_z + cfg.pick.hover_height_m, "lift")
        if not self.gripper.last or not self.gripper.last.holding:
            raise GraspFailed("lost the object")
        self.log(f"holding {spec.name!r} ({h_tip * 100:.1f} cm below the tool tip)")

        try:
            line = self.plan_place(loc, h_tip, q_lift, carry)
        except IKError as e:
            result.state = "place_failed"
            raise IKError(f"cannot put it at {loc.name!r}: {e} -- still holding it") from None
        (_, T_pre, q_pre), (_, T_put, q_put) = line[0], line[-1]
        self.move_j(q_pre, "pre_place", table=carry)
        self.move_l(q_put, T_put, "place")
        self.at_target(T_put, "place")
        self.gripper.open()
        self.move_l(q_pre, T_pre, "retreat")
        self.go_home()

    # ================================================================ steps
    def precheck_location(self, loc: Location):
        """Reachable at all? Checked with a typical release height before
        anything moves; the real plan follows once the grip is known."""
        t = self.table
        over_table = all(r is None or r[0] <= v <= r[1] for v, r in ((loc.x, t.x_range), (loc.y, t.y_range)))
        lowest = self.table_z if loc.z_reference == "surface" else self.table_z + t.tip_clearance
        if over_table and loc.z < lowest - 0.001:
            what = "surface" if loc.z_reference == "surface" else "gripper tip height (z_reference tool)"
            raise TaskError(
                f"location {loc.name!r}: its {what} z = {loc.z * 1000:.0f} mm is {(lowest - loc.z) * 1000:.0f} mm "
                f"below the table (robot.table_z_m = {self.table_z * 1000:.0f} mm). For a place on the table use "
                f"\"z\": \"table\" (or {lowest * 1000:.0f}); for a place beside the table and lower, set "
                "robot.table_x_range_m / table_y_range_m to the table's footprint")
        tip_z = loc.z + 0.03 if loc.z_reference == "surface" else loc.z
        errors = []
        for yaw in self._place_yaws(loc):
            try:
                plan_place(self._place_agent(), self.table, tool_down(loc.x, loc.y, tip_z, math.radians(yaw)),
                           self.home)
                return
            except IKError as e:
                errors.append(str(e))
        raise TaskError(f"location {loc.name!r} can't be reached safely: " + " | ".join(dict.fromkeys(errors)))

    def _place_yaws(self, loc):
        return [loc.w_deg] + ([loc.w_deg + 180.0] if self.cfg.pick.try_flipped_place else [])

    def _agent(self, **kw):
        c = self.cfg
        return ManipulatorAgent(table_z=self.table_z, link_clearance=c.safety.link_clearance_m,
                                tip_clearance=c.safety.tip_clearance_m, table_x_range=c.robot.table_x_range_m,
                                table_y_range=c.robot.table_y_range_m, approach_height=c.pick.hover_height_m,
                                **kw)

    def _place_agent(self):
        agent = self._agent()
        agent.approach_height = self.cfg.pick.place_approach_m
        return agent

    def detect_and_plan(self, spec, run_dir, tag="detect"):
        cfg = self.cfg
        q = self.robot.get_angle()
        t_still = time.monotonic()
        frame = None
        for _ in range(max(1, cfg.perception.detect_frames)):
            frame = self.rig.read(self.wrist, newer_than=t_still)
            t_still = frame.stamp
        if np.abs(self.robot.get_angle() - q).max() > 0.05:
            raise DobotError("the arm moved while the camera was looking")
        T_base_camera = self.rig.T_base_camera(self.wrist, q)
        ctx = SceneContext(frame.K, T_base_camera, self.table_z)
        found = self.finder.find(spec, frame, ctx)
        annotate(frame.rgb, found, run_dir / f"{tag}.jpg")
        self.log(found.reason)
        if found.target is None:
            raise ObjectNotFound(found.reason)

        obj = localize(found.target.box, frame.depth, frame.K, T_base_camera, self.table_z,
                       found.target.mask, cfg.perception.min_object_height_m)
        self.log(f"{spec.name} at ({obj.xy[0]:.3f}, {obj.xy[1]:.3f}) m, top z {obj.top_z:.3f} m, "
                 f"{obj.length * 100:.1f} x {obj.width * 100:.1f} x {obj.height * 100:.1f} cm, "
                 f"yaw {math.degrees(yaw_of(obj.T_base_object)):.0f} deg")
        if obj.width > cfg.gripper.max_opening_m - 0.005:
            raise TaskError(f"{spec.name} is {obj.width * 100:.1f} cm across its narrow side; the gripper "
                            f"opens {cfg.gripper.max_opening_m * 100:.1f} cm")

        depth = min(float(spec.meta.get("grasp_depth_m", cfg.pick.grasp_depth_m)), 0.5 * obj.height)
        tip_z = max(obj.top_z - depth, self.table_z + cfg.safety.tip_clearance_m + 0.002)
        yaw_off = float(spec.meta.get("grasp_yaw_offset_deg", cfg.pick.grasp_yaw_offset_deg))
        agent = self._agent(grasp_height=tip_z - obj.top_z, grasp_yaw_offset_deg=yaw_off)
        T_camera_object = np.linalg.inv(T_base_camera) @ obj.T_base_object
        plan = agent.plan_pick(T_camera_object, q, q)
        return obj, plan, agent

    def move_to_hover(self, hover):
        q_hover = hover.q_servo_deg
        jit_m, jit_deg = self.hover_jitter
        if jit_m or jit_deg:      # demos: start from a slightly different pose each time
            T = hover.T_base_tool
            dx, dy = self.rng.uniform(-jit_m, jit_m, 2)
            yaw = yaw_of(T) + math.radians(self.rng.uniform(-jit_deg, jit_deg))
            T_j = tool_down(T[0, 3] + dx, T[1, 3] + dy, T[2, 3], yaw)
            try:
                q_hover = solve_ik(T_j, hover.q_servo_deg, self.table)
            except IKError:
                q_hover = hover.q_servo_deg
        self.move_j(q_hover, "hover")

    def scripted_grasp(self, spec, plan) -> GraspOutcome:
        wp = plan.waypoints
        pre = wp[0]
        grasp = next(w for w in wp if w.name == "grasp")
        sampler = self._start_recording(spec)
        ok = False
        try:
            if np.abs(self.robot.get_angle() - pre.q_servo_deg).max() > 0.3:
                self.move_l(pre.q_servo_deg, pre.T_base_tool, "center")   # jittered hover -> above the object
            self.move_l(grasp.q_servo_deg, grasp.T_base_tool, "grasp")
            self.at_target(grasp.T_base_tool, "grasp")
            grip = self.gripper.close()
            tip_z = float(grasp.T_base_tool[2, 3])
            if not grip.holding:
                self.gripper.open()
                self.move_l(pre.q_servo_deg, pre.T_base_tool, "lift")
                return GraspOutcome(False, "the gripper closed on nothing", tip_z)
            self.move_l(pre.q_servo_deg, pre.T_base_tool, "lift")
            ok = bool(self.gripper.last and self.gripper.last.holding)
            return GraspOutcome(ok, "scripted grasp", tip_z)
        finally:
            if sampler:
                sampler.finish(keep=ok)

    def vla_grasp(self, spec, obj, hover) -> GraspOutcome:
        cfg, v = self.cfg, self.cfg.vla
        task = spec.instruction(v.task_template)
        s = cfg.safety
        safety = SafetyFilter(self.table, s.max_joint_step_deg, s.max_tilt_deg, s.workspace_x_m, s.workspace_y_m,
                              s.workspace_z_max_m)
        safety.set_zone(PickZone(tuple(obj.xy), s.pick_zone_radius_m,
                                 float(hover.T_base_tool[2, 3]) + s.pick_zone_above_hover_m))
        cams = v.cameras or list(self.rig.cameras)
        self.log(f"VLA grasp: {task!r}")
        self.policy.reset()
        sampler = self._start_recording(spec) if self.record_vla else None
        streamer = ServoStreamer(self.robot, s.servo_period_s, s.max_joint_speed_deg_s).start(self.robot.joints())
        queue, closed, missed, rejected, after, tip_z, step = [], False, 0, 0, 0, None, 0
        dt = 1.0 / v.fps
        next_t = time.monotonic()
        outcome = GraspOutcome(False, f"no grasp within {v.max_steps} steps ({v.max_steps * dt:.0f} s)")
        try:
            for step in range(v.max_steps):
                if not queue:
                    frames = self.rig.read_all(cams)
                    state = np.concatenate([self.robot.joints(), [self.gripper.state]])
                    chunk = np.asarray(self.policy.predict_chunk({k: f.rgb for k, f in frames.items()}, state, task))
                    if chunk.ndim != 2 or chunk.shape[1] < 7 or not np.all(np.isfinite(chunk)):
                        raise PolicyError(f"policy returned a bad action chunk {chunk.shape}")
                    queue = list(chunk[: max(1, v.execute_steps)])
                a = queue.pop(0)
                safe = safety.filter(streamer.goal, a[:6])
                if safe.rejected:
                    rejected += 1
                    queue = []          # re-plan from where the arm really is
                    if rejected >= s.max_rejected_steps:
                        outcome = GraspOutcome(False, "policy kept proposing unsafe moves: " + "; ".join(safe.reasons),
                                               steps=step)
                        break
                else:
                    rejected = 0
                    streamer.set_goal(safe.q)

                g = float(a[6])
                if not closed and g > v.gripper_close_above:
                    streamer.settle()
                    tip_z = float(tool_pose(self.robot.joints())[2, 3])
                    grip = self.gripper.close()
                    if grip.holding:
                        closed = True
                        self.log(f"VLA closed the gripper at step {step}, tool tip z {tip_z:.3f} m: holding")
                    else:
                        self.gripper.open()
                        missed += 1
                        queue = []
                        self.log(f"VLA closed the gripper on nothing ({missed}/{v.max_missed_grips + 1})")
                        if missed > v.max_missed_grips:
                            outcome = GraspOutcome(False, f"{missed} missed grips", steps=step)
                            break
                elif closed and g < v.gripper_open_below:
                    self.log("VLA opened the gripper again")
                    self.gripper.open()
                    closed = False
                if closed:
                    after += 1
                    if after >= v.post_grasp_steps:
                        break
                next_t += dt
                sleep = next_t - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.monotonic()
            if closed:
                outcome = GraspOutcome(True, "VLA grasp", tip_z, step + 1)
        finally:
            streamer.stop()
            try:
                self.robot.wait_idle()
            finally:
                if sampler:
                    sampler.finish(keep=outcome.ok)
        streamer.check()
        return outcome

    def human_grasp(self, spec) -> GraspOutcome:
        """A demonstration: the operator hand-guides the arm (drag mode) from
        the hover pose onto the object and back up, closing the gripper from
        the keyboard, while the recorder samples cameras and joints."""
        ask = self.operator
        ask("Drag mode next: guide the gripper onto the object and back up. Press Enter to start.")
        self.robot.start_drag()
        sampler = self._start_recording(spec)
        tip_z = None
        try:
            while True:
                key = ask("[c]+Enter close, [o]+Enter open, Enter alone: done > ").strip().lower()
                if key == "c":
                    tip_z = float(tool_pose(self.robot.joints())[2, 3])
                    grip = self.gripper.close()
                    self.log("holding" if grip.holding else "closed on nothing")
                elif key == "o":
                    self.gripper.open()
                elif key == "":
                    break
        finally:
            self.robot.stop_drag()
            self.robot.wait_idle(what="leaving drag mode")
        holding = bool(self.gripper.last and self.gripper.last.holding)
        keep = holding and ask("Keep this demo? [Y/n] ").strip().lower() != "n"
        if sampler:
            sampler.finish(keep)
        return GraspOutcome(holding, "hand-guided grasp" if holding else "not holding after the demo", tip_z)

    def retreat(self, hover):
        """After a failed grasp: open, straight up to the hover height, home."""
        self.gripper.open()
        try:
            self.straight_up(float(hover.T_base_tool[2, 3]), "back up")
        except IKError as e:
            self.log(f"cannot back up in a straight line ({e}): joint move to the hover pose")
            self.move_j(hover.q_servo_deg, "hover")
        self.go_home()

    def plan_place(self, loc, h_tip, q_lift, carry):
        p = self.cfg.pick
        tip_z = loc.z + h_tip + p.place_clearance_m if loc.z_reference == "surface" else loc.z
        agent = self._place_agent()
        errors = []
        for yaw in self._place_yaws(loc):
            T_place = tool_down(loc.x, loc.y, tip_z, math.radians(yaw))
            try:
                line = plan_place(agent, self.table, T_place, q_lift)
            except IKError as e:
                errors.append(f"w {yaw:.0f}: {e}")
                continue
            bad = carry.path_violations(joint_map.servo_to_visual(q_lift), joint_map.servo_to_visual(line[0][2]))
            if bad:
                errors.append(f"w {yaw:.0f}: the carried object would hit the table: " + ", ".join(bad))
                continue
            if yaw != loc.w_deg:
                self.log(f"placing with w {yaw:.0f} deg (w {loc.w_deg:.0f} is out of reach)")
            return line
        raise IKError(" | ".join(errors))

    def _start_recording(self, spec):
        if self.recorder is None:
            return None
        return self.recorder.start(spec.instruction(self.cfg.vla.task_template))

    def _recover(self):
        """After a failed task: gripper open unless holding, arm home."""
        try:
            if not (self.gripper.last and self.gripper.last.holding):
                self.gripper.open()
            q = self.robot.get_angle()
            if np.abs(q - self.home).max() > 0.3:
                z = float(tool_pose(q)[2, 3])
                try:
                    self.straight_up(z + 0.05, "back up")
                except IKError:
                    pass
                self.go_home()
        except IKError as e:
            self.log(f"could not return home safely ({e}): jog the arm home")

    # ================================================================ records
    def _run_dir(self, obj, loc):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        d = self.cfg.path(self.cfg.paths.runs_dir) / f"{stamp}_{obj}_to_{loc}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _save(run_dir: Path, result: TaskResult):
        (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2, default=str))
