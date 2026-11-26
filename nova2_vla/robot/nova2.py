"""
Nova 2 client for closed-loop control: the twin's DobotTCP (dashboard port
29999: enable, MovJ / MovL with reach checks, GetAngle, GetPose, alarms)
plus what a learned policy needs:

    ServoJ(J1..J6, t, aheadtime, gain)   joint set-point streaming, ~33 Hz
    StartDrag() / StopDrag()             hand-guiding, for recording demos
    port 30004                           joints every 8 ms (RealtimeFeedback)
"""

import time

import numpy as np

from ..twin import (
    MODE_COLLISION, MODE_DISABLED, MODE_ERROR, MODE_IDLE, MODE_NAMES, MODE_POWER_OFF,
    SERVO_LIMITS_DEG, DobotError, DobotTCP,
)
from .feedback import RealtimeFeedback

MODE_DRAG = 6

# GetPose forms, newest firmware first: V4.5+ takes user/tool as keywords, some
# earlier V4 builds only positionally, and the bare form reports the pose in the
# controller's global user and tool frames (selected with User(i) / Tool(i)).
POSE_COMMANDS = ("GetPose(user=0,tool=0)", "GetPose(0,0)", "GetPose()")


def _is_format_error(e):
    """ErrorIDs that mean "this firmware wants other parameters"
    (unknown command, parameter count, type or range)."""
    eid = e.error_id
    return eid is not None and (eid in (-10000, -20000) or -60010 < eid <= -30000)


def check_joints(servo_deg):
    q = np.asarray(servo_deg, dtype=float).reshape(6)
    lo, hi = SERVO_LIMITS_DEG[:, 0], SERVO_LIMITS_DEG[:, 1]
    bad = np.where((q < lo) | (q > hi) | ~np.isfinite(q))[0]
    if bad.size:
        raise DobotError("target outside the joint limits: " + ", ".join(
            f"J{i + 1}={q[i]:.2f} (limits {lo[i]:.0f}..{hi[i]:.0f})" for i in bad))
    return q


class Nova2(DobotTCP):
    def __init__(self, ip="192.168.5.1", port=29999, feedback_port=30004, realtime_feedback=False, **kw):
        super().__init__(ip, port, **kw)
        self.feedback_port = feedback_port
        self.use_feedback = realtime_feedback
        self.feedback = None
        self._pose_cmd = None        # the GetPose form this controller accepts, once known
        self._frames_selected = False
        self.pose_frame = "tool0"    # "selected" if only the bare GetPose() works

    def connect(self):
        """Dashboard port 29999 -- and, only if realtime_feedback is on,
        port 30004 as a second connection."""
        try:
            super().connect()
            self.robot_mode()       # the controller answers on this connection
        except (OSError, DobotError) as e:
            super().close()
            raise DobotError(f"no answer from the controller at {self.ip}:{self.port} ({e}). It takes one TCP "
                             "connection at a time: close every other program connected to it (manipulator-agent "
                             "scripts, DobotStudio, another nova2_vla) and try again") from None
        if self.use_feedback and self.feedback is None:
            try:
                self.feedback = RealtimeFeedback(self.ip, self.feedback_port).start()
            except OSError as e:
                print(f"note: no real-time feedback on port {self.feedback_port} ({e}); polling GetAngle")
                self.feedback = None
        return self

    def close(self):
        if self.feedback:
            self.feedback.stop()
            self.feedback = None
        super().close()

    # ------------------------------------------------------------ state
    def joints(self):
        """Current joint angles [deg]: from port 30004 when it is fresh,
        else GetAngle."""
        fb = self.feedback.fresh(0.05) if self.feedback else None
        return fb.q_actual.copy() if fb is not None else self.get_angle()

    def get_pose(self):
        """Flange pose (user 0, tool 0): x y z [mm], rx ry rz [deg]. Tries
        the GetPose forms of the different V4 firmwares and keeps the one
        this controller accepts."""
        if self._pose_cmd:
            return self._numbers(self._pose_cmd, 6)
        errors = []
        for cmd in POSE_COMMANDS:
            try:
                pose = self._numbers(cmd, 6)
            except DobotError as e:
                if not _is_format_error(e):
                    raise
                errors.append(f"{cmd}: ErrorID {e.error_id}")
                continue
            self._pose_cmd = cmd
            self.pose_frame = "selected" if cmd == "GetPose()" else "tool0"
            if cmd == "GetPose()":
                self._select_base_frames()
                pose = self._numbers(cmd, 6)
            return pose
        raise DobotError("the controller accepts no GetPose form (" + "; ".join(errors) + ")")

    def _select_base_frames(self):
        """The bare GetPose() reports in the global user / tool frames: make
        them the base frame and the bare flange (both are also the
        controller's defaults). Motion here uses joint targets, so this
        changes no path of ours."""
        ok = True
        for cmd in ("User(0)", "Tool(0)"):
            try:
                self.send(cmd)
            except DobotError:
                ok = False
        self._frames_selected = ok

    @property
    def pose_hint(self):
        """Added to pose-mismatch errors when the controller can only report
        the pose in its global frames."""
        if self.pose_frame != "selected" or self._frames_selected:
            return ""
        return (" -- this controller reports GetPose() in its global user and tool frames, and User(0) / "
                "Tool(0) were refused (robot disabled?): the reading may be of a tool, not the flange")

    def mode_fast(self):
        fb = self.feedback.fresh(0.05) if self.feedback else None
        return fb.robot_mode if fb is not None else self.robot_mode()

    def check_ok(self, what="robot"):
        mode = self.mode_fast()
        if mode in (MODE_ERROR, MODE_COLLISION):
            raise DobotError(f"{what}: robot {MODE_NAMES[mode]} (alarms {self.error_ids()})", mode)
        if mode in (MODE_POWER_OFF, MODE_DISABLED):
            raise DobotError(f"{what}: robot {MODE_NAMES[mode]}", mode)
        return mode

    def wait_idle(self, timeout=5.0, what="settle"):
        """Wait until the motion queue is empty (RobotMode 5)."""
        t0 = time.monotonic()
        while True:
            mode = self.robot_mode()
            if mode == MODE_IDLE:
                return
            if mode in (MODE_ERROR, MODE_COLLISION, MODE_POWER_OFF, MODE_DISABLED):
                self.check_ok(what)
            if time.monotonic() - t0 > timeout:
                raise DobotError(f"{what}: robot still {MODE_NAMES.get(mode, mode)} after {timeout:.0f} s")
            time.sleep(0.02)

    # ------------------------------------------------------------ streaming
    def servo_j(self, servo_deg, t=0.03, aheadtime=50.0, gain=500.0):
        """One ServoJ set-point; the controller reaches it in t seconds.
        Call it every t seconds with points that are already rate-limited
        (ServoStreamer does)."""
        q = check_joints(servo_deg)
        if not 0.004 <= t <= 3600.0:
            raise DobotError(f"ServoJ t={t} outside 0.004..3600 s")
        args = ",".join(f"{a:.4f}" for a in q)
        return self.send(f"ServoJ({args},t={t:.3f},aheadtime={aheadtime:.1f},gain={gain:.1f})")

    # ------------------------------------------------------------ hand guiding
    def start_drag(self):
        self.send("StartDrag()")

    def stop_drag(self):
        self.send("StopDrag()")
