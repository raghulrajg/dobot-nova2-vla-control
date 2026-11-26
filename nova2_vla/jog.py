"""
Moving the arm without a teach pendant, over the cell's single TCP
connection: key presses move the gripper tip in small steps through the
digital twin's IK (each step a MovJ to the IK joints -- the command your
nova2_control.py uses), or drag mode lets you guide the arm by hand.
Used by `jog`, `measure-table` and `teach-location`.
"""

import math
import sys

import numpy as np

from .geometry import tilt_deg, tool_pose, yaw_of
from .twin import IKError, joint_map, pose, rot_z, solve_ik

HELP = ("  keys: w/s +X/-X   a/d +Y/-Y   r/f up/down   q/e turn gripper   1-4 step size\n"
        "        o/c open/close gripper   g drag mode on/off (guide by hand)   h home\n"
        "        Enter = take this position   x = stop")
STEPS = {"1": (0.001, 1.0), "2": (0.005, 5.0), "3": (0.010, 15.0), "4": (0.050, 45.0)}   # (m, deg)
MOVES = {"w": (1, 0, 0, 0), "s": (-1, 0, 0, 0), "a": (0, 1, 0, 0), "d": (0, -1, 0, 0),
         "r": (0, 0, 1, 0), "f": (0, 0, -1, 0), "q": (0, 0, 0, 1), "e": (0, 0, 0, -1)}
MAX_JOINT_SWING_DEG = 20.0    # a step needing more than this would flip an IK branch


class JogAborted(Exception):
    pass


class KeyReader:
    """Single key presses from the terminal (no Enter needed); falls back to
    reading lines when stdin is not a terminal (an empty line = Enter)."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdin
        self._old = None
        self._buffer = []

    def __enter__(self):
        if self.stream.isatty():
            import termios
            import tty
            self._old = termios.tcgetattr(self.stream.fileno())
            tty.setcbreak(self.stream.fileno())       # Ctrl+C still works
        return self

    def __exit__(self, *exc):
        if self._old is not None:
            import termios
            termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN, self._old)

    def get(self):
        if self._old is not None:
            return self.stream.read(1)
        if not self._buffer:
            line = self.stream.readline()
            if not line:
                return "x"
            self._buffer = list(line.rstrip("\n")) or ["\n"]
        return self._buffer.pop(0)


class Jogger:
    def __init__(self, robot, keys, gripper=None, table=None, home=None, speed=10, log=print):
        """table: a TableConstraint every step is checked against (None =
        unchecked: go down in 1 mm steps near the table)."""
        self.robot, self.keys, self.gripper, self.table = robot, keys, gripper, table
        self.home = np.asarray(joint_map.home_servo_deg if home is None else home, float)
        self.speed, self.log = speed, log
        self.step = STEPS["2"]
        self.dragging = False

    def tip(self):
        q = self.robot.get_angle()
        return q, tool_pose(q)

    def show(self):
        _, T = self.tip()
        x, y, z = T[:3, 3] * 1000
        lin, ang = self.step
        self.log(f"  tip x {x:7.1f}  y {y:7.1f}  z {z:7.1f} mm   turn {math.degrees(yaw_of(T)):6.1f} deg   "
                 f"tilt {tilt_deg(T):4.1f} deg   step {lin * 1000:.0f} mm / {ang:.0f} deg"
                 + ("   [drag mode]" if self.dragging else ""))

    def position(self, prompt, show_help=True):
        """Let the operator move the arm until they press Enter."""
        if prompt:
            self.log(prompt)
        if show_help:
            self.log(HELP)
        self.show()
        while True:
            k = self.keys.get()
            if k in ("\n", "\r"):
                return
            if k in ("x", "\x1b"):
                raise JogAborted()
            if k in STEPS:
                self.step = STEPS[k]
                self.show()
            elif k in MOVES:
                if self.dragging:
                    self.log("  in drag mode: press g to lock the arm before stepping")
                else:
                    self.move(*MOVES[k])
            elif k == "g":
                self.drag(not self.dragging)
                self.show()
            elif k in ("o", "c"):
                self.grip(k == "c")
            elif k == "h":
                self.go_home()

    def move(self, dx, dy, dz, dyaw, lin=None, ang=None):
        """One step of the tool tip in the base frame (and a turn about the
        vertical through the tip), as a joint move to the IK solution."""
        q, T = self.tip()
        lin = self.step[0] if lin is None else lin
        ang = self.step[1] if ang is None else ang
        T_new = pose(rot_z(math.radians(dyaw * ang)) @ T[:3, :3], T[:3, 3] + lin * np.array([dx, dy, dz]))
        try:
            q_new = solve_ik(T_new, q, self.table)
        except IKError as e:
            self.log(f"  not there: {e}")
            return False
        swing = float(np.abs(q_new - q).max())
        if swing > MAX_JOINT_SWING_DEG:
            self.log(f"  refused: this step would swing a joint {swing:.0f} deg (the arm would flip); "
                     "take another way or a smaller step")
            return False
        if self.table is not None:
            bad = self.table.path_violations(joint_map.servo_to_visual(q), joint_map.servo_to_visual(q_new))
            if bad:
                self.log("  refused: " + ", ".join(bad))
                return False
        self.robot.move_j(q_new, v=self.speed, a=self.speed, what="jog")
        self.show()
        return True

    def up(self, metres):
        """Straight up (e.g. off the table after a reading)."""
        return self.move(0, 0, 1, 0, lin=metres, ang=0.0)

    def drag(self, on):
        if on:
            self.robot.start_drag()
            self.log("  drag mode: guide the arm by hand; g locks it again")
        else:
            self.robot.stop_drag()
            self.robot.wait_idle(what="leaving drag mode")
        self.dragging = on

    def grip(self, close):
        if self.gripper is None:
            self.log("  no gripper connected")
            return
        if close:
            r = self.gripper.close()
            self.log(f"  gripper closed ({'holding something' if r.holding else 'empty'})")
        else:
            self.gripper.open()
            self.log("  gripper open")

    def go_home(self):
        q = self.robot.get_angle()
        if self.table is not None:
            bad = self.table.path_violations(joint_map.servo_to_visual(q), joint_map.servo_to_visual(self.home))
            if bad:
                self.log("  refused, the way home goes below the table: " + ", ".join(bad) + " -- step up first")
                return
        self.robot.move_j(self.home, v=self.speed, a=self.speed, what="home")
        self.show()

    def close(self):
        if self.dragging:
            self.drag(False)
