"""
FakeNova2Controller: the twin's fake controller (fake_dobot.py: dashboard
commands, queue IDs, RobotMode, MovJ / MovL in a background thread) plus
ServoJ, StartDrag / StopDrag and a real-time feedback port that pushes
1440-byte packets every 8 ms -- enough for dry runs and tests of the whole
VLA loop with no robot.

ServoJ moves the joints straight to the set-point, but no faster than
servo_deg_per_s (the real controller is also speed limited).
"""

import re
import socket
import threading
import time

import numpy as np

from .. import twin  # noqa: F401 -- puts the twin on sys.path
from fake_dobot import FakeDobot, euler_xyz_deg  # noqa: E402
from ..twin import fk, joint_map  # noqa: E402
from .feedback import build_packet  # noqa: E402

_KW = re.compile(r"(\w+)\s*=\s*([-+0-9.eE]+)")


class FakeNova2Controller(FakeDobot):
    def __init__(self, joints=None, servo_deg_per_s=180.0, **kw):
        super().__init__(joints=joints, **kw)
        self.servo_deg_per_s = servo_deg_per_s
        self.servo_count = 0
        self._fb_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._fb_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._fb_server.bind(("127.0.0.1", 0))
        self._fb_server.listen(4)
        self.feedback_port = self._fb_server.getsockname()[1]

    def start(self):
        super().start()
        threading.Thread(target=self._accept_feedback, daemon=True).start()
        return self

    def stop(self):
        super().stop()
        self._fb_server.close()

    # ------------------------------------------------------------ feedback
    def _accept_feedback(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._fb_server.accept()
            except OSError:
                return
            threading.Thread(target=self._push_feedback, args=(conn,), daemon=True).start()

    def _push_feedback(self, conn):
        with conn:
            while not self._stop.is_set():
                with self.lock:
                    q, mode = self.joints.copy(), self.mode
                T = fk(joint_map.servo_to_visual(q), "flange")
                tool = [*(T[:3, 3] * 1000), *euler_xyz_deg(T[:3, :3])]
                try:
                    conn.sendall(build_packet(q, mode, int(time.time() * 1000), tool))
                except OSError:
                    return
                time.sleep(0.008)

    # ------------------------------------------------------------ commands
    def _run(self, name, args):
        if name == "ServoJ":
            if self.mode in (9, 11):
                return -2, ""
            if self.mode not in (5, 7) or self.queue:
                return -1, ""
            parts = [p.strip() for p in args.split(",")]
            try:
                target = np.array([float(p) for p in parts[:6]])
            except ValueError:
                return -30001, ""
            kw = {k: float(v) for k, v in _KW.findall(args)}
            t = kw.get("t", 0.1)
            if target.shape != (6,) or not 0.004 <= t <= 3600:
                return -40001, ""
            step = np.clip(target - self.joints, -self.servo_deg_per_s * t, self.servo_deg_per_s * t)
            self.joints = self.joints + step
            self.servo_count += 1
            self.next_id += 1
            self.current_id = self.next_id
            return 0, str(self.next_id)
        if name == "StartDrag":
            if self.mode != 5:
                return -1, ""
            self.mode = 6
            return 0, ""
        if name == "StopDrag":
            if self.mode == 6:
                self.mode = 5
            return 0, ""
        return super()._run(name, args)

    def drag_to(self, joints):
        """Tests: move the arm as a hand would in drag mode."""
        with self.lock:
            if self.mode != 6:
                raise RuntimeError("not in drag mode")
            self.joints = np.asarray(joints, dtype=float).copy()
