"""
ServoStreamer: sends ServoJ every `period` seconds, walking its set-point
towards the latest goal at no more than `max_speed_deg_s` per joint. The
policy loop only sets goals (at its own, slower rate); this thread turns
them into a smooth, rate-limited stream and holds still when no new goal
comes. Any controller error stops the stream and is re-raised to the
caller on its next set_goal() / check().
"""

import threading
import time

import numpy as np

from ..twin import DobotError


class ServoStreamer:
    def __init__(self, robot, period=0.03, max_speed_deg_s=45.0, aheadtime=50.0, gain=500.0,
                 mode_check_every=10):
        self.robot = robot
        self.period = float(period)
        self.max_step = float(max_speed_deg_s) * self.period
        self.aheadtime, self.gain = aheadtime, gain
        self.mode_check_every = mode_check_every
        self._goal = None
        self._setpoint = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.error = None
        self.sent = 0

    # ------------------------------------------------------------ control
    def start(self, q_start):
        q = np.asarray(q_start, dtype=float).copy()
        with self._lock:
            self._goal, self._setpoint = q.copy(), q.copy()
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, name="nova2-servoj", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_goal(self, q):
        self.check()
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError(f"bad joint goal {q}")
        with self._lock:
            self._goal = q.copy()

    @property
    def goal(self):
        with self._lock:
            return self._goal.copy()

    @property
    def setpoint(self):
        with self._lock:
            return self._setpoint.copy()

    def check(self):
        if self.error is not None:
            raise self.error

    def settle(self, tol_deg=0.2, timeout=3.0):
        """Wait until the set-point has reached the goal and the arm has
        followed it within tol_deg."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self.check()
            goal = self.goal
            if np.abs(self.setpoint - goal).max() < 1e-6 and np.abs(self.robot.joints() - goal).max() <= tol_deg:
                return True
            time.sleep(self.period)
        return False

    # ------------------------------------------------------------ thread
    def _run(self):
        next_t = time.monotonic()
        n = 0
        while not self._stop.is_set():
            with self._lock:
                delta = np.clip(self._goal - self._setpoint, -self.max_step, self.max_step)
                self._setpoint = self._setpoint + delta
                target = self._setpoint.copy()
            try:
                self.robot.servo_j(target, t=self.period, aheadtime=self.aheadtime, gain=self.gain)
                self.sent += 1
                n += 1
                if n % self.mode_check_every == 0:
                    self.robot.check_ok("ServoJ streaming")
            except (DobotError, OSError) as e:
                self.error = e if isinstance(e, DobotError) else DobotError(f"ServoJ streaming: {e}")
                try:
                    self.robot.stop()
                except (DobotError, OSError):
                    pass
                return
            next_t += self.period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()   # fell behind: don't burst to catch up
