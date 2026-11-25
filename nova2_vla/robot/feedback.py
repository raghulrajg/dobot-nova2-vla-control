"""
Real-time feedback from the Nova 2 controller: port 30004 pushes a
1440-byte little-endian packet every 8 ms (TCP/IP guide V4, chapter 3).
Only the fields used here are decoded.
"""

import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

PACKET_SIZE = 1440
TEST_VALUE = 0x0123456789ABCDEF

_ROBOT_MODE = 24          # uint64
_TIMESTAMP = 32           # uint64, ms
_TEST_VALUE = 48          # uint64
_Q_TARGET = 192           # 6 x double, deg
_Q_ACTUAL = 432           # 6 x double, deg
_QD_ACTUAL = 480          # 6 x double, deg/s
_TOOL_VECTOR_ACTUAL = 624  # 6 x double, mm / deg


@dataclass
class Feedback:
    robot_mode: int
    stamp_ms: int
    q_actual: np.ndarray       # joint angles [deg]
    qd_actual: np.ndarray      # joint speeds [deg/s]
    q_target: np.ndarray
    tool_vector: np.ndarray    # x y z [mm], rx ry rz [deg] of the selected tool
    received: float            # time.monotonic() on arrival


def parse_packet(buf: bytes) -> Feedback:
    if len(buf) < PACKET_SIZE:
        raise ValueError(f"feedback packet has {len(buf)} bytes, expected {PACKET_SIZE}")
    (size,) = struct.unpack_from("<H", buf, 0)
    (test,) = struct.unpack_from("<Q", buf, _TEST_VALUE)
    if size != PACKET_SIZE or test != TEST_VALUE:
        raise ValueError(f"bad feedback packet (size {size}, test value {test:#x})")

    def six(offset):
        return np.array(struct.unpack_from("<6d", buf, offset))

    (mode,) = struct.unpack_from("<Q", buf, _ROBOT_MODE)
    (stamp,) = struct.unpack_from("<Q", buf, _TIMESTAMP)
    return Feedback(int(mode), int(stamp), six(_Q_ACTUAL), six(_QD_ACTUAL), six(_Q_TARGET),
                    six(_TOOL_VECTOR_ACTUAL), time.monotonic())


def build_packet(q_actual, robot_mode=5, stamp_ms=0, tool_vector=(0,) * 6, q_target=None):
    """A valid packet (for tests and the fake controller)."""
    buf = bytearray(PACKET_SIZE)
    struct.pack_into("<H", buf, 0, PACKET_SIZE)
    struct.pack_into("<Q", buf, _ROBOT_MODE, int(robot_mode))
    struct.pack_into("<Q", buf, _TIMESTAMP, int(stamp_ms))
    struct.pack_into("<Q", buf, _TEST_VALUE, TEST_VALUE)
    struct.pack_into("<6d", buf, _Q_TARGET, *(q_actual if q_target is None else q_target))
    struct.pack_into("<6d", buf, _Q_ACTUAL, *q_actual)
    struct.pack_into("<6d", buf, _TOOL_VECTOR_ACTUAL, *tool_vector)
    return bytes(buf)


class RealtimeFeedback:
    """Background reader of port 30004; .latest is the newest Feedback."""

    def __init__(self, ip, port=30004, timeout=2.0):
        self.ip, self.port, self.timeout = ip, port, timeout
        self.latest: Optional[Feedback] = None
        self.error: Optional[Exception] = None
        self._sock = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._sock = socket.create_connection((self.ip, self.port), self.timeout)
        self._sock.settimeout(self.timeout)
        self._thread = threading.Thread(target=self._run, name="nova2-feedback", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)

    def fresh(self, max_age_s=0.1) -> Optional[Feedback]:
        fb = self.latest
        if fb is None or time.monotonic() - fb.received > max_age_s:
            return None
        return fb

    def _run(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(65536)
            except (OSError, socket.timeout) as e:
                if not self._stop.is_set():
                    self.error = e
                return
            if not chunk:
                self.error = ConnectionError("feedback port closed by the controller")
                return
            buf += chunk
            # keep only whole packets; resynchronise on the size/test-value header
            while len(buf) >= PACKET_SIZE:
                try:
                    self.latest = parse_packet(buf[:PACKET_SIZE])
                    buf = buf[PACKET_SIZE:]
                except ValueError:
                    buf = buf[1:]
