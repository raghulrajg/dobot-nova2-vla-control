"""Gripper factory and a thin wrapper that also tracks the commanded state
(0 = open, 1 = closed) for the policy's observation and for recording."""

from ..twin import FakeGripper, FeetechGripper, GripResult

TORQUE_LIMIT_ADDR = 48   # Feetech SMS/STS torque limit register (as in the twin's gripper.py)


class Gripper:
    def __init__(self, device, closed_position=255):
        self.device = device
        self.closed_position = closed_position
        self.command = 0.0          # last command: 0 open, 1 closed
        self.last: GripResult = None

    def open(self, **kw):
        self.device.open(**kw)
        self.command = 0.0
        self.last = None

    def close(self, **kw) -> GripResult:
        self.command = 1.0
        self.last = self.device.close(**kw)
        return self.last

    @property
    def state(self) -> float:
        """Finger closure 0 (open) .. 1 (fully closed), from the last
        reading (holding an object reads somewhere in between)."""
        if self.last is None:
            return self.command
        return min(1.0, max(0.0, self.last.position / self.closed_position))

    def set_torque(self, limit: int):
        """Squeeze for the next grip (Feetech 0..1000), e.g. per product."""
        dev = self.device
        if getattr(dev, "servo", None) is None:
            return
        comm, err = dev.servo.write2ByteTxRx(dev.servo_id, TORQUE_LIMIT_ADDR, int(limit))
        dev._check(comm, err, "set the torque limit")
        dev.torque_limit = int(limit)

    def disconnect(self):
        self.device.disconnect()


def make_gripper(cfg) -> Gripper:
    if cfg.type == "fake":
        return Gripper(FakeGripper())
    return Gripper(FeetechGripper(cfg.port, cfg.servo_id, torque_limit=cfg.torque_limit))
