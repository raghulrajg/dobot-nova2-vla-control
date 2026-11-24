"""
Cameras. Each runs a grabber thread and keeps only its newest frame, so the
control loop never reads a stale, buffered image.

  realsense  colour + depth aligned to colour (pyrealsense2), intrinsics from the stream
  opencv     any UVC camera (colour only), e.g. a fixed scene camera
  sim        rendered by nova2_vla.sim for dry runs

Frame.rgb is HxWx3 uint8 RGB; Frame.depth is HxW float32 metres (0 = no
reading), aligned to rgb; Frame.K is the 3x3 colour intrinsics.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..geometry import camera_pose
from ..twin import pose, rot_z


class CameraError(RuntimeError):
    pass


@dataclass
class Frame:
    rgb: np.ndarray
    depth: Optional[np.ndarray]
    K: Optional[np.ndarray]
    stamp: float


class ThreadedCamera:
    def __init__(self, name, cfg):
        self.name, self.cfg = name, cfg
        self._latest: Optional[Frame] = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self.error = None

    def start(self):
        self._open()
        self._thread = threading.Thread(target=self._loop, name=f"cam-{self.name}", daemon=True)
        self._thread.start()
        self.read(timeout=5.0)
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                frame = self._grab()
            except Exception as e:   # surfaced on the next read()
                self.error = e
                with self._cond:
                    self._cond.notify_all()
                return
            with self._cond:
                self._latest = frame
                self._cond.notify_all()

    def read(self, newer_than: float = None, timeout=2.0) -> Frame:
        """Newest frame; with newer_than, wait for one taken after that
        time.monotonic() stamp (e.g. after the arm stopped)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                if self.error:
                    raise CameraError(f"camera {self.name}: {self.error}")
                f = self._latest
                if f is not None and (newer_than is None or f.stamp > newer_than):
                    return f
                left = deadline - time.monotonic()
                if left <= 0:
                    raise CameraError(f"camera {self.name}: no frame for {timeout:.1f} s")
                self._cond.wait(left)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._close()

    def _open(self):
        raise NotImplementedError

    def _grab(self) -> Frame:
        raise NotImplementedError

    def _close(self):
        pass


class RealSenseCamera(ThreadedCamera):
    def _open(self):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise CameraError("pyrealsense2 is not installed: pip install pyrealsense2") from e
        self.rs = rs
        c = self.cfg
        self.pipeline = rs.pipeline()
        config = rs.config()
        if c.serial:
            config.enable_device(str(c.serial))
        config.enable_stream(rs.stream.color, c.width, c.height, rs.format.rgb8, c.fps)
        if c.depth:
            config.enable_stream(rs.stream.depth, c.width, c.height, rs.format.z16, c.fps)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color) if c.depth else None
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale() if c.depth else None
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
        for _ in range(15):   # let auto-exposure settle
            self.pipeline.wait_for_frames(5000)

    def _grab(self):
        frames = self.pipeline.wait_for_frames(2000)
        if self.align is not None:
            frames = self.align.process(frames)
        color = frames.get_color_frame()
        rgb = np.asanyarray(color.get_data()).copy()
        depth = None
        if self.align is not None:
            d = frames.get_depth_frame()
            depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
        return Frame(rgb, depth, self.K.copy(), time.monotonic())

    def _close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class OpenCVCamera(ThreadedCamera):
    def _open(self):
        import cv2
        self.cv2 = cv2
        c = self.cfg
        self.cap = cv2.VideoCapture(c.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        self.cap.set(cv2.CAP_PROP_FPS, c.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise CameraError(f"cannot open video device {c.index}")

    def _grab(self):
        ok, bgr = self.cap.read()
        if not ok:
            raise CameraError("read failed")
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (self.cfg.height, self.cfg.width):
            rgb = self.cv2.resize(rgb, (self.cfg.width, self.cfg.height))
        return Frame(rgb, None, None, time.monotonic())

    def _close(self):
        self.cap.release()


def intrinsics_from_fov(width, height, hfov_deg=69.0):
    f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.0]])


class CameraRig:
    """All cameras of the cell, by name."""

    def __init__(self, cameras: Dict[str, object], configs):
        self.cameras = cameras
        self.configs = configs

    @classmethod
    def from_config(cls, cfg, sim=None):
        cams = {}
        for name, c in cfg.cameras.items():
            if c.type == "realsense":
                cams[name] = RealSenseCamera(name, c)
            elif c.type == "opencv":
                cams[name] = OpenCVCamera(name, c)
            else:
                if sim is None:
                    raise CameraError(f"camera {name} is type sim: only for --dry-run")
                cams[name] = sim.camera(name, c)
        return cls(cams, cfg.cameras)

    def start(self):
        for cam in self.cameras.values():
            cam.start()
        return self

    def stop(self):
        for cam in self.cameras.values():
            cam.stop()

    def read(self, name, newer_than=None) -> Frame:
        return self.cameras[name].read(newer_than)

    def read_all(self, names=None, newer_than=None) -> Dict[str, Frame]:
        return {n: self.read(n, newer_than) for n in (names or list(self.cameras))}

    def shapes(self, names=None):
        return {n: (self.configs[n].height, self.configs[n].width, 3) for n in (names or list(self.cameras))}

    def T_base_camera(self, name, q_servo_deg):
        """Pose of a wrist camera's optical frame at these joints."""
        c = self.configs[name]
        if c.mount != "wrist":
            raise CameraError(f"camera {name} is not on the wrist")
        T = camera_pose(q_servo_deg)
        if c.optical_axis_rotation_deg:
            T = T @ pose(rot_z(math.radians(c.optical_axis_rotation_deg)), [0, 0, 0])
        return T
