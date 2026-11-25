"""
Command line.

  python -m nova2_vla run "pick the object 'gear_housing' place the 'bin_A'"
  python -m nova2_vla shell                       # type commands one after another
  python -m nova2_vla list                        # known objects and locations
  python -m nova2_vla check [--robot]             # GPU / packages / config / reach (/ robot, read only)
  python -m nova2_vla capture-refs gear_housing   # reference photos with the wrist camera
  python -m nova2_vla collect --object gear_housing --episodes 50            # self-resetting scripted demos
  python -m nova2_vla collect --object gear_housing --episodes 20 --mode human   # hand-guided demos
  python -m nova2_vla train                       # fine-tune SmolVLA on them (lerobot-train)

Add --dry-run to any of them to use the simulated cell (fake controller on
localhost, rendered wrist camera, simulated gripper and a stand-in policy).
"""

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .config import ConfigError, load_config
from .task import LocationLibrary, ObjectLibrary, TaskError

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "cell.yaml"


class Cell:
    """Everything a session holds open; close() releases it all."""

    def __init__(self):
        self.closers = []
        self.cell = None
        self.robot = None

    def close(self):
        for fn in reversed(self.closers):
            try:
                fn()
            except Exception as e:   # keep closing the rest
                print(f"warning: while shutting down: {e}")


def build(cfg, args, log=print) -> Cell:
    from .perception.cameras import CameraRig
    from .perception.grounding import ColorHistEmbedder, DepthProposer, ReferenceObjectFinder, make_finder
    from .pipeline import PickPlaceCell
    from .robot.gripper import Gripper, make_gripper
    from .robot.nova2 import Nova2

    s = Cell()
    policy = None
    objects_dir = cfg.path(cfg.paths.objects_dir)
    strategy = getattr(args, "strategy", None) or cfg.pick.strategy
    policy_path = getattr(args, "policy", None) or cfg.vla.policy_path

    if args.dry_run:
        from .robot.fake import FakeNova2Controller
        from .sim import DEMO_BOXES, MockGraspPolicy, SimCamera, SimGripper, SimWorld, write_demo_objects
        if cfg.robot.table_z_m is None:
            cfg.robot.table_z_m = -0.05
        ctrl = FakeNova2Controller(deg_per_s=600.0, servo_deg_per_s=600.0).start()
        s.closers.append(ctrl.stop)
        s.ctrl = ctrl
        robot = Nova2("127.0.0.1", ctrl.port, feedback_port=ctrl.feedback_port, poll_s=0.01)
        world = SimWorld(cfg.robot.table_z_m, copy.deepcopy(DEMO_BOXES), joints=lambda: ctrl.joints.copy())
        gripper = Gripper(SimGripper(world, cfg.gripper.max_opening_m))
        for c in cfg.cameras.values():
            c.type = "sim"
        rig = CameraRig({n: SimCamera(n, c, world) for n, c in cfg.cameras.items()}, cfg.cameras)
        finder = ReferenceObjectFinder([DepthProposer(cfg.perception.min_object_height_m, cfg.perception.min_area_px)],
                                       ColorHistEmbedder(), min_similarity=0.5)
        objects_dir = cfg.path(cfg.paths.runs_dir) / "dry_run_objects"
        write_demo_objects(objects_dir, world)
        if strategy != "scripted":
            policy = MockGraspPolicy(world) if not policy_path else None
        s.world = world
        log(f"dry run: simulated cell, objects {[b.name for b in world.boxes]} on the table")
    else:
        if cfg.robot.table_z_m is None:
            raise ConfigError("robot.table_z_m must be set in the config (the measured table height, metres)")
        robot = Nova2(cfg.robot.ip, realtime_feedback=cfg.robot.realtime_feedback)
        gripper = make_gripper(cfg.gripper)
        s.closers.append(gripper.disconnect)
        rig = CameraRig.from_config(cfg)
        finder = make_finder(cfg.perception)

    if policy is None and policy_path and strategy != "scripted":
        from .policy import load_policy
        log(f"loading VLA policy {policy_path} ...")
        policy = load_policy(policy_path, cfg.vla.device, cfg.vla.cameras)

    sampler = None
    if getattr(args, "record", False):
        from .data.recorder import EpisodeRecorder, Sampler
        cams = cfg.vla.cameras or list(cfg.cameras)
        root = cfg.path(cfg.paths.dataset_root)
        recorder = EpisodeRecorder(root, cfg.paths.dataset_repo_id, cfg.vla.fps, rig.shapes(cams))
        s.closers.append(recorder.close)
        sampler = Sampler(recorder, rig, robot, gripper, log)
        log(f"recording episodes to {root}")

    rig.start()
    s.closers.append(rig.stop)
    robot.connect()
    s.closers.append(robot.close)
    s.robot = robot

    jitter = (getattr(args, "jitter_cm", 0.0) / 100.0, getattr(args, "jitter_deg", 0.0))
    s.cell = PickPlaceCell(cfg, robot, gripper, rig, finder, ObjectLibrary(objects_dir),
                           LocationLibrary.load(cfg.path(cfg.paths.locations_file), cfg.robot.table_z_m), policy,
                           recorder=sampler, record_vla=getattr(args, "record", False),
                           hover_jitter=jitter, log=log)
    s.cell.startup()
    return s


def _session(cfg, args, body):
    """Open the cell, run body(cell), stop the robot on anything unexpected."""
    s = None
    try:
        s = build(cfg, args)
        return body(s.cell)
    except KeyboardInterrupt:
        print("\nstopping the robot")
        _stop(s)
        return 130
    except Exception as e:   # robot fault, camera, gripper, policy, anything: stop first
        print(f"error: {type(e).__name__}: {e}")
        if s and s.robot:
            print("stopping the robot")
        _stop(s)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        if s:
            s.close()


def _stop(s):
    if s and s.robot:
        try:
            s.robot.stop()
        except Exception:
            pass


def _print_result(r):
    print(f"\n=> {r.state.upper()}: {r.detail}  ({r.duration_s:.1f} s, log in {r.run_dir})")
    return 0 if r.ok else 2


# ---------------------------------------------------------------- commands

def cmd_run(cfg, args):
    return _session(cfg, args, lambda cell: _print_result(cell.run(args.text)))


def cmd_shell(cfg, args):
    def loop(cell):
        print("type: pick the object 'NAME' place the 'LOCATION'   |  list  |  home  |  quit")
        while True:
            try:
                text = input("nova2> ").strip()
            except EOFError:
                return 0
            if not text:
                continue
            if text in ("quit", "exit", "q"):
                return 0
            if text == "list":
                print("objects:  ", ", ".join(cell.objects.names()) or "none")
                print("locations:", ", ".join(cell.locations.names()))
                continue
            if text == "home":
                cell.go_home()
                continue
            _print_result(cell.run(text))
    return _session(cfg, args, loop)


def cmd_list(cfg, args):
    try:
        objects = ObjectLibrary(cfg.path(cfg.paths.objects_dir))
        for name in objects.names():
            spec = objects.get(name)
            print(f"object   {name:<24} {len(spec.images)} reference image(s)  \"{spec.description}\"")
    except TaskError as e:
        print(f"objects: {e}")
    locs = LocationLibrary.load(cfg.path(cfg.paths.locations_file), cfg.robot.table_z_m)
    for name in locs.names():
        l = locs.get(name)
        print(f"location {name:<24} x {l.x * 1000:7.1f}  y {l.y * 1000:7.1f}  z {l.z * 1000:7.1f} mm  "
              f"w {l.w_deg:6.1f} deg  (z = {l.z_reference})")
    return 0


def cmd_check(cfg, args):
    ok = True
    print(f"python      {sys.version.split()[0]}")
    try:
        import torch
        print(f"torch       {torch.__version__}, CUDA build {torch.version.cuda}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            arch = f"sm_{cap[0]}{cap[1]}"
            mem = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"GPU         {name}, {arch}, {mem:.1f} GB")
            if arch not in torch.cuda.get_arch_list():
                ok = False
                print(f"  !! this torch build has no {arch} kernels (RTX 50xx needs a CUDA 12.8+ build): "
                      "pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128")
            else:
                x = torch.randn(256, 256, device="cuda")
                torch.cuda.synchronize()
                print(f"  CUDA test ok ({float((x @ x).sum()):.1f})")
        else:
            print("GPU         not visible to torch (driver? CUDA build?)")
            ok = False
    except ImportError:
        print("torch       not installed (pip install -e '.[vla]')")
        ok = False
    for mod in ("lerobot", "transformers", "pyrealsense2", "cv2"):
        try:
            m = __import__(mod)
            print(f"{mod:<11} {getattr(m, '__version__', 'installed')}")
        except ImportError:
            needed = {"lerobot": "for the VLA", "transformers": "for OWLv2 / DINOv2",
                      "pyrealsense2": "for RealSense cameras", "cv2": "required"}[mod]
            print(f"{mod:<11} not installed ({needed})")
            ok = ok and mod != "cv2"

    print(f"\nconfig      table z {cfg.robot.table_z_m} m, strategy {cfg.pick.strategy}, "
          f"policy {cfg.vla.policy_path or '(none: scripted grasps only)'}")
    if cfg.robot.table_z_m is not None:
        from .pipeline import PickPlaceCell
        cell = PickPlaceCell(cfg, None, None, None, None, None, None, log=lambda m: None)   # planning only
        locs = LocationLibrary.load(cfg.path(cfg.paths.locations_file), cfg.robot.table_z_m)
        for name in locs.names():
            try:
                cell.precheck_location(locs.get(name))
                print(f"location    {name:<20} reachable")
            except TaskError as e:
                ok = False
                print(f"location    {name:<20} !! {e}")
    if args.robot:
        from .robot.nova2 import Nova2
        from .twin import MODE_NAMES, fk, joint_map
        with Nova2(cfg.robot.ip, realtime_feedback=cfg.robot.realtime_feedback) as robot:
            mode = robot.robot_mode()
            q = robot.get_angle()
            p = robot.get_pose()
            model = fk(joint_map.servo_to_visual(q), "flange")[:3, 3] * 1000
            err = float(np.linalg.norm(model - p[:3]))
            print(f"\nrobot       {cfg.robot.ip}: {MODE_NAMES.get(mode, mode)}, joints {np.round(q, 2).tolist()}")
            print(f"model       twin FK vs controller pose: {err:.2f} mm"
                  + ("" if err <= cfg.robot.model_tol_mm else "  !! see the twin README's DH correction"))
            print(f"feedback    port 30004 {'ok' if robot.feedback and robot.feedback.fresh(0.5) else 'not available'}")
            ok = ok and err <= cfg.robot.model_tol_mm
    print("\nall good" if ok else "\nsome checks failed (see !!)")
    return 0 if ok else 1


def cmd_capture_refs(cfg, args):
    import cv2
    from .perception.grounding import SceneContext, crop

    def capture(cell):
        folder = cell.objects.root / args.name
        folder.mkdir(parents=True, exist_ok=True)
        meta = folder / "object.json"
        if not meta.exists():
            desc = args.description or input(f"short description of {args.name} (for the VLA instruction): ").strip()
            meta.write_text(json.dumps({"description": desc or args.name.replace("_", " ")}, indent=2) + "\n")
        for i in range(args.views):
            if not args.dry_run:
                input(f"[{i + 1}/{args.views}] put {args.name} alone under the camera "
                      "(turn it between shots), then press Enter")
            cell.go_home()
            q = cell.robot.get_angle()
            frame = cell.rig.read(cell.wrist, newer_than=time.monotonic())
            T = cell.rig.T_base_camera(cell.wrist, q)
            cands = cell.finder.propose(frame, SceneContext(frame.K, T, cell.table_z))
            if not cands:
                print("  nothing found on the table: is table_z right? is the object in view?")
                continue
            h, w = frame.rgb.shape[:2]
            c = min(cands, key=lambda c: np.hypot((c.box[0] + c.box[2]) / 2 - w / 2, (c.box[1] + c.box[3]) / 2 - h / 2))
            path = folder / f"ref_{time.strftime('%Y%m%d-%H%M%S')}_{i}.png"
            cv2.imwrite(str(path), cv2.cvtColor(crop(frame.rgb, c.box, 0.1), cv2.COLOR_RGB2BGR))
            print(f"  saved {path}")
        return 0
    return _session(cfg, args, capture)


def cmd_collect(cfg, args):
    from .collect import collect
    args.record = True
    args.strategy = "scripted"      # no policy needed to collect demos

    def body(cell):
        spec = cell.objects.get(args.object)
        area = tuple(args.area) if args.area else None
        collect(cell, spec, args.episodes, args.mode, area)
        return 0
    return _session(cfg, args, body)


def cmd_train(cfg, args):
    from .train import run
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    return run(cfg, args.base, args.steps, args.batch, args.output_dir, args.save_freq, extra, args.print_only)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m nova2_vla",
                                description="VLA pick-and-place for the Dobot Nova 2")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="cell YAML (default config/cell.yaml)")
    p.add_argument("--dry-run", action="store_true", help="simulated cell: no robot, camera, gripper or GPU")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def task_opts(sp):
        sp.add_argument("--strategy", choices=("vla", "scripted", "vla_with_fallback"))
        sp.add_argument("--policy", help="LeRobot checkpoint (overrides vla.policy_path)")
        sp.add_argument("--record", action="store_true", help="save every successful grasp as a training episode")

    sp = sub.add_parser("run", help="one task")
    sp.add_argument("text", help="pick the object 'NAME' place the 'LOCATION'")
    task_opts(sp)
    sp = sub.add_parser("shell", help="type tasks one after another")
    task_opts(sp)
    sub.add_parser("list", help="known objects and locations")
    sp = sub.add_parser("check", help="GPU, packages, config, reach")
    sp.add_argument("--robot", action="store_true", help="also connect to the robot (read only)")
    sp = sub.add_parser("capture-refs", help="reference photos of a new product with the wrist camera")
    sp.add_argument("name")
    sp.add_argument("--views", type=int, default=4)
    sp.add_argument("--description")
    sp = sub.add_parser("collect", help="record grasp demos for fine-tuning")
    sp.add_argument("--object", required=True)
    sp.add_argument("--episodes", type=int, default=50)
    sp.add_argument("--mode", choices=("scripted", "human"), default="scripted")
    sp.add_argument("--area", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
                    help="where to put it down between episodes, base frame [m] (default: under the camera)")
    sp.add_argument("--jitter-cm", type=float, default=2.0, help="random hover offset [cm]")
    sp.add_argument("--jitter-deg", type=float, default=15.0, help="random hover yaw [deg]")

    sp = sub.add_parser("train", help="fine-tune the VLA on the recorded demos (runs lerobot-train)")
    sp.add_argument("--base", default="lerobot/smolvla_base", help="pretrained policy to start from")
    sp.add_argument("--steps", type=int, default=20000)
    sp.add_argument("--batch", type=int, default=32, help="lower it if the GPU runs out of memory")
    sp.add_argument("--save-freq", type=int, default=5000)
    sp.add_argument("--output-dir", default="outputs/train/nova2_smolvla")
    sp.add_argument("--print-only", action="store_true", help="show the lerobot-train command, don't run it")
    sp.add_argument("extra", nargs=argparse.REMAINDER, help="-- then more lerobot-train options")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    overrides = {}
    if getattr(args, "strategy", None):
        overrides["pick"] = {"strategy": args.strategy}
    try:
        cfg = load_config(args.config if Path(args.config).exists() else None, overrides)
    except (ConfigError, OSError) as e:
        sys.exit(f"config error: {e}")
    handlers = {"run": cmd_run, "shell": cmd_shell, "list": cmd_list, "check": cmd_check,
                "capture-refs": cmd_capture_refs, "collect": cmd_collect, "train": cmd_train}
    try:
        return handlers[args.cmd](cfg, args)
    except (TaskError, ConfigError) as e:
        print(f"error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
