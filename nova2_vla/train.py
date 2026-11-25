"""
Fine-tune a pretrained VLA on the demos recorded in this cell, through
LeRobot's own trainer (`lerobot-train`).

    python -m nova2_vla train                       # SmolVLA, 20k steps, batch 32
    python -m nova2_vla train --steps 30000 --batch 16
    python -m nova2_vla train --base lerobot/pi05_base --batch 4   # needs > 16 GB: a cloud GPU

What this adds over calling lerobot-train yourself: the dataset's camera
names (observation.images.wrist, ...) are mapped onto the camera names the
base model was pretrained with (e.g. smolvla_base's camera1, camera2, ...)
in the cell's camera order -- LeRobot refuses to fine-tune when they don't
overlap -- and the run lands where `vla.policy_path` expects it.
LeRobotVLA applies the same mapping at run time.
"""

import json
import shlex
import shutil
import subprocess
from pathlib import Path


def dataset_cameras(root):
    info = json.loads((Path(root) / "meta" / "info.json").read_text())
    return [k for k, v in info["features"].items() if v.get("dtype") in ("video", "image")]


def camera_rename_map(ours, expected):
    """Our image keys -> the base model's, in order, for the ones that
    don't already match. {} when no renaming is needed."""
    if not expected or set(ours) <= set(expected) or set(expected) <= set(ours):
        return {}
    free = [k for k in expected if k not in ours]
    rename = {}
    for key in ours:
        if key in expected:
            continue
        if not free:
            raise ValueError(f"the base model takes {len(expected)} cameras {expected}; "
                             f"the dataset has more: {ours}")
        rename[key] = free.pop(0)
    return rename


def base_visual_keys(base):
    import lerobot.policies.factory  # noqa: F401 -- registers the policy config classes
    from lerobot.configs import FeatureType, PreTrainedConfig
    cfg = PreTrainedConfig.from_pretrained(base)
    return [k for k, f in (cfg.input_features or {}).items() if f.type is FeatureType.VISUAL]


def build_command(cfg, base, steps, batch, output_dir, save_freq, extra=()):
    root = cfg.path(cfg.paths.dataset_root)
    ours = dataset_cameras(root)
    rename = camera_rename_map(ours, base_visual_keys(base))
    cmd = ["lerobot-train",
           f"--policy.path={base}",
           f"--policy.device={cfg.vla.device}",
           "--policy.push_to_hub=false",
           f"--dataset.repo_id={cfg.paths.dataset_repo_id}",
           f"--dataset.root={root}",
           f"--batch_size={batch}",
           f"--steps={steps}",
           f"--save_freq={save_freq}",
           f"--output_dir={output_dir}",
           f"--job_name={Path(output_dir).name}"]
    if rename:
        cmd.append(f"--rename_map={json.dumps(rename)}")
    return cmd + list(extra), rename


def run(cfg, base="lerobot/smolvla_base", steps=20000, batch=32, output_dir="outputs/train/nova2_smolvla",
        save_freq=5000, extra=(), dry=False):
    if shutil.which("lerobot-train") is None:
        raise RuntimeError("lerobot-train not found: pip install -e '.[vla]'")
    output_dir = cfg.path(output_dir)
    cmd, rename = build_command(cfg, base, steps, batch, output_dir, save_freq, extra)
    if rename:
        print(f"camera names mapped onto the base model's: {rename}")
    print(" ".join(shlex.quote(c) for c in cmd))
    if dry:
        return 0
    code = subprocess.call(cmd)
    if code == 0:
        print(f"\ndone: set vla.policy_path: {output_dir}/checkpoints/last/pretrained_model")
    return code
