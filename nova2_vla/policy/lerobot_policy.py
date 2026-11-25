"""
Any LeRobot policy checkpoint as a GraspPolicy -- SmolVLA (fine-tuned from
lerobot/smolvla_base) is the default for a 16 GB RTX 5080; pi0.5 / pi0 /
GR00T / X-VLA checkpoints load the same way (LeRobot picks the class from
the checkpoint's config.json).

    policy = LeRobotVLA("outputs/train/nova2_smolvla/checkpoints/last/pretrained_model")
    chunk = policy.predict_chunk({"wrist": rgb, "scene": rgb2}, state, "pick up the blue gear housing")

The checkpoint's own pre/post-processors (normalisation, tokenizer, any
camera rename map from training) are loaded with it, so the observation is
passed in the dataset's names: observation.images.<camera>, observation.state.
"""

import logging

import numpy as np

from ..train import camera_rename_map
from .base import ROBOT_TYPE, PolicyError

log = logging.getLogger(__name__)


class LeRobotVLA:
    def __init__(self, path, device="cuda", cameras=None):
        try:
            import torch
            from lerobot.configs import FeatureType, PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
            from lerobot.policies.utils import prepare_observation_for_inference
        except ImportError as e:
            raise PolicyError(f"LeRobot is not installed ({e}): pip install -e '.[vla]'") from e
        self.torch = torch
        self._prepare = prepare_observation_for_inference
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise PolicyError("CUDA is not available to PyTorch: see `python -m nova2_vla check`")
        self.device = device

        cfg = PreTrainedConfig.from_pretrained(str(path))
        cfg.device = device
        self.cfg = cfg
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(str(path), config=cfg)
        self.policy.to(device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(path), preprocessor_overrides={"device_processor": {"device": device}})

        feats = cfg.input_features or {}
        self.image_features = {k: f for k, f in feats.items() if f.type is FeatureType.VISUAL}
        state = feats.get("observation.state")
        action = (cfg.output_features or {}).get("action")
        if state is not None and tuple(state.shape) != (7,):
            raise PolicyError(f"checkpoint state is {tuple(state.shape)}, this cell sends 7 (6 joints + gripper)")
        if action is not None and tuple(action.shape) != (7,):
            raise PolicyError(f"checkpoint action is {tuple(action.shape)}, this cell needs 7 (6 joints + gripper)")
        self.cameras = list(cameras) if cameras else None
        log.info("loaded %s policy from %s on %s (image inputs %s)", cfg.type, path, device,
                 list(self.image_features))

    def reset(self):
        self.policy.reset()
        for p in (self.pre, self.post):
            if hasattr(p, "reset"):
                p.reset()

    def _observation(self, images, state):
        names = [n for n in (self.cameras or list(images)) if n in images]
        ours = [f"observation.images.{n}" for n in names]
        expected = list(self.image_features)
        try:   # the same camera-name mapping `python -m nova2_vla train` used
            rename = camera_rename_map(ours, expected)
        except ValueError as e:
            raise PolicyError(str(e)) from None
        obs = {}
        for name, key in zip(names, ours):
            key = rename.get(key, key)
            rgb = images[name]
            feat = self.image_features.get(key)
            if feat is not None and tuple(feat.shape[1:]) != rgb.shape[:2]:
                import cv2   # trained at another resolution
                rgb = cv2.resize(rgb, (feat.shape[2], feat.shape[1]), interpolation=cv2.INTER_AREA)
            obs[key] = np.ascontiguousarray(rgb)
        if expected and not set(obs) & set(expected):
            raise PolicyError(f"the policy expects cameras {expected}, got {sorted(obs)}")
        obs["observation.state"] = np.asarray(state, dtype=np.float32)
        return obs

    def predict_chunk(self, images, state, task):
        torch = self.torch
        batch = self._prepare(self._observation(images, state), torch.device(self.device), task, ROBOT_TYPE)
        with torch.inference_mode():
            batch = self.pre(batch)
            chunk = self.policy.predict_action_chunk(batch)
            chunk = self.post(chunk)
        if isinstance(chunk, dict):
            chunk = chunk["action"]
        chunk = chunk.float().cpu().numpy()
        if chunk.ndim == 3:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[1] < 7:
            raise PolicyError(f"policy returned actions of shape {chunk.shape}, expected (horizon, 7)")
        return chunk[:, :7]
