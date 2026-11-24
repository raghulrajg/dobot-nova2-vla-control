"""
Find "the object named X" in the wrist camera image using only the
reference photos in objects/X/ -- no training per product.

  1. Proposals: everything on the table that could be an object.
       depth   pixels standing min_object_height above the table plane
               (twin FK + aligned depth), split into connected blobs.
               No model; robust for anything with a few mm of height.
       owlv2   OWLv2's class-agnostic objectness (google/owlv2-base-patch16-
               ensemble) plus generic and object-description text queries:
               catches flat, black or shiny parts the depth camera misses.
  2. Matching: every proposal crop and every reference image goes through
     the same image encoder (DINOv2 by default: strong at instance-level
     look-alike retrieval); a proposal's score is its cosine similarity to
     the reference set (mean of its two best references).
  3. The best proposal above min_similarity is the target. If the
     runner-up is within ambiguity_margin, it is still picked but flagged.

Reference photos work best taken in the cell with the wrist camera at the
detection pose (`python -m nova2_vla capture-refs NAME` does that), a few
per product, turned to different sides.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .localize import height_map

log = logging.getLogger(__name__)


@dataclass
class SceneContext:
    """What the depth proposer needs to know about the camera."""
    K: Optional[np.ndarray]
    T_base_camera: Optional[np.ndarray]
    table_z: Optional[float]


@dataclass(eq=False)          # compared by identity (they hold arrays)
class Candidate:
    box: np.ndarray                 # x0, y0, x1, y1 [px]
    proposal_score: float
    source: str
    mask: Optional[np.ndarray] = None   # HxW bool, when the proposer knows it
    similarity: float = float("nan")


@dataclass
class FindResult:
    target: Optional[Candidate]
    candidates: List[Candidate] = field(default_factory=list)
    reason: str = ""
    ambiguous: bool = False


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_candidates(cands: List[Candidate], iou=0.5) -> List[Candidate]:
    """Non-maximum suppression across proposers; a box with a mask wins
    over one without, then the higher proposal score."""
    order = sorted(cands, key=lambda c: (c.mask is not None, c.proposal_score), reverse=True)
    kept = []
    for c in order:
        if all(box_iou(c.box, k.box) < iou for k in kept):
            kept.append(c)
    return kept


def square_pad(rgb, value=128):
    """Letterbox to a square so encoders that centre-crop keep the whole object."""
    h, w = rgb.shape[:2]
    if h == w:
        return rgb
    side = max(h, w)
    out = np.full((side, side, 3), value, dtype=rgb.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = rgb
    return out


def crop(rgb, box, pad=0.1, min_size=16):
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
    x1, y1 = int(min(w, x1 + px)), int(min(h, y1 + py))
    if x1 - x0 < min_size or y1 - y0 < min_size:
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, x1 = max(0, cx - min_size // 2), min(w, cx + min_size // 2)
        y0, y1 = max(0, cy - min_size // 2), min(h, cy + min_size // 2)
    return rgb[y0:y1, x0:x1]


# ---------------------------------------------------------------- proposers

class DepthProposer:
    name = "depth"

    def __init__(self, min_height=0.006, min_area_px=150, max_area_frac=0.5):
        self.min_height, self.min_area_px, self.max_area_frac = min_height, min_area_px, max_area_frac

    def propose(self, frame, ctx: SceneContext) -> List[Candidate]:
        import cv2
        if frame.depth is None or ctx.K is None or ctx.T_base_camera is None or ctx.table_z is None:
            return []
        hmap = height_map(frame.depth, ctx.K, ctx.T_base_camera, ctx.table_z)
        mask = np.nan_to_num(hmap, nan=0.0) > self.min_height
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        h, w = mask.shape
        out = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < self.min_area_px or area > self.max_area_frac * h * w:
                continue
            blob = labels == i
            peak = float(np.nanmax(np.where(blob, hmap, np.nan)))
            out.append(Candidate(np.array([x, y, x + bw, y + bh], float), min(1.0, peak / 0.05), self.name, blob))
        return out


class Owlv2Proposer:
    name = "owlv2"
    GENERIC = ["an object", "a product", "a part", "a box", "a package", "a tool", "a bottle", "a component"]

    def __init__(self, model="google/owlv2-base-patch16-ensemble", device="cuda", top_k=25,
                 min_score=0.1, max_area_frac=0.5):
        self.model_name, self.device = model, device
        self.top_k, self.min_score, self.max_area_frac = top_k, min_score, max_area_frac
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            self.torch = torch
            self._processor = Owlv2Processor.from_pretrained(self.model_name)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_name).to(self.device).eval()

    def propose(self, frame, ctx: SceneContext, description: str = None) -> List[Candidate]:
        from PIL import Image
        self._load()
        torch = self.torch
        rgb = frame.rgb
        h, w = rgb.shape[:2]
        queries = self.GENERIC + ([f"a photo of {description}"] if description else [])
        inputs = self._processor(text=[queries], images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._model(**inputs)
        objectness = torch.sigmoid(out.objectness_logits[0]).float().cpu().numpy()
        cls = torch.sigmoid(out.logits[0].max(dim=-1).values).float().cpu().numpy()
        score = np.maximum(objectness, cls)
        boxes = out.pred_boxes[0].float().cpu().numpy()      # cx, cy, w, h of the padded square
        side = max(h, w)                                      # OWLv2 pads to a square (bottom / right)
        xyxy = np.stack([(boxes[:, 0] - boxes[:, 2] / 2) * side, (boxes[:, 1] - boxes[:, 3] / 2) * side,
                         (boxes[:, 0] + boxes[:, 2] / 2) * side, (boxes[:, 1] + boxes[:, 3] / 2) * side], 1)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        ok = (score >= self.min_score) & (area > 64) & (area < self.max_area_frac * h * w)
        idx = np.argsort(-score[ok])[: self.top_k * 3]
        cands = [Candidate(xyxy[ok][i], float(score[ok][i]), self.name) for i in idx]
        return merge_candidates(cands, iou=0.5)[: self.top_k]


# ---------------------------------------------------------------- embedders

class Dinov2Embedder:
    name = "dinov2"

    def __init__(self, model="facebook/dinov2-base", device="cuda", batch=32):
        self.model_name, self.device, self.batch = model, device, batch
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            self.torch = torch
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def embed(self, images: List[np.ndarray]) -> np.ndarray:
        from PIL import Image
        self._load()
        torch = self.torch
        out = []
        for i in range(0, len(images), self.batch):
            pil = [Image.fromarray(im) for im in images[i:i + self.batch]]
            inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hidden = self._model(**inputs).last_hidden_state
            # CLS token + mean of the patch tokens: identity and texture
            feat = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=-1).float()
            out.append(torch.nn.functional.normalize(feat, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)


class ColorHistEmbedder:
    """Hue-saturation histogram. No model: for dry runs, tests, and cells
    whose products differ mainly by colour."""
    name = "color_hist"

    def __init__(self, bins=(18, 8)):
        self.bins = bins

    def embed(self, images: List[np.ndarray]) -> np.ndarray:
        import cv2
        out = []
        for im in images:
            hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, list(self.bins), [0, 180, 0, 256]).ravel()
            v = np.sqrt(hist / max(hist.sum(), 1e-9))       # Hellinger: cosine = Bhattacharyya coefficient
            out.append(v / max(np.linalg.norm(v), 1e-9))
        return np.stack(out)


# ---------------------------------------------------------------- finder

class ReferenceObjectFinder:
    def __init__(self, proposers, embedder, min_similarity=0.55, ambiguity_margin=0.03, crop_pad=0.1):
        self.proposers = list(proposers)
        self.embedder = embedder
        self.min_similarity = min_similarity
        self.ambiguity_margin = ambiguity_margin
        self.crop_pad = crop_pad
        self._refs = {}

    def reference_embeddings(self, spec) -> np.ndarray:
        cached = self._refs.get(spec.name)
        if cached and cached[0] == spec.stamp:
            return cached[1]
        views = []
        for im in spec.load_images():
            h, w = im.shape[:2]
            views.append(im)
            views.append(im[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)])   # tighter crop
        emb = self.embedder.embed([square_pad(v) for v in views])
        self._refs[spec.name] = (spec.stamp, emb)
        return emb

    def propose(self, frame, ctx, description=None) -> List[Candidate]:
        cands = []
        for p in self.proposers:
            try:
                if isinstance(p, Owlv2Proposer):
                    cands += p.propose(frame, ctx, description)
                else:
                    cands += p.propose(frame, ctx)
            except Exception as e:   # one proposer failing must not hide the others' results
                log.warning("proposer %s failed: %s", p.name, e)
        return merge_candidates(cands)

    def find(self, spec, frame, ctx: SceneContext) -> FindResult:
        cands = self.propose(frame, ctx, spec.description)
        if not cands:
            return FindResult(None, [], "nothing on the table was found")
        refs = self.reference_embeddings(spec)
        emb = self.embedder.embed([square_pad(crop(frame.rgb, c.box, self.crop_pad)) for c in cands])
        sims = emb @ refs.T                                   # (candidates, reference views)
        k = min(2, sims.shape[1])
        scores = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
        for c, s in zip(cands, scores):
            c.similarity = float(s)
        cands.sort(key=lambda c: c.similarity, reverse=True)
        best = cands[0]
        threshold = float(spec.meta.get("min_similarity", self.min_similarity))
        if best.similarity < threshold:
            return FindResult(None, cands, f"{spec.name!r} not found: best match {best.similarity:.2f} "
                                           f"< {threshold:.2f} among {len(cands)} objects")
        ambiguous = len(cands) > 1 and cands[1].similarity >= best.similarity - self.ambiguity_margin
        reason = f"matched {spec.name!r} with similarity {best.similarity:.2f}"
        if ambiguous:
            reason += f" (runner-up {cands[1].similarity:.2f} is close: add more reference photos)"
        return FindResult(best, cands, reason, ambiguous)


def make_finder(pcfg) -> ReferenceObjectFinder:
    proposers = []
    for part in pcfg.proposer.split("+"):
        if part == "depth":
            proposers.append(DepthProposer(pcfg.min_object_height_m, pcfg.min_area_px))
        elif part == "owlv2":
            proposers.append(Owlv2Proposer(pcfg.owl_model, pcfg.device))
    embedder = (Dinov2Embedder(pcfg.embed_model, pcfg.device) if pcfg.embedder == "dinov2"
                else ColorHistEmbedder())
    return ReferenceObjectFinder(proposers, embedder, pcfg.min_similarity, pcfg.ambiguity_margin)


def annotate(rgb, result: FindResult, path=None):
    """Draw the candidates (grey), the target (green) and their scores."""
    import cv2
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for c in result.candidates:
        colour = (0, 200, 0) if c is result.target else (160, 160, 160)
        x0, y0, x1, y1 = [int(v) for v in c.box]
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, 2 if c is result.target else 1)
        cv2.putText(img, f"{c.similarity:.2f}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    if path is not None:
        cv2.imwrite(str(path), img)
    return img
