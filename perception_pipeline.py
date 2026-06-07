#!/usr/bin/env python3
"""perception_pipeline.py — Stage 1: Dual-channel perception layer.

Architecture:
  Channel A — YOLO-BDD100K detection + BoT-SORT tracking, vehicle-only filter
  Channel B — Depth Anything v2 ViT-S monocular depth estimation (~95 MB)

Output per frame: {"frame_id", "timestamp", "objects": [{id, cls_id, class, conf, cx, cy, depth}]}
"""

import sys, os, time, json, pickle
from pathlib import Path

import cv2, torch
import numpy as np
from ultralytics import YOLO

from project_config import BASE_DIR as PROJECT_DIR, MODELS as CFG_MODELS, DEPTH_LIB

OUTPUT_DIR = Path("output/stage1_result")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = "."
sys.path.insert(0, DEPTH_LIB)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── BDD100K class mapping ──
BDD_CLASSES = {
    0: "pedestrian", 1: "rider", 2: "car", 3: "truck", 4: "bus",
    5: "train", 6: "motorcycle", 7: "bicycle", 8: "traffic_light", 9: "traffic_sign",
}
VEHICLE_CLASS_IDS = {2, 3, 4, 5}
CONF_THRESH = 0.25


class TrafficPerceptionSystem:
    """Stage 1: YOLO detection + BoT-SORT tracking + Depth Anything v2 ViT-S."""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INIT] Device: {self.device}")
        self._init_yolo()
        self._init_depth()
        print("[INIT] TrafficPerceptionSystem ready")

    # ── Channel A: YOLO detection + tracking ────────────────

    def _init_yolo(self):
        # Try project_config path first, then legacy fallbacks
        candidates = [
            Path(CFG_MODELS.get("yolo_bdd", "")),                         # models/best.pt
            PROJECT_DIR / "models" / "best.pt",
            PROJECT_DIR / "backup" / "external" / "intelligent_traffic_ai"
                / "bdd_yolo_train_m_vs" / "weights" / "best.pt",
        ]
        model_path = None
        for p in candidates:
            if p.exists():
                model_path = p
                break
        if model_path is None:
            raise FileNotFoundError(
                f"YOLO model not found. Searched: {[str(p) for p in candidates]}"
            )
        print(f"[Detect] Loading YOLO: {model_path}")
        self.yolo_model = YOLO(str(model_path))

    # ── Channel B: Depth Anything v2 ViT-S ───────────────────

    def _init_depth(self):
        print("[Depth] Loading Depth Anything v2 ViT-S ...")
        from depth_anything_v2.dpt import DepthAnythingV2

        ckpt_path = Path(CFG_MODELS.get("depth_vits", "models/depth_anything_v2_vits.pth"))
        if not ckpt_path.exists():
            ckpt_path = PROJECT_DIR / "models" / "depth_anything_v2_vits.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Depth weights not found: {ckpt_path}")

        self.depth_model = DepthAnythingV2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384]
        )
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=True)
        state = ckpt.get("model", ckpt)
        self.depth_model.load_state_dict(state, strict=False)
        self.depth_model.to(self.device)
        self.depth_model.eval()

    # ── Per-frame processing ─────────────────────────────────

    @torch.no_grad()
    def process_frame(self, frame: np.ndarray, frame_id: int,
                      timestamp: float) -> dict:
        """Run dual-channel pipeline on a single frame."""
        H, W = frame.shape[:2]

        # Channel A: YOLO detection + tracking
        results = self.yolo_model.track(
            frame, persist=True, verbose=False, conf=CONF_THRESH
        )[0]

        vehicles_raw = []
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].cpu().numpy()
                track_id = int(boxes.id[i].item())
                x1, y1, x2, y2 = map(int, xyxy)
                cx = max(0, min(int((x1 + x2) / 2), W - 1))
                cy = max(0, min(int((y1 + y2) / 2), H - 1))
                vehicles_raw.append({
                    "id": track_id, "cls_id": cls_id,
                    "class": BDD_CLASSES.get(cls_id, f"cls_{cls_id}"),
                    "cx": cx, "cy": cy, "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                })

        # Channel B: Depth estimation
        depth_map = self._estimate_depth(frame)

        # Sample depth at each vehicle centroid
        objects = []
        for v in vehicles_raw:
            depth_val = float(depth_map[v["cy"], v["cx"]])
            objects.append({
                "id": v["id"], "cls_id": v["cls_id"], "class": v["class"],
                "conf": v["conf"], "cx": v["cx"], "cy": v["cy"],
                "depth": depth_val,
            })

        return {"frame_id": frame_id, "timestamp": timestamp, "objects": objects}

    def _estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Depth Anything v2: return normalised uint8 depth map."""
        self.depth_model.eval()
        with torch.no_grad():
            depth = self.depth_model.infer_image(frame, input_size=518)
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
        h, w = frame.shape[:2]
        if depth_norm.shape != (h, w):
            depth_norm = cv2.resize(depth_norm, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth_norm

    # ── Batch video processing ───────────────────────────────

    def process_video(self, video_path, max_frames=None, start_frame=0,
                      progress_interval=20):
        """Process video and return all frame dicts.

        Returns: (frame_dicts, fps, total_processed)
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 10.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, start_frame + max_frames)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        print(f"[VIDEO] {video_path}  FPS={fps:.1f}  frames={start_frame}→{total - 1}")

        frame_dicts = []
        fi = start_frame
        t0 = time.time()
        while fi < total:
            ret, frame = cap.read()
            if not ret:
                break
            fd = self.process_frame(frame, fi, fi / fps)
            frame_dicts.append(fd)
            fi += 1
            if (fi - start_frame) % progress_interval == 0:
                elapsed = time.time() - t0
                print(f"  [{fi - start_frame}/{total - start_frame}] "
                      f"{(fi - start_frame) / elapsed:.1f} fps  "
                      f"vehicles={len(fd['objects'])}")
        cap.release()
        elapsed = time.time() - t0
        processed = fi - start_frame
        print(f"[VIDEO] Done: {processed} frames in {elapsed:.1f}s "
              f"({processed / elapsed:.1f} fps)")
        return frame_dicts, fps, processed


def run_stage1(video_path=None, max_secs=None):
    """Convenience entry point: process a video and save results."""
    if video_path is None:
        video_path = Path("videos/1/S05_c028.avi")
    video_path = Path(video_path)
    video_name = video_path.stem

    system = TrafficPerceptionSystem()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    max_frames = int(max_secs * fps) if max_secs else total_frames
    print(f"[STAGE1] {video_path.name}  {max_frames} frames  "
          f"({max_frames / fps:.1f}s)")

    frame_dicts, actual_fps, processed = system.process_video(
        video_path, max_frames=max_frames, progress_interval=50
    )

    # Summary
    total_vehicles = set()
    vehicle_types = {}
    for fd in frame_dicts:
        for v in fd["objects"]:
            total_vehicles.add(v["id"])
            vehicle_types[v["class"]] = vehicle_types.get(v["class"], 0) + 1

    print(f"[STAGE1] {processed} frames, {len(total_vehicles)} unique vehicles")
    print(f"[STAGE1] Types: {vehicle_types}")

    # Save
    pkl_path = OUTPUT_DIR / f"frame_dicts_{video_name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(frame_dicts, f)

    summary = {
        "video": str(video_path), "fps": actual_fps,
        "frames_processed": processed, "unique_vehicles": len(total_vehicles),
        "vehicle_type_distribution": vehicle_types, "pipeline_version": "MVP v4",
    }
    json_path = OUTPUT_DIR / f"stage1_summary_{video_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[STAGE1] Saved: {pkl_path.name}, {json_path.name}")
    return frame_dicts, summary


if __name__ == "__main__":
    run_stage1()
