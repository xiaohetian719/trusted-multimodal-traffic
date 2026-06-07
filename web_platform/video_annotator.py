#!/usr/bin/env python3
"""video_annotator.py - Generate annotated output video with detection boxes, tracks, and collision warnings."""

import os, sys, pickle, json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Color palette for vehicle tracks
TRACK_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 128, 128), (128, 255, 128),
    (128, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255),
    (0, 200, 100), (200, 0, 100), (100, 0, 200), (200, 100, 0),
]

COLLISION_RED = (0, 0, 255)
WARNING_ORANGE = (0, 165, 255)


def _get_track_color(track_id: int) -> tuple:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def generate_annotated_video(
    video_path: str,
    pkl_path: str,
    st_graph_path: str = None,
    output_path: str = None,
    max_frames: int = None,
) -> str:
    """Generate annotated MP4 video with detection boxes and collision warnings.

    Args:
        video_path: Path to input video
        pkl_path: Path to Stage 1 frame_dicts pickle
        st_graph_path: Path to Stage 2 ST-Graph JSON (for collision events)
        output_path: Output video path (default: auto-generated)
        max_frames: Limit to N frames (default: all)

    Returns:
        Path to the generated annotated video
    """
    if output_path is None:
        vp = Path(video_path)
        output_path = str(vp.parent / f"{vp.stem}_annotated.mp4")

    # Load frame dicts
    with open(pkl_path, "rb") as f:
        frame_dicts = pickle.load(f)

    if max_frames:
        frame_dicts = frame_dicts[:max_frames]

    # Load collision events
    collision_map = defaultdict(list)  # frame_idx -> list of collision events
    if st_graph_path and os.path.exists(st_graph_path):
        with open(st_graph_path, "r", encoding="utf-8") as f:
            st_graph = json.load(f)
        events = st_graph.get("temporal_event_chains", [])
        fps = 10.0  # default
        for evt in events:
            if evt.get("interaction", {}).get("event_type") == "Collision_Risk":
                ts = evt.get("time_span", "0")
                t_val = float(ts.split()[0]) if ts else 0
                frame_idx = int(t_val * fps)
                collision_map[frame_idx].append(evt)

    # Open video
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, orig_fps, (width, height))

    print(f"[ANNOTATOR] Generating annotated video: {output_path}")
    print(f"[ANNOTATOR] {len(frame_dicts)} frames, {orig_fps:.1f} fps, {width}x{height}")

    for fi, fd in enumerate(frame_dicts):
        # Read original frame
        ret, frame = cap.read()
        if not ret:
            break

        # Draw detection boxes
        for obj in fd.get("objects", []):
            cx, cy = obj.get("cx", 0), obj.get("cy", 0)
            w, h = obj.get("w", 60), obj.get("h", 60)
            track_id = obj.get("id", -1)
            cls_name = obj.get("class", "?")
            conf = obj.get("conf", 0)
            depth = obj.get("depth", 0)

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            color = _get_track_color(track_id)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"#{track_id} {cls_name} {conf:.2f} d:{depth:.1f}m"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Draw collision warnings
        if fi in collision_map:
            for evt in collision_map[fi]:
                inter = evt.get("interaction", {})
                sev = inter.get("severity", "Moderate")
                pred = inter.get("causal_prediction", {})
                ttc = pred.get("min_TTC", "?")

                if sev == "Critical":
                    border_color = (0, 0, 255)
                    alpha = 0.15
                elif sev == "High":
                    border_color = (0, 140, 255)
                    alpha = 0.10
                else:
                    border_color = (0, 200, 255)
                    alpha = 0.05

                # Flash red border
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), border_color, 12)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Warning text at top
                nodes = ", ".join(f.get("node", "?") for f in evt.get("facts", []))
                warn_text = f"!! COLLISION RISK [{sev}] TTC:{ttc} | {nodes}"
                cv2.putText(frame, warn_text, (width // 2 - 350, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, border_color, 2)

        # Frame counter overlay
        cv2.putText(frame, f"Frame: {fi}", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out.write(frame)

    cap.release()
    out.release()

    # Re-encode with H.264 for web compatibility using imageio-ffmpeg
    try:
        import subprocess, imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_path = output_path + ".tmp.mp4"
        os.rename(output_path, temp_path)
        subprocess.run([
            ffmpeg_exe, "-y", "-i", temp_path, "-c:v", "libx264",
            "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ], check=True, capture_output=True, timeout=300)
        os.remove(temp_path)
        print(f"[ANNOTATOR] H.264 re-encode complete - browser ready!")
    except ImportError:
        print(f"[ANNOTATOR] imageio-ffmpeg not installed, keeping MP4V")
        if os.path.exists(temp_path):
            os.rename(temp_path, output_path)
    except Exception as e:
        print(f"[ANNOTATOR] H.264 encode failed: {e}")
        if os.path.exists(temp_path):
            os.rename(temp_path, output_path)
        print(f"[ANNOTATOR] Kept MP4V format - video downloadable")

    print(f"[ANNOTATOR] Done: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--st-graph")
    parser.add_argument("--output")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    generate_annotated_video(args.video, args.pkl, args.st_graph, args.output, args.max_frames)
