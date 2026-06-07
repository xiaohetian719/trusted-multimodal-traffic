#!/usr/bin/env python3
"""app.py - Web platform for Traffic AI system: upload video, run pipeline, view results."""

import os, sys, json, uuid, threading, shutil, time, traceback
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, url_for

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ["YOLO_CONFIG_DIR"] = str(BASE_DIR)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max

# Task tracking
tasks = {}
tasks_lock = threading.Lock()


class PipelineTask:
    def __init__(self, task_id: str, video_path: str):
        self.task_id = task_id
        self.video_path = video_path
        self.video_stem = Path(video_path).stem
        self.status = "pending"  # pending, running, annotating, done, error
        self.progress = 0
        self.message = "Waiting..."
        self.results = {}
        self.error = None
        self.start_time = time.time()

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "video_stem": self.video_stem,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "elapsed": round(time.time() - self.start_time, 1),
            "results": self.results,
        }


def _run_pipeline_thread(task: PipelineTask, max_secs: float = None):
    """Run the full pipeline in a background thread, updating task progress."""
    try:
        task.status = "running"
        task.progress = 5
        task.message = "Stage 1: Vehicle detection & depth estimation..."

        # Stage 1
        from perception_pipeline import TrafficPerceptionSystem
        import cv2

        cap = cv2.VideoCapture(task.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        max_frames = int(max_secs * fps) if max_secs else total_frames
        system = TrafficPerceptionSystem()
        frame_dicts, actual_fps, processed = system.process_video(
            task.video_path, max_frames=max_frames, progress_interval=50
        )

        # Save Stage 1 pickle
        STAGE1_DIR = OUTPUT_DIR / "stage1_result"
        STAGE1_DIR.mkdir(parents=True, exist_ok=True)
        import pickle
        pkl_path = STAGE1_DIR / f"frame_dicts_{task.video_stem}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(frame_dicts, f)

        total_vehicles = set()
        for fd in frame_dicts:
            for v in fd["objects"]:
                total_vehicles.add(v["id"])

        task.progress = 30
        task.message = f"Stage 1 done: {processed} frames, {len(total_vehicles)} vehicles"

        # Stage 2
        task.message = "Stage 2: Topology engine & collision detection..."
        from topology_engine import TrafficTopologyEngine
        engine = TrafficTopologyEngine()
        for fd in frame_dicts:
            engine.update_frame(fd)
        engine.finalize_tracks()
        engine.cascade_stitch_tracks()
        engine.refine_tracks()
        engine.finalize_m2_semantics()
        st_graph = engine.generate_st_graph()

        # Save ST-Graph
        st_graph_path = OUTPUT_DIR / f"st_graph_frame_dicts_{task.video_stem}.json"
        with open(st_graph_path, "w", encoding="utf-8") as f:
            json.dump(st_graph, f, indent=2, ensure_ascii=False)

        collision_count = len([e for e in st_graph.get("temporal_event_chains", [])
                              if e.get("interaction", {}).get("event_type") == "Collision_Risk"])
        task.progress = 55
        task.message = f"Stage 2 done: {collision_count} collision risks detected"

        # Stage 3 (try Ollama, fallback to facts-based report)
        task.message = "Stage 3: Generating traffic report..."
        report_text = ""
        try:
            from narrative_agent import run_stage3
            report_text = run_stage3(str(st_graph_path), str(OUTPUT_DIR))
        except Exception as e:
            # Generate basic report from facts
            from narrative_agent import TrafficNarrativeSystem
            agent = TrafficNarrativeSystem()
            facts = agent.serialize_graph_to_facts(st_graph)
            report_text = f"## 交通场景报告\n\n### 场景概览\n- 监控时长: {processed/fps:.1f} 秒\n- 涉及车辆: {len(total_vehicles)} 辆\n- 碰撞风险: {collision_count} 次\n\n### 事实数据\n{facts}\n\n### 安全评估\n（Ollama 未运行，以上为事实数据摘要。）"
            report_path = OUTPUT_DIR / f"traffic_report_st_graph_frame_dicts_{task.video_stem}.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)

        task.progress = 75
        task.message = "Stage 3 done"

        # Annotated video
        task.status = "annotating"
        task.message = "Generating annotated video..."
        from web_platform.video_annotator import generate_annotated_video
        annotated_path = OUTPUT_DIR / f"{task.video_stem}_annotated.mp4"
        generate_annotated_video(
            task.video_path, str(pkl_path), str(st_graph_path), str(annotated_path)
        )
        task.progress = 90

        # Evaluation
        task.message = "Running evaluation..."
        try:
            from evaluation.config import EvaluationConfig, VisualizationConfig, ReportConfig
            from evaluation.runner import EvaluationRunner
            from evaluation.visualization import VisualizationManager
            from evaluation.report import ReportGenerator

            eval_data_dir = OUTPUT_DIR / "eval_data" / task.video_stem
            eval_data_dir.mkdir(parents=True, exist_ok=True)
            eval_output_dir = OUTPUT_DIR / "eval_output" / task.video_stem
            eval_output_dir.mkdir(parents=True, exist_ok=True)

            # Copy files to eval input format
            shutil.copy(str(st_graph_path), str(eval_data_dir / f"st_graph_output_{task.video_stem}.json"))
            report_src = OUTPUT_DIR / f"traffic_report_st_graph_frame_dicts_{task.video_stem}.txt"
            report_dst = eval_data_dir / f"traffic_report_{task.video_stem}.txt"
            if report_src.exists():
                shutil.copy(str(report_src), str(report_dst))

            eval_config = EvaluationConfig(
                data_path=OUTPUT_DIR / "eval_data",
                output_path=eval_output_dir,
                enable_visualization=True,
            )
            runner = EvaluationRunner(eval_config)
            runner.load_data([task.video_stem])
            runner.evaluate_all_scenes()
            runner.generate_summary()
            runner.save_results("json")

            from evaluation.visualization import VisualizationManager, VisualizationConfig
            viz_config = VisualizationConfig()
            viz_manager = VisualizationManager(runner, viz_config)
            viz_manager.generate_all_visualizations()

            from evaluation.report import ReportGenerator, ReportConfig
            report_config = ReportConfig()
            report_gen = ReportGenerator(report_config)
            report_gen.generate_all_reports(runner, eval_output_dir, formats=["html"])

            overall_score = runner.summary.get("average_score", 0)
            task.message = f"Evaluation done. Score: {overall_score:.1f}"
        except Exception as e:
            task.message = f"Evaluation skipped: {str(e)[:80]}"

        # Done
        task.results = {
            "video_stem": task.video_stem,
            "annotated_video": f"{task.video_stem}_annotated.mp4",
            "report_text": report_text,
            "report_file": f"traffic_report_st_graph_frame_dicts_{task.video_stem}.txt",
            "st_graph_file": f"st_graph_frame_dicts_{task.video_stem}.json",
            "eval_dir": f"eval_output/{task.video_stem}",
            "frames_processed": processed,
            "vehicles_detected": len(total_vehicles),
            "collision_risks": collision_count,
            "fps": round(actual_fps, 1),
        }
        task.status = "done"
        task.progress = 100

    except Exception as e:
        task.status = "error"
        task.error = f"{str(e)}\n{traceback.format_exc()}"
        task.message = f"Error: {str(e)[:100]}"


# ============================================================
#  Routes
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save with unique name
    task_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix or ".avi"
    safe_name = f"{task_id}{ext}"
    save_path = UPLOAD_DIR / safe_name
    file.save(str(save_path))

    # Create task
    task = PipelineTask(task_id, str(save_path))
    with tasks_lock:
        tasks[task_id] = task

    return jsonify(task.to_dict())


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json()
    task_id = data.get("task_id")
    max_secs = data.get("max_secs")

    with tasks_lock:
        if task_id not in tasks:
            return jsonify({"error": "Task not found"}), 404
        task = tasks[task_id]

    if task.status not in ("pending", "error"):
        return jsonify({"error": f"Task already {task.status}"}), 400

    # Start pipeline in background thread
    thread = threading.Thread(target=_run_pipeline_thread, args=(task, max_secs), daemon=True)
    thread.start()

    return jsonify({"status": "started", "task_id": task_id})


@app.route("/api/status/<task_id>")
def api_status(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task.to_dict())


@app.route("/api/result/<task_id>")
def api_result(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    if task.status != "done":
        return jsonify({"error": "Task not completed", "status": task.status}), 400

    # Read report text
    report_text = ""
    report_file = OUTPUT_DIR / task.results.get("report_file", "")
    if report_file.exists():
        report_text = report_file.read_text(encoding="utf-8")

    # Read evaluation metrics
    eval_metrics = {}
    eval_json = OUTPUT_DIR / task.results.get("eval_dir", "") / "evaluation_results.json"
    if eval_json.exists():
        try:
            eval_metrics = json.loads(eval_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    # List evaluation charts
    charts = []
    charts_dir = OUTPUT_DIR / task.results.get("eval_dir", "") / "figures"
    if charts_dir.exists():
        charts = sorted([f"/output/{task.results['eval_dir']}/figures/{p.name}"
                        for p in charts_dir.glob("*.png")])

    return jsonify({
        **task.results,
        "report_text": report_text,
        "eval_metrics": eval_metrics,
        "eval_charts": charts,
        "elapsed": round(time.time() - task.start_time, 1),
    })


@app.route("/output/<path:filepath>")
def serve_output(filepath):
    """Serve static output files (videos, charts, reports) with proper MIME types."""
    full_path = OUTPUT_DIR / filepath
    if not full_path.exists():
        return jsonify({"error": "File not found"}), 404

    # Determine MIME type
    ext = full_path.suffix.lower()
    mime_map = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".html": "text/html",
        ".json": "application/json",
        ".txt": "text/plain",
        ".pdf": "application/pdf",
        ".csv": "text/csv",
    }
    mimetype = mime_map.get(ext, "application/octet-stream")

    # For videos, support Range requests (required by HTML5 <video>)
    if ext in (".mp4", ".avi", ".webm", ".mov"):
        return _serve_video_with_range(full_path, mimetype)

    return send_file(str(full_path), mimetype=mimetype)


def _serve_video_with_range(path: Path, mimetype: str):
    """Serve video with HTTP Range support for seeking."""
    from flask import Response
    file_size = path.stat().st_size
    range_header = request.headers.get("Range")

    if range_header:
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0]) if byte_range[0] else 0
        end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1
        if start >= file_size:
            return jsonify({"error": "Range not satisfiable"}), 416

        length = end - start + 1
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(length)

        resp = Response(data, 206, mimetype=mimetype, direct_passthrough=True)
        resp.headers.add("Content-Range", f"bytes {start}-{end}/{file_size}")
        resp.headers.add("Accept-Ranges", "bytes")
        resp.headers.add("Content-Length", str(length))
    else:
        resp = send_file(str(path), mimetype=mimetype)
        resp.headers["Accept-Ranges"] = "bytes"

    resp.headers["Content-Disposition"] = f'attachment; filename="{path.name}"'
    return resp


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})


if __name__ == "__main__":
    print("=" * 60)
    print("  Traffic AI Platform")
    print(f"  Open: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
