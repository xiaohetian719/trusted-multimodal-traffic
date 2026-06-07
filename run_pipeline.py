#!/usr/bin/env python3
"""run_pipeline.py — One-click end-to-end pipeline runner (with evaluation + logging)

Usage:
    python run_pipeline.py                          # process all videos in default dir
    python run_pipeline.py --video path/to/video.avi
    python run_pipeline.py --max-secs 30            # limit each video to 30s
    python run_pipeline.py --skip-s1                # skip Stage 1 (use existing pickle)
    python run_pipeline.py --no-eval                # skip evaluation

Stages:
    S1 → S2 → S3  →  Evaluation
    perception → topology → narrative → metrics+report
"""

import sys, os, json, argparse, time
from pathlib import Path
from datetime import datetime


# ============================================================
#  Terminal Output Logger
# ============================================================

class TeeLogger:
    """Duplicates stdout to both terminal and a log file."""

    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


def enable_logging(output_dir: Path, video_name: str) -> TeeLogger:
    """Enable terminal output logging to a timestamped text file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = output_dir / f'pipeline_log_{video_name}_{timestamp}.txt'
    logger = TeeLogger(str(log_path))
    sys.stdout = logger
    print(f'[LOG] Terminal output will be saved to: {log_path.name}')
    return logger


# ============================================================
#  Dependency Check
# ============================================================

def _check_deps():
    """Verify all required packages are installed."""
    required = {
        'numpy': 'numpy', 'cv2': 'opencv-python', 'torch': 'torch',
        'ultralytics': 'ultralytics', 'requests': 'requests',
        'matplotlib': 'matplotlib', 'scipy': 'scipy',
    }
    missing = []
    for import_name, pip_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f'[ERROR] Missing dependencies: {", ".join(missing)}')
        print(f'[FIX] Run: .\\activate_env.ps1  or  pip install -r requirements.txt')
    return len(missing) == 0, missing


# ============================================================
#  Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'output'
STAGE1_DIR = OUTPUT_DIR / 'stage1_result'
STAGE1_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_VIDEO_DIRS = [
    BASE_DIR / 'videos' / '1',
    BASE_DIR / 'videos',
]


def find_videos() -> list:
    """Auto-discover .avi/.mp4 files in default directories."""
    videos = []
    for d in DEFAULT_VIDEO_DIRS:
        if d.is_dir():
            for ext in ('*.avi', '*.mp4', '*.mov'):
                videos.extend(sorted(d.glob(ext)))
    return [str(v) for v in videos]


# ============================================================
#  Stage 1
# ============================================================

def run_stage1(video_path: str, max_secs: float = None) -> str:
    """Run Stage 1 perception on a single video. Returns pickle path."""
    from perception_pipeline import TrafficPerceptionSystem

    vp = Path(video_path)
    video_name = vp.stem
    system = TrafficPerceptionSystem()

    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    max_frames = int(max_secs * fps) if max_secs else total_frames
    bar = '=' * 60
    print(f'\n{bar}')
    print(f'[PIPELINE] Stage 1: {vp.name}  {max_frames} frames ({max_frames/fps:.1f}s)')
    print(f'{bar}')

    frame_dicts, actual_fps, processed = system.process_video(
        video_path, max_frames=max_frames, progress_interval=100
    )

    total_vehicles = set()
    vehicle_types = {}
    for fd in frame_dicts:
        for v in fd['objects']:
            total_vehicles.add(v['id'])
            vehicle_types[v['class']] = vehicle_types.get(v['class'], 0) + 1

    print(f'[STAGE1] {processed} frames, {len(total_vehicles)} unique vehicles: {vehicle_types}')

    pkl_path = STAGE1_DIR / f'frame_dicts_{video_name}.pkl'
    import pickle
    with open(pkl_path, 'wb') as f:
        pickle.dump(frame_dicts, f)

    summary = {
        'video': str(video_path), 'fps': actual_fps,
        'frames_processed': processed, 'unique_vehicles': len(total_vehicles),
        'vehicle_type_distribution': vehicle_types,
    }
    json_path = STAGE1_DIR / f'stage1_summary_{video_name}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'[STAGE1] Saved: {pkl_path.name}')
    return str(pkl_path)


# ============================================================
#  Evaluation Bridge
# ============================================================

def run_evaluation(video_stem: str):
    """Run the evaluation framework on pipeline outputs.

    Maps pipeline output naming to evaluation framework expected input format,
    then runs all metrics, visualization, and report generation.
    """
    bar = '=' * 60
    print(f'\n{bar}')
    print(f'[EVAL] Running evaluation for {video_stem}...')
    print(f'{bar}')

    # Prepare evaluation data directory
    eval_data_dir = OUTPUT_DIR / 'eval_data' / video_stem
    eval_data_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = OUTPUT_DIR / 'eval_output' / video_stem
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Map pipeline output files to evaluation input naming convention
    st_graph_src = OUTPUT_DIR / f'st_graph_frame_dicts_{video_stem}.json'
    report_src = OUTPUT_DIR / f'traffic_report_st_graph_frame_dicts_{video_stem}.txt'

    st_graph_dst = eval_data_dir / f'st_graph_output_{video_stem}.json'
    report_dst = eval_data_dir / f'traffic_report_{video_stem}.txt'

    if not st_graph_src.exists():
        print(f'[EVAL] WARNING: ST-Graph not found at {st_graph_src}')
        print(f'[EVAL] Skipping evaluation - run full pipeline first.')
        return

    # Copy/map files to evaluation input format
    import shutil
    shutil.copy(str(st_graph_src), str(st_graph_dst))
    print(f'[EVAL] ST-Graph mapped: {st_graph_dst.name}')

    if report_src.exists():
        shutil.copy(str(report_src), str(report_dst))
        print(f'[EVAL] Report mapped: {report_dst.name}')
    else:
        print(f'[EVAL] NOTE: No Stage 3 report found (Ollama may not be running)')
        print(f'[EVAL] Will evaluate data quality only.')

    try:
        from evaluation.config import (
            EvaluationConfig, VisualizationConfig, ReportConfig
        )
        from evaluation.runner import EvaluationRunner
        from evaluation.visualization import VisualizationManager
        from evaluation.report import ReportGenerator

        # Configure evaluation
        eval_config = EvaluationConfig(
            data_path=OUTPUT_DIR / 'eval_data',
            output_path=eval_output_dir,
            report_format='json',
            enable_visualization=True,
        )

        # Run evaluation
        runner = EvaluationRunner(eval_config)
        runner.load_data([video_stem])
        runner.evaluate_all_scenes()
        runner.generate_summary()
        runner.save_results(eval_config.report_format)

        # Print summary
        print(f'\n{bar}')
        print(f'[EVAL] Evaluation Results for {video_stem}')
        print(f'{bar}')
        runner.print_summary_report()

        # Generate visualization
        viz_config = VisualizationConfig()
        viz_manager = VisualizationManager(runner, viz_config)
        viz_manager.generate_all_visualizations()
        print(f'[EVAL] Visualizations saved to: {eval_output_dir / "figures"}')

        # Generate reports
        report_config = ReportConfig()
        report_gen = ReportGenerator(report_config)
        generated = report_gen.generate_all_reports(
            runner, eval_output_dir, formats=['html']
        )
        for fmt, path in generated.items():
            if path:
                print(f'[EVAL] {fmt.upper()} report: {path}')

        print(f'[EVAL] Evaluation complete!')
        return runner

    except ImportError as e:
        print(f'[EVAL] WARNING: Evaluation modules not available: {e}')
        return None
    except Exception as e:
        print(f'[EVAL] ERROR: {e}')
        import traceback
        traceback.print_exc()
        return None


# ============================================================
#  Main Pipeline
# ============================================================

def run_pipeline(video_path: str = None, max_secs: float = None,
                 skip_s1: bool = False, no_eval: bool = False):
    """Run full S1->S2->S3 pipeline, then auto-evaluate.

    Args:
        video_path: path to input video file
        max_secs: limit processing to first N seconds
        skip_s1: skip Stage 1 if pickle already exists
        no_eval: skip evaluation phase
    """
    t_start = time.time()

    if video_path:
        videos = [video_path]
    else:
        videos = find_videos()
        if not videos:
            print('[PIPELINE] No videos found. Specify --video path/to/video.avi')
            return
        print(f'[PIPELINE] Found {len(videos)} video(s):')
        for v in videos:
            print(f'  {v}')

    all_results = {}
    bar = '=' * 60

    for vid in videos:
        vp = Path(vid)
        video_stem = vp.stem
        pkl_path = STAGE1_DIR / f'frame_dicts_{video_stem}.pkl'

        # ---- Stage 1 ----
        if skip_s1 and pkl_path.exists():
            print(f'\n[PIPELINE] Stage 1 SKIP - pickle exists: {pkl_path.name}')
        else:
            run_stage1(vid, max_secs)

        # ---- Stage 2 ----
        print(f'\n{bar}')
        print(f'[PIPELINE] Stage 2: Topology Engine')
        print(f'{bar}')
        from topology_engine import run_stage2
        st_graph, s2_stats = run_stage2(str(pkl_path), str(OUTPUT_DIR))

        # ---- Stage 3 ----
        st_graph_path = OUTPUT_DIR / f'st_graph_frame_dicts_{video_stem}.json'
        print(f'\n{bar}')
        print(f'[PIPELINE] Stage 3: Narrative Agent')
        print(f'{bar}')
        try:
            from narrative_agent import run_stage3
            report = run_stage3(str(st_graph_path), str(OUTPUT_DIR))
        except Exception as e:
            print(f'[PIPELINE] Stage 3 failed (Ollama may not be available): {e}')
            report = None

        elapsed = time.time() - t_start
        print(f'\n{bar}')
        print(f'[PIPELINE] Complete! Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)')
        print(f'[PIPELINE] Output: {OUTPUT_DIR}')
        print(f'{bar}')

        # ---- Evaluation ----
        if not no_eval:
            run_evaluation(video_stem)

        all_results[video_stem] = {
            'report': report,
            'st_graph': st_graph,
        }

    # ---- Pipeline Summary ----
    print(f'\n{bar}')
    print(f'[PIPELINE] Summary')
    print(f'{bar}')
    for name, result in all_results.items():
        status_s3 = 'OK' if result['report'] else 'SKIPPED'
        print(f'  {name}: S1 OK | S2 OK | S3 {status_s3}')
    print(f'[PIPELINE] All output in: {OUTPUT_DIR}')
    print(f'{bar}')


# ============================================================
#  Entry Point
# ============================================================

if __name__ == '__main__':
    ok, missing = _check_deps()
    if not ok:
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description='Trusted Multimodal Traffic - One-Click Pipeline (with Evaluation)'
    )
    parser.add_argument('--video', '-v', help='Path to input video file')
    parser.add_argument('--max-secs', type=float, help='Limit processing to first N seconds')
    parser.add_argument('--skip-s1', action='store_true', help='Skip Stage 1 (use existing pickle)')
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation phase')
    args = parser.parse_args()

    # Enable terminal output logging
    video_tag = Path(args.video).stem if args.video else 'all'
    logger = enable_logging(OUTPUT_DIR, video_tag)

    try:
        run_pipeline(args.video, args.max_secs, args.skip_s1, args.no_eval)
    finally:
        logger.close()
        print(f'\n[LOG] Terminal output saved to: output/')