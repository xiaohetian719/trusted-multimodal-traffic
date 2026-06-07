#!/usr/bin/env python3
"""topology_engine.py — Stage 2: Spatiotemporal topology graph engine.

Four-phase pipeline:
  Phase 1: Dual-pool split (stable ≥15f + avg_conf ≥0.4 vs suspended fragments)
  Phase 2: LCSS cascade stitching with Hungarian global assignment
  Phase 3: Trajectory refinement (gap interpolation + uniform smoothing + stationary detection)
  Phase 4: M2 curvature-integral turning semantics with adaptive epsilon

Consumes Stage 1 pickle → produces ST-Graph JSON for Stage 3.
"""

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

VEHICLE_CLS_IDS = {2, 3, 4, 5}          # car / truck / bus / train
SLIDING_WINDOW_SIZE = 15                # frames for FSM
STATIONARY_CONFIDENCE_RATIO = 0.80      # ratio of low-delta frames to lock stationary
DELTA_D_STATIONARY_MAX = 1.0            # max |Δdepth| for stationary
DELTA_D_CREEP_MAX = 3.0                 # max |Δdepth| for creeping
DELTA_D_CRUISE_MAX = 8.0                # max |Δdepth| for cruising
TTC_SAFETY_THRESHOLD = 1.5              # seconds
COLLISION_PROXIMITY_THRESHOLD = 150.0   # pixels: max 2D distance for collision concern
COLLISION_APPROACH_MIN = 0.5            # pixels/frame: min approach speed to be concerning
STATIONARY_PIXEL_THRESHOLD = 20.0       # total displacement → stationary override
FOLLOWING_LATERAL_THRESHOLD = 80        # |cx_a - cx_b| → same lane
M2_MIN_TRAJECTORY_LENGTH = 5            # minimum frames for turning analysis


@dataclass
class ObjectState:
    """Per-object state tracked across frames."""

    obj_id: int
    obj_type: str = "vehicle"
    first_seen: float = 0.0
    last_seen: float = 0.0
    depth_history: deque = field(default_factory=lambda: deque(maxlen=SLIDING_WINDOW_SIZE))
    pos_history: deque = field(default_factory=lambda: deque(maxlen=SLIDING_WINDOW_SIZE))
    time_history: deque = field(default_factory=lambda: deque(maxlen=SLIDING_WINDOW_SIZE))
    class_history: deque = field(default_factory=lambda: deque(maxlen=SLIDING_WINDOW_SIZE * 2))
    fsm_state: str = "Unknown"
    state_start_time: float = 0.0
    state_lock_counter: int = 0
    total_frames_seen: int = 0
    depth_smoothed: float = 0.0
    delta_d_smoothed: float = 0.0
    destroyed: bool = False

    # Phase 1 dual-pool split: YOLO confidence per frame
    conf_history: list = field(default_factory=list)

    # Full unbounded trajectories for M2 / Phase 3
    full_pos_history: list = field(default_factory=list)
    full_depth_history: list = field(default_factory=list)
    full_time_history: list = field(default_factory=list)
    turn_semantic: str = "Unknown"
    longitudinal_trend: str = "Unknown"
    following: Optional[str] = None
    _m2_overridden_by_stationary: bool = False

    # Phase 3: pre-refinement snapshots
    _raw_pos_history: list = field(default_factory=list)
    _raw_depth_history: list = field(default_factory=list)
    stationary_segments: list = field(default_factory=list)


class TrafficTopologyEngine:
    """Stage 2: four-phase topology engine consuming Stage 1 frame dicts."""

    def __init__(self):
        self.frame_count = 0
        self.current_time = 0.0
        self.total_duration = 0.0
        self.scene_environment = "Unknown"

        # Phase 1 params
        self.min_frame_len = 15
        self.avg_conf_thresh = 0.4

        # Phase 2 params
        self.max_frame_gap = 150
        self.lcss_spatial_epsilon = 30
        self.lcss_similarity_thresh = 0.8

        # Phase 3 params
        self.smooth_window_size = 15
        self.stationary_long_period = 60
        self.stationary_median_window = 45
        self.min_gap_frames = 3

        # Phase 4 params
        self.epsilon = None

        self.objects: Dict[int, ObjectState] = {}
        self.stable_tracks: Dict[int, ObjectState] = {}
        self.suspended_fragments: Dict[int, ObjectState] = {}
        self.event_chains: List[dict] = []

    # ── Per-frame ingestion ──────────────────────────────────

    def update_frame(self, frame_dict: dict):
        """Ingest one Stage 1 frame dict, update all per-track caches."""
        self.frame_count += 1
        self.current_time = frame_dict["timestamp"]
        self.total_duration = max(self.total_duration, self.current_time)

        all_objects = frame_dict.get("objects") or frame_dict.get("vehicles", [])
        if all_objects and "cls_id" in all_objects[0]:
            vehicles = [o for o in all_objects if o["cls_id"] in VEHICLE_CLS_IDS]
        else:
            vehicles = all_objects

        for v in vehicles:
            vid = v["id"]
            if vid < 0:
                continue

            cx, cy = v["cx"], v["cy"]
            depth_val = v["depth"]
            pos_pt = (float(cx), float(cy))

            if vid not in self.objects:
                self.objects[vid] = ObjectState(
                    obj_id=vid,
                    first_seen=self.current_time,
                    state_start_time=self.current_time,
                )

            obj = self.objects[vid]
            obj.last_seen = self.current_time
            obj.total_frames_seen += 1
            obj.depth_history.append(depth_val)
            obj.pos_history.append(pos_pt)
            obj.time_history.append(self.current_time)
            obj.full_pos_history.append(pos_pt)
            obj.full_depth_history.append(depth_val)
            obj.full_time_history.append(self.current_time)
            obj.conf_history.append(v.get("conf", 0.0))
            obj.class_history.append(v.get("class", "unknown"))

            self._update_m2_per_frame(obj)
            self._update_fsm(obj)

        if vehicles:
            self._check_interactions(vehicles)

        # Garbage-collect objects unseen > 3s
        for oid, obj in list(self.objects.items()):
            if not obj.destroyed and self.current_time - obj.last_seen > 3.0:
                obj.destroyed = True

    # ── M2 per-frame: longitudinal trend + stationary override

    def _update_m2_per_frame(self, obj: ObjectState):
        """Lightweight per-frame M2 checks. Full curvature integral deferred to Phase 4."""
        depths = obj.full_depth_history
        if len(depths) >= M2_MIN_TRAJECTORY_LENGTH:
            obj.longitudinal_trend = (
                "Approaching" if depths[-1] < depths[0] else "Departing"
            )

        pts = obj.full_pos_history
        if len(pts) >= 2:
            total_disp = float(np.linalg.norm(
                np.array(pts[-1], dtype=np.float64) - np.array(pts[0], dtype=np.float64)
            ))
            if total_disp < STATIONARY_PIXEL_THRESHOLD:
                obj.turn_semantic = "Stationary"
                obj.longitudinal_trend = "Stationary"
                obj._m2_overridden_by_stationary = True
                obj.fsm_state = "Stationary"
            else:
                obj._m2_overridden_by_stationary = False

    # ── FSM motion semantics (15-frame sliding window) ────────

    def _update_fsm(self, obj: ObjectState):
        """Finite state machine: classify motion from 15-frame depth deltas."""
        if obj._m2_overridden_by_stationary:
            obj.fsm_state = "Stationary"
            return
        if len(obj.depth_history) < 3:
            return

        depths = list(obj.depth_history)
        times = list(obj.time_history)
        deltas = []
        for i in range(1, len(depths)):
            dt = times[i] - times[i - 1]
            if dt > 0.001:
                deltas.append((depths[i] - depths[i - 1]) / dt)
        if not deltas:
            return

        obj.delta_d_smoothed = float(np.clip(np.mean(deltas), -50, 50))
        obj.depth_smoothed = (
            float(np.mean(depths[-5:])) if len(depths) >= 5 else float(depths[-1])
        )

        stationary_ratio = sum(1 for d in deltas if abs(d) < DELTA_D_STATIONARY_MAX) / len(deltas)
        prev_state = obj.fsm_state

        if stationary_ratio >= STATIONARY_CONFIDENCE_RATIO:
            obj.fsm_state = "Stationary"
        else:
            abs_dd = abs(obj.delta_d_smoothed)
            if abs_dd < DELTA_D_STATIONARY_MAX:
                obj.fsm_state = "Stationary"
            elif abs_dd < DELTA_D_CREEP_MAX:
                obj.fsm_state = "Creeping"
            elif abs_dd < DELTA_D_CRUISE_MAX:
                obj.fsm_state = "Cruising"
            elif obj.delta_d_smoothed > 0:
                obj.fsm_state = "Accelerating"
            else:
                obj.fsm_state = "Decelerating"

        if obj.fsm_state != prev_state:
            obj.state_start_time = self.current_time
            obj.state_lock_counter = 0
        else:
            obj.state_lock_counter += 1

    # ── Phase 1: Dual-pool split ─────────────────────────────

    def finalize_tracks(self) -> tuple:
        """Split ingested tracks into stable (≥15f, avg_conf≥0.4) and suspended."""
        self.stable_tracks.clear()
        self.suspended_fragments.clear()
        ejected_ids = []

        for oid, obj in self.objects.items():
            avg_conf = float(np.mean(obj.conf_history)) if obj.conf_history else 0.0
            if obj.total_frames_seen >= self.min_frame_len and avg_conf >= self.avg_conf_thresh:
                self.stable_tracks[oid] = obj
            else:
                self.suspended_fragments[oid] = obj
                ejected_ids.append(oid)

        for oid in ejected_ids:
            self.objects.pop(oid, None)

        print(f"[PHASE1] {len(self.stable_tracks)} stable + "
              f"{len(self.suspended_fragments)} suspended")
        return len(self.stable_tracks), len(self.suspended_fragments)

    # ── Phase 2: LCSS cascade stitching ─────────────────────

    @staticmethod
    def _lcss_2d(traj_a: list, traj_b: list, eps: float) -> tuple:
        """Longest Common Subsequence similarity for 2D point sequences."""
        n, m = len(traj_a), len(traj_b)
        if n == 0 or m == 0:
            return 0, 0.0
        dp = np.zeros((n + 1, m + 1), dtype=np.int32)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if np.linalg.norm(
                    np.array(traj_a[i - 1], dtype=np.float64)
                    - np.array(traj_b[j - 1], dtype=np.float64)
                ) <= eps:
                    dp[i, j] = dp[i - 1, j - 1] + 1
                else:
                    dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
        lcss_len = int(dp[n, m])
        return lcss_len, lcss_len / max(n, m)

    @staticmethod
    def _extrapolate_trajectory(obj: ObjectState) -> np.ndarray:
        """Estimate velocity vector from tail of position history."""
        pts = obj.full_pos_history
        if len(pts) < 3:
            return np.array([0.0, 0.0])
        tail = pts[-5:] if len(pts) >= 5 else pts
        velocities = []
        for i in range(1, len(tail)):
            velocities.append(
                np.array(tail[i], dtype=np.float64) - np.array(tail[i - 1], dtype=np.float64)
            )
        return np.mean(velocities, axis=0) if velocities else np.array([0.0, 0.0])

    def cascade_stitch_tracks(self) -> dict:
        """Phase 2: LCSS + Hungarian global assignment for trajectory stitching."""
        all_tracks = {}
        all_tracks.update(self.stable_tracks)
        all_tracks.update(self.suspended_fragments)

        if len(all_tracks) < 2:
            print("[PHASE2] <2 tracks, nothing to stitch")
            return {"pairs_evaluated": 0, "stitched": 0, "merges": []}

        COMPARE_PTS = 20
        MIN_PTS = 8
        MAX_DEPTH_DIFF = 50.0
        candidates = []
        track_items = list(all_tracks.items())

        for i in range(len(track_items)):
            for j in range(len(track_items)):
                if i == j:
                    continue
                oid_a, obj_a = track_items[i]
                oid_b, obj_b = track_items[j]

                # Temporal ordering: A dies before B is born
                if obj_a.last_seen >= obj_b.first_seen:
                    continue

                frame_gap = int((obj_b.first_seen - obj_a.last_seen) * 10)
                if frame_gap < 2 or frame_gap > self.max_frame_gap:
                    continue

                # Depth sanity check
                depth_a_end = (
                    float(np.mean(obj_a.full_depth_history[-5:]))
                    if obj_a.full_depth_history else 0.0
                )
                depth_b_start = (
                    float(np.mean(obj_b.full_depth_history[:5]))
                    if obj_b.full_depth_history else 0.0
                )
                depth_cost = abs(depth_a_end - depth_b_start)
                if depth_cost > MAX_DEPTH_DIFF:
                    continue

                # Extrapolate A forward through the gap
                vel = self._extrapolate_trajectory(obj_a)
                last_pt = np.array(obj_a.full_pos_history[-1], dtype=np.float64)
                n_extrap = max(1, int(frame_gap))
                virtual_bridge = [tuple(last_pt + vel * k) for k in range(1, n_extrap + 1)]

                a_tail = (
                    obj_a.full_pos_history[-COMPARE_PTS:]
                    if len(obj_a.full_pos_history) >= COMPARE_PTS
                    else obj_a.full_pos_history[:]
                )
                virtual_traj = a_tail + virtual_bridge
                b_head = (
                    obj_b.full_pos_history[:COMPARE_PTS]
                    if len(obj_b.full_pos_history) >= COMPARE_PTS
                    else obj_b.full_pos_history[:]
                )

                if len(virtual_traj) < MIN_PTS or len(b_head) < MIN_PTS:
                    continue

                _, lcss_sim = self._lcss_2d(virtual_traj, b_head, self.lcss_spatial_epsilon)
                if lcss_sim < self.lcss_similarity_thresh:
                    continue

                # Prevent fragment from absorbing a stable track
                if oid_a not in self.stable_tracks and oid_b in self.stable_tracks:
                    continue

                candidates.append((oid_a, oid_b, obj_a, obj_b, lcss_sim, depth_cost))

        n_evaluated = len(track_items) * (len(track_items) - 1)
        print(f"[PHASE2] {len(candidates)} candidates from {n_evaluated} pairs")

        if not candidates:
            return {"pairs_evaluated": n_evaluated, "stitched": 0, "merges": []}

        # Hungarian assignment via cost matrix (depth_cost)
        old_ids = list(set(c[0] for c in candidates))
        new_ids = list(set(c[1] for c in candidates))
        old_idx = {oid: i for i, oid in enumerate(old_ids)}
        new_idx = {nid: j for j, nid in enumerate(new_ids)}
        cost_matrix = np.full((len(old_ids), len(new_ids)), 1e9)
        candidate_map = {}

        for c in candidates:
            oi, nj = old_idx[c[0]], new_idx[c[1]]
            cost_matrix[oi, nj] = c[5]
            candidate_map[(oi, nj)] = c

        if _HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            print("[PHASE2] scipy not available — using greedy assignment")
            used_cols = set()
            used_rows = set()
            for c in sorted(candidates, key=lambda x: x[4] - x[5] * 0.01):
                oi, nj = old_idx[c[0]], new_idx[c[1]]
                if oi not in used_rows and nj not in used_cols:
                    used_rows.add(oi)
                    used_cols.add(nj)
            row_ind = np.array(list(used_rows), dtype=np.int64)
            col_ind = np.array(list(used_cols), dtype=np.int64)

        # Execute merges
        merges = []
        for r, c_idx in zip(row_ind, col_ind):
            if cost_matrix[r, c_idx] >= 1e8:
                continue
            cand = candidate_map.get((r, c_idx))
            if cand is None:
                continue
            _, _, obj_old, obj_new, lcss_sim, depth_cost = cand

            self._merge_track_data(obj_old, obj_new)
            self.stable_tracks.pop(cand[1], None)
            self.suspended_fragments.pop(cand[1], None)
            self.objects.pop(cand[1], None)

            merges.append((cand[0], cand[1], round(lcss_sim, 3), round(depth_cost, 1)))
            print(f"[PHASE2]   stitched: {cand[1]} → {cand[0]}  "
                  f"LCSS={lcss_sim:.3f}  Δdepth={depth_cost:.1f}")

        # Clear ghost noise (<5 frame fragments not claimed by any merge)
        ghost_cleared = 0
        merged_new_ids = {m[1] for m in merges}
        for oid in list(self.suspended_fragments.keys()):
            if oid not in merged_new_ids and self.suspended_fragments[oid].total_frames_seen < 5:
                del self.suspended_fragments[oid]
                ghost_cleared += 1

        print(f"[PHASE2] {len(merges)} merged, {ghost_cleared} ghosts cleared, "
              f"{len(self.suspended_fragments)} fragments remain, "
              f"{len(self.stable_tracks)} stable")

        return {
            "pairs_evaluated": n_evaluated, "stitched": len(merges),
            "merges": merges, "ghosts_cleared": ghost_cleared,
        }

    def _merge_track_data(self, obj_old: ObjectState, obj_new: ObjectState):
        """Append obj_new's history chains into obj_old."""
        obj_old.full_pos_history.extend(obj_new.full_pos_history)
        obj_old.full_depth_history.extend(obj_new.full_depth_history)
        obj_old.full_time_history.extend(obj_new.full_time_history)
        obj_old.conf_history.extend(obj_new.conf_history)

        for pt in obj_new.pos_history:
            obj_old.pos_history.append(pt)
        for d in obj_new.depth_history:
            obj_old.depth_history.append(d)
        for t in obj_new.time_history:
            obj_old.time_history.append(t)
        for c in obj_new.class_history:
            obj_old.class_history.append(c)

        obj_old.last_seen = max(obj_old.last_seen, obj_new.last_seen)
        obj_old.total_frames_seen += obj_new.total_frames_seen

    # ── Phase 3: Trajectory refinement ──────────────────────

    def _interpolate_gaps(self, obj: ObjectState) -> tuple:
        """Detect and fill temporal gaps via linear interpolation (conf × 0.5)."""
        times = obj.full_time_history
        positions = obj.full_pos_history
        depths = obj.full_depth_history
        confs = obj.conf_history
        if len(times) < 2:
            return 0, 0

        sorted_data = sorted(zip(times, positions, depths, confs), key=lambda x: x[0])
        times = [x[0] for x in sorted_data]
        positions = [x[1] for x in sorted_data]
        depths = [x[2] for x in sorted_data]
        confs = [x[3] for x in sorted_data]

        diffs = [times[i] - times[i - 1] for i in range(1, len(times))]
        typical_dt = float(np.median(diffs)) if diffs else 0.1

        new_times, new_positions = [times[0]], [positions[0]]
        new_depths, new_confs = [depths[0]], [confs[0]]
        n_gaps, n_interp = 0, 0

        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            n_missing = int(round(dt / typical_dt)) - 1
            if n_missing >= self.min_gap_frames:
                n_gaps += 1
                for k in range(1, n_missing + 1):
                    alpha = k / (n_missing + 1)
                    cx = positions[i - 1][0] + alpha * (positions[i][0] - positions[i - 1][0])
                    cy = positions[i - 1][1] + alpha * (positions[i][1] - positions[i - 1][1])
                    d = depths[i - 1] + alpha * (depths[i] - depths[i - 1])
                    c = (confs[i - 1] + confs[i]) / 2.0 * 0.5
                    new_times.append(times[i - 1] + alpha * dt)
                    new_positions.append((float(cx), float(cy)))
                    new_depths.append(float(d))
                    new_confs.append(float(c))
                    n_interp += 1
            new_times.append(times[i])
            new_positions.append(positions[i])
            new_depths.append(depths[i])
            new_confs.append(confs[i])

        obj.full_time_history = new_times
        obj.full_pos_history = new_positions
        obj.full_depth_history = new_depths
        obj.conf_history = new_confs
        obj.total_frames_seen = len(new_times)
        return n_gaps, n_interp

    def _uniform_weighted_smooth(self, obj: ObjectState):
        """15-frame centred uniform smoothing — confidence NOT used as weight."""
        n = len(obj.full_pos_history)
        if n < 3:
            return

        positions = np.array(obj.full_pos_history, dtype=np.float64)
        depths = np.array(obj.full_depth_history, dtype=np.float64)
        smoothed_pos = np.zeros_like(positions)
        smoothed_dep = np.zeros_like(depths)
        half = self.smooth_window_size // 2

        for i in range(n):
            left = max(0, i - half)
            right = min(n, i + half + 1)
            if right > left:
                smoothed_pos[i] = np.mean(positions[left:right], axis=0)
                smoothed_dep[i] = float(np.mean(depths[left:right]))
            else:
                smoothed_pos[i] = positions[i]
                smoothed_dep[i] = depths[i]

        obj.full_pos_history = [(float(p[0]), float(p[1])) for p in smoothed_pos]
        obj.full_depth_history = [float(d) for d in smoothed_dep]

    def _detect_and_stabilize_stationary(self, obj: ObjectState) -> int:
        """Detect ≥60-frame stationary plateaus, apply median anchoring.

        If >95% of trajectory is stationary, override semantics to Stationary.
        """
        positions = np.array(obj.full_pos_history, dtype=np.float64)
        depths = np.array(obj.full_depth_history, dtype=np.float64)
        n = len(positions)
        if n < self.stationary_long_period:
            return 0

        K = self.stationary_long_period
        stationary_mask = np.zeros(n, dtype=bool)

        for i in range(n - K + 1):
            segment = positions[i:i + K]
            net_disp = float(np.linalg.norm(segment[-1] - segment[0]))
            if net_disp < STATIONARY_PIXEL_THRESHOLD:
                variance = float(np.sum(np.var(segment, axis=0)))
                if variance > 1.0:   # real YOLO jitter, not a constant value
                    stationary_mask[i:i + K] = True

        n_stat_frames = int(np.sum(stationary_mask))
        if n_stat_frames == 0:
            return 0

        # Median anchoring on stationary frames
        M = self.stationary_median_window
        half_m = M // 2
        smoothed_pos = positions.copy()
        smoothed_dep = depths.copy()
        for i in range(n):
            if stationary_mask[i]:
                left = max(0, i - half_m)
                right = min(n, i + half_m + 1)
                smoothed_pos[i, 0] = float(np.median(positions[left:right, 0]))
                smoothed_pos[i, 1] = float(np.median(positions[left:right, 1]))
                smoothed_dep[i] = float(np.median(depths[left:right]))

        obj.full_pos_history = [(float(p[0]), float(p[1])) for p in smoothed_pos]
        obj.full_depth_history = [float(d) for d in smoothed_dep]

        # Extract contiguous segments
        segments = []
        in_seg = False
        seg_start = 0
        for i in range(n):
            if stationary_mask[i] and not in_seg:
                seg_start = i
                in_seg = True
            elif not stationary_mask[i] and in_seg:
                if i - seg_start >= K:
                    segments.append((seg_start, i - 1))
                in_seg = False
        if in_seg and n - seg_start >= K:
            segments.append((seg_start, n - 1))
        obj.stationary_segments = segments

        # Override only if >95% stationary
        if n_stat_frames > n * 0.95:
            obj.turn_semantic = "Stationary"
            obj.longitudinal_trend = "Stationary"
            obj._m2_overridden_by_stationary = True

        return len(segments)

    def refine_tracks(self) -> dict:
        """Phase 3: gap interpolation → uniform smoothing → stationary detection."""
        stats = {
            "tracks_processed": 0, "total_gaps_found": 0,
            "total_interpolated_frames": 0, "total_stationary_segments": 0,
            "total_stationary_frames": 0, "per_track": [],
        }

        for oid, obj in self.stable_tracks.items():
            obj._raw_pos_history = list(obj.full_pos_history)
            obj._raw_depth_history = list(obj.full_depth_history)

            n_gaps, n_interp = self._interpolate_gaps(obj)
            self._uniform_weighted_smooth(obj)
            n_stat_seg = self._detect_and_stabilize_stationary(obj)

            n_stat_frames = sum(e - s + 1 for s, e in obj.stationary_segments)
            stats["tracks_processed"] += 1
            stats["total_gaps_found"] += n_gaps
            stats["total_interpolated_frames"] += n_interp
            stats["total_stationary_segments"] += n_stat_seg
            stats["total_stationary_frames"] += n_stat_frames

            if n_gaps > 0 or n_interp > 0 or n_stat_seg > 0:
                stats["per_track"].append({
                    "track_id": oid, "total_frames": obj.total_frames_seen,
                    "n_gaps": n_gaps, "n_interpolated": n_interp,
                    "n_stationary_segments": n_stat_seg,
                    "n_stationary_frames": n_stat_frames,
                    "stationary_segments": [
                        {"start_idx": s, "end_idx": e} for s, e in obj.stationary_segments
                    ],
                })

        stats["per_track"].sort(
            key=lambda t: t["n_interpolated"] + t["n_stationary_frames"], reverse=True
        )
        print(f"[PHASE3] {stats['tracks_processed']} tracks → "
              f"{stats['total_gaps_found']} gaps, {stats['total_interpolated_frames']} interp, "
              f"{stats['total_stationary_segments']} stat-segments "
              f"({stats['total_stationary_frames']} frames)")
        return stats

    # ── Phase 4 / M2: Curvature-integral turning semantics ───

    @staticmethod
    def _curvature_integral(positions: list, min_speed: float = 0.5) -> float:
        """Total signed bending energy W = Σ ω_t via np.gradient curvature.

        W > 0 → right turn (clockwise in image y↓ coords)
        W < 0 → left turn  (counter-clockwise)
        |W| ≈ total turning angle in radians.
        """
        if len(positions) < M2_MIN_TRAJECTORY_LENGTH:
            return 0.0

        pts = np.array(positions, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        xp, yp = np.gradient(x), np.gradient(y)
        xpp, ypp = np.gradient(xp), np.gradient(yp)

        speed_sq = xp * xp + yp * yp
        valid = speed_sq > min_speed * min_speed
        if float(np.mean(valid)) < 0.3:
            return 0.0

        omega = np.zeros_like(x)
        omega[valid] = (xp[valid] * ypp[valid] - yp[valid] * xpp[valid]) / speed_sq[valid]
        omega = np.clip(omega, -np.pi, np.pi)
        return float(np.sum(omega))

    def _compute_adaptive_epsilon(self) -> float:
        """Fit ε at P85 of |W| distribution; clamp [0.12, 1.5] rad (≈7°–86°)."""
        abs_Ws = []
        for oid, obj in self.stable_tracks.items():
            if obj.total_frames_seen < M2_MIN_TRAJECTORY_LENGTH:
                continue
            if obj._m2_overridden_by_stationary or obj.fsm_state == "Stationary":
                continue
            W = self._curvature_integral(obj.full_pos_history)
            abs_Ws.append(abs(W))

        if len(abs_Ws) < 3:
            fallback = 0.15
            print(f"[PHASE4] too few tracks ({len(abs_Ws)}), fallback ε = {fallback:.4f} rad")
            return fallback

        abs_Ws.sort()
        epsilon = float(np.percentile(abs_Ws, 85))
        epsilon = max(0.12, min(epsilon, 1.50))

        p50, p75, p90, p95 = np.percentile(abs_Ws, [50, 75, 90, 95])
        print(f"[PHASE4] |W| ∈ [{abs_Ws[0]:.4f}, {abs_Ws[-1]:.4f}], "
              f"P50={p50:.4f}, P75={p75:.4f}, P85={epsilon:.4f}, "
              f"P90={p90:.4f}, P95={p95:.4f} → ε = {epsilon:.4f} rad ({np.degrees(epsilon):.1f}°)")
        return epsilon

    def finalize_m2_semantics(self) -> dict:
        """Phase 4: curvature-integral turning classification for all stable tracks."""
        self.epsilon = self._compute_adaptive_epsilon()

        results = []
        counts = {"Go Straight": 0, "Left Turn": 0, "Right Turn": 0}

        for oid, obj in self.stable_tracks.items():
            if obj.total_frames_seen < M2_MIN_TRAJECTORY_LENGTH:
                continue
            if obj._m2_overridden_by_stationary:
                obj.turn_semantic = "Stationary"
                obj.longitudinal_trend = "Stationary"
                continue

            W = self._curvature_integral(obj.full_pos_history)
            if abs(W) <= self.epsilon:
                obj.turn_semantic = "Go Straight"
                counts["Go Straight"] += 1
            elif W > self.epsilon:
                obj.turn_semantic = "Right Turn"
                counts["Right Turn"] += 1
            else:
                obj.turn_semantic = "Left Turn"
                counts["Left Turn"] += 1

            # Re-run longitudinal trend on refined data
            depths = obj.full_depth_history
            if len(depths) >= M2_MIN_TRAJECTORY_LENGTH:
                obj.longitudinal_trend = (
                    "Approaching" if depths[-1] < depths[0] else "Departing"
                )

            results.append({
                "track_id": oid, "W": round(W, 6),
                "turn_semantic": obj.turn_semantic,
                "longitudinal_trend": obj.longitudinal_trend,
                "n_frames": obj.total_frames_seen,
            })

        print(f"[PHASE4] ε = {self.epsilon:.4f} rad ({np.degrees(self.epsilon):.1f}°), "
              f"{len(results)} tracks → Straight: {counts['Go Straight']}, "
              f"Left: {counts['Left Turn']}, Right: {counts['Right Turn']}")

        return {
            "epsilon": round(self.epsilon, 6),
            "epsilon_deg": round(float(np.degrees(self.epsilon)), 2),
            "tracks_evaluated": len(results),
            "distribution": counts,
            "per_track": results,
        }

    # ── TTC & interactions ──────────────────────────────────

    def compute_ttc(self, obj_a: ObjectState, obj_b: ObjectState) -> Optional[float]:
        """Time-to-collision using 2D Euclidean distance and approach rate.

        Only meaningful when vehicles are spatially close (< COLLISION_PROXIMITY_THRESHOLD px)
        and approaching each other (distance decreasing over recent frames).
        """
        if not obj_a.pos_history or not obj_b.pos_history:
            return None

        # Get most recent 2D positions
        pa = obj_a.pos_history[-1]
        pb = obj_b.pos_history[-1]
        dist_2d = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5

        # Skip if vehicles are far apart in image space
        if dist_2d > COLLISION_PROXIMITY_THRESHOLD:
            return None

        # Compute approach rate from recent position history (last 3 frames)
        if len(obj_a.pos_history) >= 3 and len(obj_b.pos_history) >= 3:
            pa_old = obj_a.pos_history[-3]
            pb_old = obj_b.pos_history[-3]
            dist_old = ((pa_old[0] - pb_old[0]) ** 2 + (pa_old[1] - pb_old[1]) ** 2) ** 0.5
            approach_rate = dist_old - dist_2d  # positive = getting closer
        else:
            approach_rate = 0.0

        # Only report if vehicles are actually approaching
        if approach_rate < COLLISION_APPROACH_MIN:
            return None

        # TTC = current distance / approach rate (in frame units, convert to seconds)
        fps = 10.0  # default FPS
        if approach_rate > 0:
            ttc_frames = dist_2d / approach_rate
            return ttc_frames / fps
        return None

    def _check_interactions(self, vehicles: list):
        """Detect vehicle pairs with TTC below safety threshold.

        Uses 2D spatial proximity check before computing TTC.
        Deduplicates by tracking per-pair minimum TTC seen.
        """
        active = [v for v in vehicles if v["id"] in self.objects]
        # Track per-pair collision risks within this frame to deduplicate
        frame_risks = {}
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                vi, vj = active[i], active[j]
                vi_obj = self.objects.get(vi["id"])
                vj_obj = self.objects.get(vj["id"])
                if not vi_obj or not vj_obj:
                    continue
                if vi_obj.total_frames_seen < 5 or vj_obj.total_frames_seen < 5:
                    continue

                # Quick spatial pre-check: skip if 2D distance > threshold
                if vi_obj.pos_history and vj_obj.pos_history:
                    pa = vi_obj.pos_history[-1]
                    pb = vj_obj.pos_history[-1]
                    dist_2d = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
                    if dist_2d > COLLISION_PROXIMITY_THRESHOLD * 1.5:
                        continue

                ttc = self.compute_ttc(vi_obj, vj_obj)
                if ttc is not None and ttc < TTC_SAFETY_THRESHOLD:
                    pair_key = tuple(sorted([vi_obj.obj_id, vj_obj.obj_id]))
                    if pair_key not in frame_risks or ttc < frame_risks[pair_key]["ttc"]:
                        frame_risks[pair_key] = {
                            "ttc": ttc,
                            "vi_obj": vi_obj,
                            "vj_obj": vj_obj,
                        }

        # Only emit deduplicated risks (one per pair with min TTC)
        for pair_key, risk in frame_risks.items():
            vi_obj = risk["vi_obj"]
            vj_obj = risk["vj_obj"]
            ttc = risk["ttc"]
            # Determine severity level
            if ttc < 0.3:
                severity = "Critical"
            elif ttc < 0.8:
                severity = "High"
            else:
                severity = "Moderate"

            self.event_chains.append({
                "time_span": f"{self.current_time:.2f}",
                "facts": [
                    {"node": f"car_{vi_obj.obj_id}", "behavior": vi_obj.fsm_state},
                    {"node": f"car_{vj_obj.obj_id}", "behavior": vj_obj.fsm_state},
                ],
                "interaction": {
                    "event_type": "Collision_Risk",
                    "severity": severity,
                    "trigger_cause": "vehicle_vehicle_proximity",
                    "causal_prediction": {
                        "min_TTC": f"{ttc:.2f}_seconds",
                        "distance_px": f"{dist_2d:.0f}",
                        "counterfactual": (
                            f"if_car_{vi_obj.obj_id}_and_car_{vj_obj.obj_id}"
                            f"_maintain_speed_collision_in_{ttc:.1f}s"
                        ),
                    },
                },
            })

    # ── Collision event aggregation ────────────────────────

    def _aggregate_collision_events(self, max_events: int = 50):
        """Post-process collision events: deduplicate adjacent frames, keep most severe.

        Merges consecutive collision events for the same vehicle pair into single
        entries, retaining the minimum TTC observed. Limits total events to max_events.
        """
        collision_events = [
            e for e in self.event_chains
            if e.get("interaction", {}).get("event_type") == "Collision_Risk"
        ]
        if not collision_events:
            return

        # Group by vehicle pair
        from collections import defaultdict
        pair_groups = defaultdict(list)
        for evt in collision_events:
            nodes = tuple(sorted(
                f.get("node", "?") for f in evt.get("facts", [])
            ))
            pair_groups[nodes].append(evt)

        # For each pair, keep only the event with minimum TTC per severity
        aggregated = []
        for pair, events in pair_groups.items():
            # Sort by severity: Critical > High > Moderate
            severity_order = {"Critical": 0, "High": 1, "Moderate": 2}
            events.sort(key=lambda e: (
                severity_order.get(e.get("interaction", {}).get("severity", "Moderate"), 2),
                float(e.get("interaction", {}).get("causal_prediction", {}).get("min_TTC", "999").replace("_seconds", ""))
            ))
            # Keep the most severe, and up to 3 events per pair (spread across time)
            kept = []
            seen_times = set()
            for evt in events:
                t = evt.get("time_span", "")
                t_key = float(t.split()[0]) if t else 0
                # Keep if at least 0.5s apart from already-kept events
                if not any(abs(t_key - st) < 1.0 for st in seen_times):
                    kept.append(evt)
                    seen_times.add(t_key)
                if len(kept) >= 3:
                    break
            aggregated.extend(kept)

        # Also keep non-collision events
        other_events = [
            e for e in self.event_chains
            if e.get("interaction", {}).get("event_type") != "Collision_Risk"
        ]

        # Sort and limit
        aggregated.sort(key=lambda e: (
            severity_order.get(e.get("interaction", {}).get("severity", "Moderate"), 2),
            float(e.get("interaction", {}).get("causal_prediction", {}).get("min_TTC", "999").replace("_seconds", ""))
        ))
        aggregated = aggregated[:max_events]

        self.event_chains = other_events + aggregated

    # ── ST-Graph generation ─────────────────────────────────

    def generate_st_graph(self) -> dict:
        """Produce the final ST-Graph JSON causal ledger."""
        temporal_events = self._aggregate_fsm_events()

        nodes = {}
        for oid, obj in self.objects.items():
            if obj.total_frames_seen < 2:
                continue
            prefix = "car" if obj.obj_type == "vehicle" else "ped"
            nodes[f"{prefix}_{oid}"] = {
                "type": obj.obj_type,
                "first_seen": f"{obj.first_seen:.2f}s",
                "last_seen": f"{obj.last_seen:.2f}s",
                "total_frames": obj.total_frames_seen,
                "final_state": obj.fsm_state,
                "avg_depth": round(obj.depth_smoothed, 1),
                "turn_semantic": obj.turn_semantic,
                "longitudinal_trend": obj.longitudinal_trend,
                "following": obj.following,
                "stationary_overridden": obj._m2_overridden_by_stationary,
                "n_interpolated_frames": (
                    obj.total_frames_seen - len(obj._raw_pos_history)
                    if obj._raw_pos_history else 0
                ),
                "stationary_segments": [
                    {"start_idx": s, "end_idx": e, "n_frames": e - s + 1}
                    for s, e in obj.stationary_segments
                ],
            }

        self._aggregate_collision_events()
        all_events = temporal_events + self.event_chains
        all_events.sort(key=lambda e: float(e.get("time_span", "0").split()[0]))

        return OrderedDict({
            "meta_info": {
                "scene_environment": self.scene_environment,
                "total_duration_secs": round(self.total_duration, 2),
                "total_frames_processed": self.frame_count,
            },
            "nodes": nodes,
            "temporal_event_chains": all_events,
        })

    def _aggregate_fsm_events(self) -> List[dict]:
        """Build per-object temporal event chains from FSM states."""
        events = []
        for oid, obj in self.objects.items():
            if obj.total_frames_seen < 2:
                continue
            prefix = "car" if obj.obj_type == "vehicle" else "ped"
            events.append({
                "time_span": f"{obj.first_seen:.2f} - {obj.last_seen:.2f}",
                "facts": [{
                    "node": f"{prefix}_{oid}",
                    "behavior": obj.fsm_state,
                    "turn_semantic": obj.turn_semantic,
                    "longitudinal_trend": obj.longitudinal_trend,
                    "following": obj.following,
                    "avg_depth": round(obj.depth_smoothed, 1),
                    "delta_d": round(obj.delta_d_smoothed, 2),
                }],
                "interaction": {
                    "event_type": "Normal_Flow",
                    "trigger_cause": "routine_traffic_monitoring",
                    "causal_prediction": {
                        "min_TTC": "N/A",
                        "counterfactual": "no_conflict_detected",
                    },
                },
            })
        return events

    def get_snapshot(self) -> dict:
        """Lightweight live-state snapshot for debugging."""
        return {
            "frame": self.frame_count,
            "time": f"{self.current_time:.2f}s",
            "active_objects": len([o for o in self.objects.values() if not o.destroyed]),
        }


def run_stage2(pickle_path: str, output_dir: str = "output"):
    """Standalone entry point: load Stage 1 pickle → run 4-phase topology → save ST-Graph JSON.

    Args:
        pickle_path: path to Stage 1 frame_dicts .pkl file
        output_dir: directory for output JSON

    Returns:
        (st_graph, stats_dict) — the ST-Graph JSON and per-phase statistics
    """
    import pickle

    print(f"[STAGE2] Loading {pickle_path}")
    with open(pickle_path, "rb") as f:
        frame_dicts = pickle.load(f)

    engine = TrafficTopologyEngine()
    for fd in frame_dicts:
        engine.update_frame(fd)

    stats = {}

    # Phase 1: dual-pool split
    n_stable, n_suspended = engine.finalize_tracks()

    # Phase 2: LCSS cascade stitching
    phase2_stats = engine.cascade_stitch_tracks()
    stats["phase2"] = phase2_stats

    # Phase 3: trajectory refinement
    phase3_stats = engine.refine_tracks()
    stats["phase3"] = phase3_stats

    # Phase 4: M2 curvature-integral turn semantics
    m2_stats = engine.finalize_m2_semantics()
    stats["phase4"] = m2_stats

    # Generate ST-Graph
    st_graph = engine.generate_st_graph()
    stats["n_nodes"] = len(st_graph.get("nodes", {}))
    stats["n_events"] = len(st_graph.get("temporal_event_chains", []))

    # Save
    import json, os as _os
    _os.makedirs(output_dir, exist_ok=True)
    stem = _os.path.splitext(_os.path.basename(pickle_path))[0]
    json_path = _os.path.join(output_dir, f"st_graph_{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(st_graph, f, indent=2, ensure_ascii=False)
    print(f"[STAGE2] Saved: {json_path}")

    return st_graph, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2: Topology Engine")
    parser.add_argument("pickle", nargs="?", help="Path to Stage 1 .pkl file")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    args = parser.parse_args()

    if args.pickle:
        run_stage2(args.pickle, args.output)
    else:
        # Demo: find all pickles in default output location
        import glob as _glob
        pickles = _glob.glob("output/stage1_result/*.pkl")
        if not pickles:
            print("[STAGE2] No pickle files found. Run Stage 1 first, or specify path.")
            print("[STAGE2] Usage: python topology_engine.py <path/to/frame_dicts.pkl>")
        else:
            for pkl in pickles:
                run_stage2(pkl, args.output)
