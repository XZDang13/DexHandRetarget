from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .dfq_retarget import (
    DfqModelAdapter,
    DfqRetargeter,
    FINGERS,
    NON_THUMB_FINGERS,
    canonical_handedness,
    default_dfq_model_path,
    extract_quest_hand_features,
)
from .frames import HandSkeletonFrame
from .replay import ReplayError, iter_replay_frames


EVAL_REPORT_TYPE = "dfq_replay_eval"
EVAL_REPORT_VERSION = 1
DEFAULT_MAX_LAG_FRAMES = 12


class EvaluationError(ValueError):
    """Raised when a replay evaluation cannot be completed."""


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    replay_path: Path
    hand: str
    model_path: Path | None = None
    ema_alpha: float = 0.45
    max_nfev: int = 25
    stride: int = 1
    max_frames: int | None = None
    include_details: bool = True


class MetricSeries:
    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: float | None) -> None:
        if value is None:
            return
        value = float(value)
        if math.isfinite(value):
            self.values.append(value)

    def extend(self, values: list[float]) -> None:
        for value in values:
            self.add(value)

    def stats(self) -> dict[str, Any]:
        return stats(self.values)


@dataclass(slots=True)
class MethodAccumulator:
    name: str
    thumb_direction_error_deg: MetricSeries
    thumb_normalized_distance_abs_error: MetricSeries
    thumb_bend_abs_error: MetricSeries
    thumb_ctrl_step_norm_per_frame: MetricSeries
    ctrl_step_norm_per_frame: MetricSeries
    ik_loss: MetricSeries
    non_thumb_direction_error_deg: MetricSeries
    all_finger_direction_error_deg: MetricSeries
    finger_direction_error_deg: dict[str, MetricSeries]
    finger_normalized_distance_abs_error: dict[str, MetricSeries]
    finger_bend_abs_error: dict[str, MetricSeries]
    thumb_prior_yaw_abs_error: MetricSeries
    thumb_prior_pitch_abs_error: MetricSeries
    thumb_yaw_output: list[float]
    thumb_pitch_output: list[float]
    target_prior_yaw: list[float]
    target_prior_pitch: list[float]
    previous_ctrl: np.ndarray | None = None

    @classmethod
    def create(cls, name: str) -> MethodAccumulator:
        return cls(
            name=name,
            thumb_direction_error_deg=MetricSeries(),
            thumb_normalized_distance_abs_error=MetricSeries(),
            thumb_bend_abs_error=MetricSeries(),
            thumb_ctrl_step_norm_per_frame=MetricSeries(),
            ctrl_step_norm_per_frame=MetricSeries(),
            ik_loss=MetricSeries(),
            non_thumb_direction_error_deg=MetricSeries(),
            all_finger_direction_error_deg=MetricSeries(),
            finger_direction_error_deg={finger: MetricSeries() for finger in FINGERS},
            finger_normalized_distance_abs_error={finger: MetricSeries() for finger in FINGERS},
            finger_bend_abs_error={finger: MetricSeries() for finger in FINGERS},
            thumb_prior_yaw_abs_error=MetricSeries(),
            thumb_prior_pitch_abs_error=MetricSeries(),
            thumb_yaw_output=[],
            thumb_pitch_output=[],
            target_prior_yaw=[],
            target_prior_pitch=[],
        )

    def add_ctrl_step(self, ctrl: np.ndarray, thumb_indices: tuple[int, int]) -> None:
        if self.previous_ctrl is not None:
            self.ctrl_step_norm_per_frame.add(float(np.linalg.norm(ctrl - self.previous_ctrl)))
            self.thumb_ctrl_step_norm_per_frame.add(
                float(np.linalg.norm(ctrl[list(thumb_indices)] - self.previous_ctrl[list(thumb_indices)]))
            )
        self.previous_ctrl = ctrl.copy()

    def to_payload(self) -> dict[str, Any]:
        return {
            "thumb_direction_error_deg": self.thumb_direction_error_deg.stats(),
            "thumb_normalized_distance_abs_error": self.thumb_normalized_distance_abs_error.stats(),
            "thumb_bend_abs_error": self.thumb_bend_abs_error.stats(),
            "thumb_ctrl_step_norm_per_frame": self.thumb_ctrl_step_norm_per_frame.stats(),
            "ctrl_step_norm_per_frame": self.ctrl_step_norm_per_frame.stats(),
            "ik_loss": self.ik_loss.stats(),
            "non_thumb_direction_error_deg": self.non_thumb_direction_error_deg.stats(),
            "all_finger_direction_error_deg": self.all_finger_direction_error_deg.stats(),
            "per_finger": {
                finger: {
                    "direction_error_deg": self.finger_direction_error_deg[finger].stats(),
                    "normalized_distance_abs_error": self.finger_normalized_distance_abs_error[finger].stats(),
                    "bend_abs_error": self.finger_bend_abs_error[finger].stats(),
                }
                for finger in FINGERS
            },
            "thumb_prior_yaw_abs_error": self.thumb_prior_yaw_abs_error.stats(),
            "thumb_prior_pitch_abs_error": self.thumb_prior_pitch_abs_error.stats(),
            "thumb_yaw_output_range": value_range(self.thumb_yaw_output),
            "thumb_pitch_output_range": value_range(self.thumb_pitch_output),
            "target_prior_yaw_range": value_range(self.target_prior_yaw),
            "target_prior_pitch_range": value_range(self.target_prior_pitch),
            "estimated_yaw_lag": lag_correlation(self.target_prior_yaw, self.thumb_yaw_output),
            "estimated_pitch_lag": lag_correlation(self.target_prior_pitch, self.thumb_pitch_output),
        }


def evaluate_replay(config: EvaluationConfig) -> dict[str, Any]:
    if config.stride <= 0:
        raise EvaluationError("--eval-stride must be greater than 0")
    if config.max_frames is not None and config.max_frames <= 0:
        raise EvaluationError("--eval-max-frames must be greater than 0")

    replay_path = Path(config.replay_path)
    model_path = Path(config.model_path) if config.model_path else default_dfq_model_path(config.hand)
    handedness = canonical_handedness(config.hand)
    frames = _load_frames(replay_path, config.max_frames)
    if not frames:
        raise EvaluationError(f"Replay file is empty: {replay_path}")

    coverage = _coverage_for_frames(frames, handedness)
    input_metrics = _input_metrics(frames, handedness)

    prior_adapter = DfqModelAdapter(model_path, config.hand)
    no_ema_adapter = DfqModelAdapter(model_path, config.hand)
    configured_adapter = DfqModelAdapter(model_path, config.hand)
    no_ema_retargeter = DfqRetargeter(
        no_ema_adapter,
        config.hand,
        ema_alpha=0.0,
        max_nfev=config.max_nfev,
    )
    configured_retargeter = DfqRetargeter(
        configured_adapter,
        config.hand,
        ema_alpha=config.ema_alpha,
        max_nfev=config.max_nfev,
    )

    prior_acc = MethodAccumulator.create("prior_lookup_only")
    no_ema_acc = MethodAccumulator.create("ik_no_ema")
    configured_name = f"ik_configured_ema_{format_alpha(config.ema_alpha)}"
    configured_acc = MethodAccumulator.create(configured_name)
    thumb_distance_projection_abs_gap = MetricSeries()
    finger_distance_projection_abs_gap = {finger: MetricSeries() for finger in FINGERS}
    finger_bend_projection_abs_gap = {finger: MetricSeries() for finger in FINGERS}
    mode_counts = {"waiting": 0, "tracking": 0, "hold": 0}
    details: list[dict[str, Any]] = []
    sampled = 0
    started_at = time.monotonic()

    for source_index, frame in enumerate(frames):
        if source_index % config.stride != 0:
            continue
        sampled += 1
        hand = frame.hands_by_name.get(handedness)
        target = extract_quest_hand_features(hand) if hand is not None else None

        prior_ctrl: np.ndarray | None = None
        retarget_target = None
        if target is not None:
            retarget_target = prior_adapter.retarget_target_features(target)
            thumb_distance_projection_abs_gap.add(
                abs(float(retarget_target.distances["thumb"] - target.distances["thumb"]))
            )
            for finger in FINGERS:
                finger_distance_projection_abs_gap[finger].add(
                    abs(float(retarget_target.distances[finger] - target.distances[finger]))
                )
                finger_bend_projection_abs_gap[finger].add(
                    abs(float(retarget_target.bends[finger] - target.bends[finger]))
                )
            prior_ctrl = prior_adapter.quest_bend_to_ctrl(retarget_target)
            _evaluate_ctrl(
                prior_acc,
                prior_adapter,
                retarget_target,
                prior_ctrl,
                prior_ctrl,
                loss=None,
            )

        no_ema_command = no_ema_retargeter.command_for_frame(frame)
        if no_ema_command.mode == "tracking" and retarget_target is not None and prior_ctrl is not None:
            _evaluate_ctrl(
                no_ema_acc,
                no_ema_adapter,
                retarget_target,
                np.asarray(no_ema_command.ctrl, dtype=float),
                prior_ctrl,
                loss=no_ema_command.loss,
            )

        configured_command = configured_retargeter.command_for_frame(frame)
        mode_counts[configured_command.mode] = mode_counts.get(configured_command.mode, 0) + 1
        configured_ctrl = np.asarray(configured_command.ctrl, dtype=float)
        if configured_command.mode == "tracking" and retarget_target is not None and prior_ctrl is not None:
            detail = _evaluate_ctrl(
                configured_acc,
                configured_adapter,
                retarget_target,
                configured_ctrl,
                prior_ctrl,
                loss=configured_command.loss,
            )
            if config.include_details:
                details.append(
                    _frame_detail(
                        frame,
                        configured_command.mode,
                        target,
                        retarget_target,
                        configured_adapter,
                        configured_ctrl,
                        prior_ctrl,
                        detail,
                    )
                )
        elif config.include_details:
            details.append(
                {
                    "sequence": frame.sequence,
                    "timestamp": frame.timestamp,
                    "mode": configured_command.mode,
                }
            )

    elapsed_seconds = time.monotonic() - started_at
    report = {
        "type": EVAL_REPORT_TYPE,
        "version": EVAL_REPORT_VERSION,
        "replay": {
            "path": str(replay_path),
            "frames_loaded": len(frames),
            "frames_sampled": sampled,
            "sequence_range": [frames[0].sequence, frames[-1].sequence],
            "timestamp_range": [frames[0].timestamp, frames[-1].timestamp],
            "duration_s": frames[-1].timestamp - frames[0].timestamp,
        },
        "method": {
            "hand": handedness,
            "model_path": str(model_path),
            "ema_alpha": config.ema_alpha,
            "max_nfev": config.max_nfev,
            "stride": config.stride,
            "max_frames": config.max_frames,
        },
        "coverage": {
            **coverage,
            "mode_counts": mode_counts,
        },
        "input_metrics": input_metrics,
        "projection_metrics": {
            "thumb_raw_normalized_distance_projection_abs_gap": thumb_distance_projection_abs_gap.stats(),
            "finger_raw_normalized_distance_projection_abs_gap": {
                finger: series.stats() for finger, series in finger_distance_projection_abs_gap.items()
            },
            "finger_raw_bend_projection_abs_gap": {
                finger: series.stats() for finger, series in finger_bend_projection_abs_gap.items()
            },
        },
        "metrics": {
            "prior_lookup_only": prior_acc.to_payload(),
            "ik_no_ema": no_ema_acc.to_payload(),
            configured_acc.name: configured_acc.to_payload(),
        },
        "per_frame": details,
        "eval_seconds": elapsed_seconds,
    }
    return sanitize_for_json(report)


def write_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def print_summary(report: dict[str, Any], stream: TextIO | None = None) -> None:
    stream = stream or sys.stdout
    replay = report["replay"]
    coverage = report["coverage"]
    metrics = report["metrics"]
    configured_name = next(name for name in metrics if name.startswith("ik_configured_ema_"))
    configured = metrics[configured_name]
    prior = metrics["prior_lookup_only"]
    no_ema = metrics["ik_no_ema"]
    projection = report.get("projection_metrics", {})
    thumb_projection_gap = projection.get("thumb_raw_normalized_distance_projection_abs_gap", {})

    print("Replay DFQ retarget evaluation", file=stream)
    print(f"  replay: {replay['path']}", file=stream)
    print(
        f"  frames: loaded={replay['frames_loaded']} sampled={replay['frames_sampled']} "
        f"duration={replay['duration_s']:.2f}s",
        file=stream,
    )
    print(
        f"  valid {coverage['hand']} features: {coverage['valid_feature_frames']} "
        f"({coverage['valid_feature_coverage_pct']:.2f}%)",
        file=stream,
    )
    print(f"  mode counts: {coverage['mode_counts']}", file=stream)
    print(
        "  thumb direction p50/p95: "
        f"prior={_fmt_stat(prior['thumb_direction_error_deg'], 'p50')}/"
        f"{_fmt_stat(prior['thumb_direction_error_deg'], 'p95')} deg, "
        f"no_ema={_fmt_stat(no_ema['thumb_direction_error_deg'], 'p50')}/"
        f"{_fmt_stat(no_ema['thumb_direction_error_deg'], 'p95')} deg, "
        f"configured={_fmt_stat(configured['thumb_direction_error_deg'], 'p50')}/"
        f"{_fmt_stat(configured['thumb_direction_error_deg'], 'p95')} deg",
        file=stream,
    )
    print(
        "  thumb projected distance abs p50/p95: "
        f"{_fmt_stat(configured['thumb_normalized_distance_abs_error'], 'p50')}/"
        f"{_fmt_stat(configured['thumb_normalized_distance_abs_error'], 'p95')}",
        file=stream,
    )
    print(
        "  raw->projected thumb distance gap p50/p95: "
        f"{_fmt_stat(thumb_projection_gap, 'p50')}/"
        f"{_fmt_stat(thumb_projection_gap, 'p95')}",
        file=stream,
    )
    print(
        "  thumb bend abs p50/p95: "
        f"{_fmt_stat(configured['thumb_bend_abs_error'], 'p50')}/"
        f"{_fmt_stat(configured['thumb_bend_abs_error'], 'p95')}",
        file=stream,
    )
    print(
        "  non-thumb direction p50/p95: "
        f"prior={_fmt_stat(prior['non_thumb_direction_error_deg'], 'p50')}/"
        f"{_fmt_stat(prior['non_thumb_direction_error_deg'], 'p95')} deg, "
        f"configured={_fmt_stat(configured['non_thumb_direction_error_deg'], 'p50')}/"
        f"{_fmt_stat(configured['non_thumb_direction_error_deg'], 'p95')} deg",
        file=stream,
    )
    per_finger = configured.get("per_finger", {})
    per_finger_summary = ", ".join(
        f"{finger}={_fmt_stat(per_finger[finger]['direction_error_deg'], 'p50')}/"
        f"{_fmt_stat(per_finger[finger]['direction_error_deg'], 'p95')} deg"
        for finger in NON_THUMB_FINGERS
        if finger in per_finger
    )
    if per_finger_summary:
        print(f"  configured four-finger direction p50/p95: {per_finger_summary}", file=stream)
    yaw_lag = configured["estimated_yaw_lag"]["best_output_lag_frames"]
    pitch_lag = configured["estimated_pitch_lag"]["best_output_lag_frames"]
    print(f"  estimated lag: yaw={yaw_lag} frames pitch={pitch_lag} frames", file=stream)


def _load_frames(path: Path, max_frames: int | None) -> list[HandSkeletonFrame]:
    frames: list[HandSkeletonFrame] = []
    try:
        for frame in iter_replay_frames(path):
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    except ReplayError:
        raise
    except OSError as exc:
        raise EvaluationError(f"Could not read replay: {exc}") from exc
    return frames


def _coverage_for_frames(frames: list[HandSkeletonFrame], handedness: str) -> dict[str, Any]:
    tracked_frames = {"Left": 0, "Right": 0}
    valid_sequences: list[int] = []
    selected_tracked_frames = 0
    for frame in frames:
        for hand in frame.hands:
            tracked_frames[hand.handedness] = tracked_frames.get(hand.handedness, 0) + int(hand.tracked)
        selected = frame.hands_by_name.get(handedness)
        if selected is not None and selected.tracked:
            selected_tracked_frames += 1
        if selected is not None and extract_quest_hand_features(selected) is not None:
            valid_sequences.append(frame.sequence)

    gaps = [b - a for a, b in zip(valid_sequences, valid_sequences[1:])]
    return {
        "hand": handedness,
        "tracked_frames": tracked_frames,
        "selected_tracked_frames": selected_tracked_frames,
        "valid_feature_frames": len(valid_sequences),
        "valid_feature_coverage_pct": percent(len(valid_sequences), len(frames)),
        "first_valid_sequence": valid_sequences[0] if valid_sequences else None,
        "last_valid_sequence": valid_sequences[-1] if valid_sequences else None,
        "valid_sequence_gap_max": max(gaps) if gaps else None,
    }


def _input_metrics(frames: list[HandSkeletonFrame], handedness: str) -> dict[str, Any]:
    thumb_dirs: list[np.ndarray] = []
    thumb_distances: list[float] = []
    thumb_bends: list[float] = []
    previous_direction: np.ndarray | None = None
    direction_steps: list[float] = []

    for frame in frames:
        hand = frame.hands_by_name.get(handedness)
        features = extract_quest_hand_features(hand) if hand is not None else None
        if features is None:
            continue
        direction = np.asarray(features.directions["thumb"], dtype=float)
        thumb_dirs.append(direction)
        thumb_distances.append(float(features.distances["thumb"]))
        thumb_bends.append(float(features.bends["thumb"]))
        if previous_direction is not None:
            direction_steps.append(angle_deg(previous_direction, direction))
        previous_direction = direction

    return {
        "thumb_distance": stats(thumb_distances),
        "thumb_bend_rad": stats(thumb_bends),
        "thumb_direction_step_deg": stats(direction_steps),
    }


def _evaluate_ctrl(
    accumulator: MethodAccumulator,
    adapter: DfqModelAdapter,
    target: Any,
    ctrl: np.ndarray,
    prior_ctrl: np.ndarray,
    *,
    loss: float | None,
) -> dict[str, Any]:
    thumb_indices = (adapter.thumb_yaw_index, adapter.thumb_pitch_index)
    accumulator.add_ctrl_step(ctrl, thumb_indices)
    current = adapter.finger_features_for_ctrl(ctrl)
    thumb_direction_error = angle_deg(current.directions["thumb"], target.directions["thumb"])
    thumb_distance_error = float(current.distances["thumb"] - target.distances["thumb"])
    thumb_bend_error = float(current.bends["thumb"] - target.bends["thumb"])
    accumulator.thumb_direction_error_deg.add(thumb_direction_error)
    accumulator.thumb_normalized_distance_abs_error.add(abs(thumb_distance_error))
    accumulator.thumb_bend_abs_error.add(abs(thumb_bend_error))
    accumulator.ik_loss.add(loss)
    accumulator.thumb_yaw_output.append(float(ctrl[adapter.thumb_yaw_index]))
    accumulator.thumb_pitch_output.append(float(ctrl[adapter.thumb_pitch_index]))
    accumulator.target_prior_yaw.append(float(prior_ctrl[adapter.thumb_yaw_index]))
    accumulator.target_prior_pitch.append(float(prior_ctrl[adapter.thumb_pitch_index]))
    accumulator.thumb_prior_yaw_abs_error.add(abs(float(ctrl[adapter.thumb_yaw_index] - prior_ctrl[adapter.thumb_yaw_index])))
    accumulator.thumb_prior_pitch_abs_error.add(
        abs(float(ctrl[adapter.thumb_pitch_index] - prior_ctrl[adapter.thumb_pitch_index]))
    )

    non_thumb_errors: list[float] = []
    all_errors = [thumb_direction_error]
    per_finger_errors: dict[str, dict[str, float]] = {}
    for finger in FINGERS:
        direction_error = angle_deg(current.directions[finger], target.directions[finger])
        distance_error = float(current.distances[finger] - target.distances[finger])
        bend_error = float(current.bends[finger] - target.bends[finger])
        accumulator.finger_direction_error_deg[finger].add(direction_error)
        accumulator.finger_normalized_distance_abs_error[finger].add(abs(distance_error))
        accumulator.finger_bend_abs_error[finger].add(abs(bend_error))
        per_finger_errors[finger] = {
            "direction_error_deg": direction_error,
            "normalized_distance_error": distance_error,
            "normalized_distance_abs_error": abs(distance_error),
            "bend_error": bend_error,
            "bend_abs_error": abs(bend_error),
        }
        if finger != "thumb":
            non_thumb_errors.append(direction_error)
            all_errors.append(direction_error)
    accumulator.non_thumb_direction_error_deg.extend(non_thumb_errors)
    accumulator.all_finger_direction_error_deg.extend(all_errors)

    return {
        "thumb_direction_error_deg": thumb_direction_error,
        "thumb_normalized_distance_error": thumb_distance_error,
        "thumb_normalized_distance_abs_error": abs(thumb_distance_error),
        "thumb_bend_error": thumb_bend_error,
        "thumb_bend_abs_error": abs(thumb_bend_error),
        "per_finger": per_finger_errors,
        "loss": loss,
    }


def _frame_detail(
    frame: HandSkeletonFrame,
    mode: str,
    raw_target: Any,
    retarget_target: Any,
    adapter: DfqModelAdapter,
    ctrl: np.ndarray,
    prior_ctrl: np.ndarray,
    errors: dict[str, Any],
) -> dict[str, Any]:
    current = adapter.finger_features_for_ctrl(ctrl)
    return {
        "sequence": frame.sequence,
        "timestamp": frame.timestamp,
        "mode": mode,
        "target_thumb": {
            "raw_direction": raw_target.directions["thumb"].tolist(),
            "raw_distance": raw_target.distances["thumb"],
            "raw_bend": raw_target.bends["thumb"],
            "direction": retarget_target.directions["thumb"].tolist(),
            "distance": retarget_target.distances["thumb"],
            "raw_distance_projection_gap": retarget_target.distances["thumb"] - raw_target.distances["thumb"],
            "bend": retarget_target.bends["thumb"],
            "prior_yaw": float(prior_ctrl[adapter.thumb_yaw_index]),
            "prior_pitch": float(prior_ctrl[adapter.thumb_pitch_index]),
        },
        "ctrl": ctrl.tolist(),
        "dfq_thumb": {
            "direction": current.directions["thumb"].tolist(),
            "distance": current.distances["thumb"],
            "bend": current.bends["thumb"],
            "yaw": float(ctrl[adapter.thumb_yaw_index]),
            "pitch": float(ctrl[adapter.thumb_pitch_index]),
        },
        "errors": errors,
    }


def stats(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def lag_correlation(target: list[float], output: list[float], max_lag: int = DEFAULT_MAX_LAG_FRAMES) -> dict[str, Any]:
    target_array = np.asarray(target, dtype=float)
    output_array = np.asarray(output, dtype=float)
    correlations: list[dict[str, Any]] = []
    for lag in range(max_lag + 1):
        if lag == 0:
            current_target = target_array
            current_output = output_array
        else:
            current_target = target_array[:-lag]
            current_output = output_array[lag:]
        corr = _corrcoef(current_target, current_output)
        correlations.append({"lag": lag, "corr": corr})

    valid = [item for item in correlations if item["corr"] is not None]
    if valid:
        best = max(valid, key=lambda item: item["corr"])
        best_lag = best["lag"]
        best_corr = best["corr"]
    else:
        best_lag = None
        best_corr = None
    return {
        "best_output_lag_frames": best_lag,
        "best_corr": best_corr,
        "corr_by_lag": correlations,
    }


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def value_range(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None]
    return [float(min(values)), float(max(values))]


def percent(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total) * 100.0


def format_alpha(value: float) -> str:
    return f"{value:.3g}".replace(".", "_")


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _corrcoef(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 3 or second.size < 3:
        return None
    if float(np.std(first)) <= 1e-8 or float(np.std(second)) <= 1e-8:
        return None
    corr = float(np.corrcoef(first, second)[0, 1])
    return corr if math.isfinite(corr) else None


def _fmt_stat(metric: dict[str, Any], key: str) -> str:
    value = metric.get(key)
    if value is None:
        return "-"
    return f"{float(value):.3g}"
