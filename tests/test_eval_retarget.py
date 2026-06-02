from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dexhand_retarget.cli import build_parser, main  # noqa: E402
from dexhand_retarget.eval_retarget import (  # noqa: E402
    EVAL_REPORT_TYPE,
    EvaluationConfig,
    angle_deg,
    evaluate_replay,
    lag_correlation,
    print_summary,
    stats,
)
from dexhand_retarget.frames import (  # noqa: E402
    HandSkeleton,
    HandSkeletonFrame,
    JointPose,
    XR_HAND_JOINT_IDS,
    frame_to_payload,
)


def synthetic_frame(
    *,
    hand: str = "Right",
    tracked: bool = True,
    sequence: int = 1,
) -> HandSkeletonFrame:
    base_positions = {
        "Wrist": (0.0, 0.0, 0.0),
        "Palm": (0.0, 0.06, 0.0),
        "ThumbMetacarpal": (0.055, 0.08, -0.015),
        "ThumbProximal": (0.07, 0.105, -0.015),
        "ThumbDistal": (0.085, 0.13, -0.015),
        "ThumbTip": (0.105, 0.155, -0.015 + 0.01 * sequence),
        "IndexMetacarpal": (0.045, 0.085, 0.0),
        "IndexProximal": (0.045, 0.115, 0.0),
        "IndexIntermediate": (0.045, 0.155, 0.0),
        "IndexDistal": (0.045, 0.185, 0.0),
        "IndexTip": (0.045, 0.215, 0.0),
        "MiddleMetacarpal": (0.0, 0.09, 0.0),
        "MiddleProximal": (0.0, 0.12, 0.0),
        "MiddleIntermediate": (0.0, 0.165, 0.0),
        "MiddleDistal": (0.0, 0.2, 0.0),
        "MiddleTip": (0.0, 0.235, 0.0),
        "RingMetacarpal": (-0.03, 0.085, 0.0),
        "RingProximal": (-0.03, 0.115, 0.0),
        "RingIntermediate": (-0.03, 0.155, 0.0),
        "RingDistal": (-0.03, 0.185, 0.0),
        "RingTip": (-0.03, 0.21, 0.0),
        "LittleMetacarpal": (-0.055, 0.08, 0.0),
        "LittleProximal": (-0.055, 0.105, 0.0),
        "LittleIntermediate": (-0.055, 0.14, 0.0),
        "LittleDistal": (-0.055, 0.165, 0.0),
        "LittleTip": (-0.055, 0.19, 0.0),
    }
    joints = tuple(
        JointPose(
            id=joint_id,
            tracked=tracked,
            position=base_positions[joint_id],
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
        for joint_id in XR_HAND_JOINT_IDS
    )
    return HandSkeletonFrame(
        sequence=sequence,
        timestamp=float(sequence) * 0.01,
        space="unity_world_meters",
        hands=(HandSkeleton(handedness=hand, tracked=tracked, joints=joints),),
    )


def write_replay(path: Path, frames: list[HandSkeletonFrame]) -> None:
    records = [
        {
            "type": "dexhand_replay_frame",
            "version": 1,
            "recorded_at": float(index),
            "elapsed": float(index) * 0.01,
            "frame": frame_to_payload(frame),
        }
        for index, frame in enumerate(frames)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


class EvalMetricTests(unittest.TestCase):
    def test_stats_and_angle_metrics_are_stable(self) -> None:
        summary = stats([1.0, 2.0, 3.0])

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["p50"], 2.0)
        self.assertAlmostEqual(angle_deg([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), 90.0)

    def test_lag_correlation_detects_output_delay(self) -> None:
        target = [0.0, 1.0, 2.0, 3.0, 4.0]
        output = [0.0, 0.0, 1.0, 2.0, 3.0]

        lag = lag_correlation(target, output, max_lag=3)

        self.assertEqual(lag["best_output_lag_frames"], 1)
        self.assertAlmostEqual(lag["best_corr"], 1.0)


class EvalReplayTests(unittest.TestCase):
    def test_evaluate_replay_outputs_schema_and_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_path = Path(tmpdir) / "sample.jsonl"
            write_replay(replay_path, [synthetic_frame(sequence=1), synthetic_frame(sequence=2)])

            report = evaluate_replay(
                EvaluationConfig(
                    replay_path=replay_path,
                    hand="right",
                    ema_alpha=0.0,
                    max_nfev=4,
                )
            )

        self.assertEqual(report["type"], EVAL_REPORT_TYPE)
        self.assertEqual(report["replay"]["frames_loaded"], 2)
        self.assertEqual(report["coverage"]["valid_feature_frames"], 2)
        self.assertIn("prior_lookup_only", report["metrics"])
        self.assertIn("ik_no_ema", report["metrics"])
        self.assertIn("ik_configured_ema_0", report["metrics"])
        self.assertIn("projection_metrics", report)
        self.assertIn(
            "thumb_raw_normalized_distance_projection_abs_gap",
            report["projection_metrics"],
        )
        self.assertIn(
            "finger_raw_normalized_distance_projection_abs_gap",
            report["projection_metrics"],
        )
        self.assertEqual(len(report["per_frame"]), 2)
        self.assertIn("target_thumb", report["per_frame"][0])
        self.assertIn("raw_distance_projection_gap", report["per_frame"][0]["target_thumb"])
        self.assertIsNotNone(report["metrics"]["ik_configured_ema_0"]["thumb_direction_error_deg"]["p50"])
        self.assertIn("per_finger", report["metrics"]["ik_configured_ema_0"])
        self.assertIn("index", report["metrics"]["ik_configured_ema_0"]["per_finger"])

    def test_print_summary_mentions_key_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_path = Path(tmpdir) / "sample.jsonl"
            write_replay(replay_path, [synthetic_frame(sequence=1)])
            report = evaluate_replay(
                EvaluationConfig(replay_path=replay_path, hand="right", max_nfev=4)
            )
            stream = io.StringIO()

            print_summary(report, stream=stream)

        self.assertIn("Replay dfq retarget evaluation", stream.getvalue())
        self.assertIn("thumb direction", stream.getvalue())
        self.assertIn("raw->projected thumb distance", stream.getvalue())
        self.assertIn("non-thumb direction", stream.getvalue())

    def test_evaluate_replay_supports_rh56e2_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_path = Path(tmpdir) / "sample.jsonl"
            write_replay(replay_path, [synthetic_frame(sequence=1)])

            report = evaluate_replay(
                EvaluationConfig(
                    replay_path=replay_path,
                    hand="right",
                    retarget_model="rh56e2",
                    ema_alpha=0.0,
                    max_nfev=4,
                )
            )

        self.assertEqual(report["method"]["backend"], "rh56e2")
        self.assertEqual(report["type"], EVAL_REPORT_TYPE)
        self.assertEqual(report["coverage"]["valid_feature_frames"], 1)
        self.assertIn("model_thumb", report["per_frame"][0])

    def test_eval_cli_args_parse(self) -> None:
        args = build_parser().parse_args(
            [
                "--eval-replay",
                "replays/right_hand.jsonl",
                "--eval-output",
                "report.json",
                "--eval-stride",
                "10",
                "--eval-max-frames",
                "100",
            ]
        )

        self.assertEqual(args.eval_replay, Path("replays/right_hand.jsonl"))
        self.assertEqual(args.eval_output, Path("report.json"))
        self.assertEqual(args.eval_stride, 10)
        self.assertEqual(args.eval_max_frames, 100)

    def test_eval_replay_rejects_live_replay_combo(self) -> None:
        with redirect_stderr(io.StringIO()):
            status = main(
                [
                    "--eval-replay",
                    "replays/right_hand.jsonl",
                    "--replay",
                    "replays/right_hand.jsonl",
                ]
            )

        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
