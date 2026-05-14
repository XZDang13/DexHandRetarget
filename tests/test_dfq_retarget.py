from __future__ import annotations

import io
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dexhand_retarget.cli import build_parser  # noqa: E402
from dexhand_retarget.dfq_retarget import (  # noqa: E402
    CommandWriter,
    DfqModelAdapter,
    DfqRetargeter,
    default_dfq_model_path,
    extract_quest_hand_features,
)
from dexhand_retarget.frames import HandSkeleton, HandSkeletonFrame, JointPose, XR_HAND_JOINT_IDS  # noqa: E402


def synthetic_frame(
    *,
    hand: str = "Right",
    tracked: bool = True,
    sequence: int = 1,
    missing_joint: str | None = None,
) -> HandSkeletonFrame:
    base_positions = {
        "Wrist": (0.0, 0.0, 0.0),
        "Palm": (0.0, 0.06, 0.0),
        "ThumbMetacarpal": (0.055, 0.08, -0.015),
        "ThumbProximal": (0.07, 0.105, -0.015),
        "ThumbDistal": (0.085, 0.13, -0.015),
        "ThumbTip": (0.105, 0.155, -0.015),
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
            tracked=tracked and joint_id != missing_joint,
            position=base_positions[joint_id],
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
        for joint_id in XR_HAND_JOINT_IDS
    )
    return HandSkeletonFrame(
        sequence=sequence,
        timestamp=float(sequence),
        space="unity_world_meters",
        hands=(HandSkeleton(handedness=hand, tracked=tracked, joints=joints),),
    )


def curled_frame(sequence: int = 1, curl: float = 0.08) -> HandSkeletonFrame:
    frame = synthetic_frame(sequence=sequence)
    hand = frame.hands[0]
    curled_tip_ids = {"IndexTip", "MiddleTip", "RingTip", "LittleTip"}
    joints = []
    for joint in hand.joints:
        position = np.asarray(joint.position, dtype=float)
        if joint.id in curled_tip_ids:
            position[1] -= curl
            position[2] -= curl
        joints.append(
            JointPose(
                id=joint.id,
                tracked=joint.tracked,
                position=tuple(float(value) for value in position),
                rotation=joint.rotation,
            )
        )
    return HandSkeletonFrame(
        sequence=frame.sequence,
        timestamp=frame.timestamp,
        space=frame.space,
        hands=(HandSkeleton(handedness=hand.handedness, tracked=hand.tracked, joints=tuple(joints)),),
        source=frame.source,
    )


def thumb_opposition_frame(sequence: int = 1, z_offset: float = 0.1) -> HandSkeletonFrame:
    frame = synthetic_frame(sequence=sequence)
    hand = frame.hands[0]
    joints = []
    for joint in hand.joints:
        position = np.asarray(joint.position, dtype=float)
        if joint.id == "ThumbProximal":
            position[2] += z_offset * 0.35
        elif joint.id == "ThumbDistal":
            position[2] += z_offset * 0.7
        elif joint.id == "ThumbTip":
            position[2] += z_offset
        joints.append(
            JointPose(
                id=joint.id,
                tracked=joint.tracked,
                position=tuple(float(value) for value in position),
                rotation=joint.rotation,
            )
        )
    return HandSkeletonFrame(
        sequence=frame.sequence,
        timestamp=frame.timestamp,
        space=frame.space,
        hands=(HandSkeleton(handedness=hand.handedness, tracked=hand.tracked, joints=tuple(joints)),),
        source=frame.source,
    )


class QuestFeatureTests(unittest.TestCase):
    def test_extracts_five_finger_features(self) -> None:
        hand = synthetic_frame().hands[0]
        features = extract_quest_hand_features(hand)

        self.assertIsNotNone(features)
        assert features is not None
        self.assertEqual(set(features.directions), {"thumb", "index", "middle", "ring", "pinky"})
        self.assertEqual(set(features.distances), {"thumb", "index", "middle", "ring", "pinky"})
        for direction in features.directions.values():
            self.assertTrue(np.isfinite(direction).all())
            self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0)
        for distance in features.distances.values():
            self.assertGreater(distance, 0.0)
        self.assertEqual(set(features.bends), {"thumb", "index", "middle", "ring", "pinky"})

    def test_missing_required_joint_is_invalid(self) -> None:
        hand = synthetic_frame(missing_joint="ThumbTip").hands[0]

        self.assertIsNone(extract_quest_hand_features(hand))


class DfqModelAdapterTests(unittest.TestCase):
    def test_loads_left_and_right_dfq_models(self) -> None:
        for side in ("left", "right"):
            adapter = DfqModelAdapter(default_dfq_model_path(side), side)

            self.assertEqual(adapter.model.nu, 6)
            self.assertEqual(len(adapter.actuator_names), 6)
            self.assertEqual(len(adapter.ctrl_ranges), 6)
            self.assertTrue(np.all(adapter.open_ctrl >= adapter.ctrl_lower))
            self.assertTrue(np.all(adapter.open_ctrl <= adapter.ctrl_upper))
            features = adapter.finger_features_for_ctrl(adapter.open_ctrl)
            self.assertEqual(set(features.directions), {"thumb", "index", "middle", "ring", "pinky"})

    def test_thumb_target_features_are_calibrated(self) -> None:
        adapter = DfqModelAdapter(default_dfq_model_path("right"), "right")
        features = extract_quest_hand_features(synthetic_frame().hands[0])
        assert features is not None

        calibrated = adapter.retarget_target_features(features)

        self.assertAlmostEqual(calibrated.directions["thumb"][0], features.directions["thumb"][0])
        self.assertAlmostEqual(calibrated.directions["thumb"][1], features.directions["thumb"][1])
        self.assertAlmostEqual(calibrated.directions["thumb"][2], -features.directions["thumb"][2])
        self.assertAlmostEqual(calibrated.bends["thumb"], adapter.thumb_bend_to_pitch(features.bends["thumb"]))
        self.assertTrue(math.isfinite(calibrated.distances["thumb"]))
        self.assertGreater(calibrated.distances["thumb"], 0.0)
        for finger in ("index", "middle", "ring", "pinky"):
            self.assertAlmostEqual(calibrated.directions[finger][0], -features.directions[finger][0])
            self.assertAlmostEqual(calibrated.directions[finger][1], features.directions[finger][1])
            self.assertAlmostEqual(calibrated.directions[finger][2], -features.directions[finger][2])
            self.assertAlmostEqual(calibrated.bends[finger], adapter.non_thumb_bend_to_ctrl(finger, features.bends[finger]))
            self.assertTrue(math.isfinite(calibrated.distances[finger]))
            self.assertGreater(calibrated.distances[finger], 0.0)


class DfqRetargeterTests(unittest.TestCase):
    def test_ik_returns_bounded_command(self) -> None:
        adapter = DfqModelAdapter(default_dfq_model_path("right"), "right")
        retargeter = DfqRetargeter(adapter, "right", max_nfev=8)

        command = retargeter.command_for_frame(synthetic_frame(sequence=5))

        self.assertEqual(command.mode, "tracking")
        self.assertEqual(command.sequence, 5)
        self.assertEqual(len(command.ctrl), 6)
        self.assertEqual(len(command.actuators), 6)
        self.assertIsNotNone(command.loss)
        assert command.loss is not None
        self.assertTrue(math.isfinite(command.loss))
        ctrl = np.asarray(command.ctrl)
        self.assertTrue(np.all(ctrl >= adapter.ctrl_lower - 1e-9))
        self.assertTrue(np.all(ctrl <= adapter.ctrl_upper + 1e-9))

    def test_curled_quest_fingers_drive_non_thumb_controls(self) -> None:
        adapter = DfqModelAdapter(default_dfq_model_path("right"), "right")
        retargeter = DfqRetargeter(adapter, "right", ema_alpha=0.0, max_nfev=20, max_ctrl_step=0.0)

        command = retargeter.command_for_frame(curled_frame(sequence=7))

        self.assertEqual(command.mode, "tracking")
        ctrl = np.asarray(command.ctrl)
        self.assertGreater(float(ctrl[2:].sum()), 0.25)

    def test_thumb_opposition_drives_thumb_yaw(self) -> None:
        adapter = DfqModelAdapter(default_dfq_model_path("right"), "right")
        retargeter = DfqRetargeter(adapter, "right", ema_alpha=0.0, max_nfev=20, max_ctrl_step=0.0)

        command = retargeter.command_for_frame(thumb_opposition_frame(sequence=9, z_offset=-0.1))

        self.assertEqual(command.mode, "tracking")
        ctrl = np.asarray(command.ctrl)
        self.assertGreater(float(ctrl[0]), 0.35)

    def test_hold_keeps_last_successful_command(self) -> None:
        adapter = DfqModelAdapter(default_dfq_model_path("right"), "right")
        retargeter = DfqRetargeter(adapter, "right", max_nfev=8)
        tracking = retargeter.command_for_frame(synthetic_frame(sequence=1))

        hold = retargeter.command_for_frame(synthetic_frame(sequence=2, tracked=False))

        self.assertEqual(hold.mode, "hold")
        self.assertEqual(hold.sequence, 2)
        self.assertEqual(hold.ctrl, tracking.ctrl)

    def test_waiting_uses_open_pose_before_first_tracking_frame(self) -> None:
        adapter = DfqModelAdapter(default_dfq_model_path("right"), "right")
        retargeter = DfqRetargeter(adapter, "right", max_nfev=8)

        waiting = retargeter.command_for_frame(synthetic_frame(sequence=1, tracked=False))

        self.assertEqual(waiting.mode, "waiting")
        self.assertEqual(waiting.ctrl, tuple(float(value) for value in adapter.open_ctrl))

    def test_rate_limit_caps_per_actuator_step(self) -> None:
        adapter = DfqModelAdapter(default_dfq_model_path("right"), "right")
        retargeter = DfqRetargeter(adapter, "right", ema_alpha=0.0, max_nfev=8, max_ctrl_step=0.05)
        first = retargeter.command_for_frame(synthetic_frame(sequence=1))

        second = retargeter.command_for_frame(curled_frame(sequence=2, curl=0.12))

        step = np.abs(np.asarray(second.ctrl) - np.asarray(first.ctrl))
        self.assertTrue(np.all(step <= 0.0500001))

    def test_rate_limit_caps_first_tracking_step_from_open_pose(self) -> None:
        adapter = DfqModelAdapter(default_dfq_model_path("right"), "right")
        retargeter = DfqRetargeter(adapter, "right", ema_alpha=0.0, max_nfev=8, max_ctrl_step=0.05)

        command = retargeter.command_for_frame(curled_frame(sequence=1, curl=0.12))

        step = np.abs(np.asarray(command.ctrl) - adapter.open_ctrl)
        self.assertTrue(np.all(step <= 0.0500001))

    def test_feature_deadband_holds_small_input_jitter(self) -> None:
        adapter = DfqModelAdapter(default_dfq_model_path("right"), "right")
        retargeter = DfqRetargeter(
            adapter,
            "right",
            ema_alpha=0.0,
            max_nfev=8,
            max_ctrl_step=0.0,
            feature_alpha=0.0,
            feature_deadband=0.05,
        )
        first = retargeter.command_for_frame(synthetic_frame(sequence=1))

        second = retargeter.command_for_frame(curled_frame(sequence=2, curl=0.002))

        np.testing.assert_allclose(second.ctrl, first.ctrl, atol=1e-9)

    def test_release_limit_slows_non_thumb_opening(self) -> None:
        adapter = DfqModelAdapter(default_dfq_model_path("right"), "right")
        retargeter = DfqRetargeter(
            adapter,
            "right",
            ema_alpha=0.0,
            max_nfev=8,
            max_ctrl_step=10.0,
            release_max_ctrl_step=0.02,
            feature_alpha=0.0,
            feature_deadband=0.0,
        )
        closed = retargeter.command_for_frame(curled_frame(sequence=1, curl=0.12))

        opening = retargeter.command_for_frame(synthetic_frame(sequence=2))

        delta = np.asarray(opening.ctrl) - np.asarray(closed.ctrl)
        self.assertTrue(np.all(delta[2:] >= -0.0200001))


class CommandWriterTests(unittest.TestCase):
    def test_stdout_jsonl_schema_is_stable(self) -> None:
        adapter = DfqModelAdapter(default_dfq_model_path("right"), "right")
        retargeter = DfqRetargeter(adapter, "right", max_nfev=8)
        command = retargeter.command_for_frame(synthetic_frame(sequence=3))
        stream = io.StringIO()

        CommandWriter("stdout", stream).write(command)

        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["type"], "dfq_command")
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["mode"], "tracking")
        self.assertEqual(payload["hand"], "Right")
        self.assertEqual(len(payload["ctrl"]), 6)
        self.assertEqual([item["name"] for item in payload["actuators"]], list(adapter.actuator_names))


class CliParserTests(unittest.TestCase):
    def test_retarget_dfq_args_parse(self) -> None:
        args = build_parser().parse_args(
            [
                "--retarget-dfq",
                "--hand",
                "right",
                "--command-output",
                "stdout",
                "--live-log-interval",
                "1.5",
                "--retarget-max-step",
                "0.12",
                "--retarget-release-max-step",
                "0.06",
                "--retarget-feature-alpha",
                "0.4",
                "--retarget-feature-deadband",
                "0.02",
            ]
        )

        self.assertTrue(args.retarget_dfq)
        self.assertEqual(args.hand, "right")
        self.assertEqual(args.command_output, "stdout")
        self.assertEqual(args.live_log_interval, 1.5)
        self.assertEqual(args.retarget_max_step, 0.12)
        self.assertEqual(args.retarget_release_max_step, 0.06)
        self.assertEqual(args.retarget_feature_alpha, 0.4)
        self.assertEqual(args.retarget_feature_deadband, 0.02)


if __name__ == "__main__":
    unittest.main()
