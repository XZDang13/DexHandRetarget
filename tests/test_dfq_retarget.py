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
        retargeter = DfqRetargeter(adapter, "right", ema_alpha=0.0, max_nfev=20)

        command = retargeter.command_for_frame(curled_frame(sequence=7))

        self.assertEqual(command.mode, "tracking")
        ctrl = np.asarray(command.ctrl)
        self.assertGreater(float(ctrl[2:].sum()), 0.25)

    def test_thumb_opposition_drives_thumb_yaw(self) -> None:
        adapter = DfqModelAdapter(default_dfq_model_path("right"), "right")
        retargeter = DfqRetargeter(adapter, "right", ema_alpha=0.0, max_nfev=20)

        command = retargeter.command_for_frame(thumb_opposition_frame(sequence=9, z_offset=0.1))

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
            ["--retarget-dfq", "--hand", "right", "--command-output", "stdout"]
        )

        self.assertTrue(args.retarget_dfq)
        self.assertEqual(args.hand, "right")
        self.assertEqual(args.command_output, "stdout")


if __name__ == "__main__":
    unittest.main()
