from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


CONTRACT_TYPE = "hand_skeleton_frame"
CONTRACT_VERSION = 3
SOURCE_QUEST3 = "quest3"
WORLD_SPACE = "unity_world_meters"

XR_HAND_JOINT_IDS: tuple[str, ...] = (
    "Wrist",
    "Palm",
    "ThumbMetacarpal",
    "ThumbProximal",
    "ThumbDistal",
    "ThumbTip",
    "IndexMetacarpal",
    "IndexProximal",
    "IndexIntermediate",
    "IndexDistal",
    "IndexTip",
    "MiddleMetacarpal",
    "MiddleProximal",
    "MiddleIntermediate",
    "MiddleDistal",
    "MiddleTip",
    "RingMetacarpal",
    "RingProximal",
    "RingIntermediate",
    "RingDistal",
    "RingTip",
    "LittleMetacarpal",
    "LittleProximal",
    "LittleIntermediate",
    "LittleDistal",
    "LittleTip",
)

XR_BONE_EDGES: tuple[tuple[str, str], ...] = (
    ("Wrist", "Palm"),
    ("Wrist", "ThumbMetacarpal"),
    ("ThumbMetacarpal", "ThumbProximal"),
    ("ThumbProximal", "ThumbDistal"),
    ("ThumbDistal", "ThumbTip"),
    ("Wrist", "IndexMetacarpal"),
    ("IndexMetacarpal", "IndexProximal"),
    ("IndexProximal", "IndexIntermediate"),
    ("IndexIntermediate", "IndexDistal"),
    ("IndexDistal", "IndexTip"),
    ("Wrist", "MiddleMetacarpal"),
    ("MiddleMetacarpal", "MiddleProximal"),
    ("MiddleProximal", "MiddleIntermediate"),
    ("MiddleIntermediate", "MiddleDistal"),
    ("MiddleDistal", "MiddleTip"),
    ("Wrist", "RingMetacarpal"),
    ("RingMetacarpal", "RingProximal"),
    ("RingProximal", "RingIntermediate"),
    ("RingIntermediate", "RingDistal"),
    ("RingDistal", "RingTip"),
    ("Wrist", "LittleMetacarpal"),
    ("LittleMetacarpal", "LittleProximal"),
    ("LittleProximal", "LittleIntermediate"),
    ("LittleIntermediate", "LittleDistal"),
    ("LittleDistal", "LittleTip"),
)

BONE_EDGES = XR_BONE_EDGES


class FrameParseError(ValueError):
    """Raised when a hand skeleton frame is malformed."""


@dataclass(frozen=True, slots=True)
class JointPose:
    id: str
    tracked: bool
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class HandSkeleton:
    handedness: str
    tracked: bool
    joints: tuple[JointPose, ...]

    @property
    def joints_by_id(self) -> dict[str, JointPose]:
        return {joint.id: joint for joint in self.joints}


@dataclass(frozen=True, slots=True)
class HandSkeletonFrame:
    sequence: int
    timestamp: float
    space: str
    hands: tuple[HandSkeleton, ...]
    source: str = SOURCE_QUEST3

    @property
    def hands_by_name(self) -> dict[str, HandSkeleton]:
        return {hand.handedness: hand for hand in self.hands}


def frame_to_payload(frame: HandSkeletonFrame) -> dict[str, Any]:
    return {
        "type": CONTRACT_TYPE,
        "version": CONTRACT_VERSION,
        "sequence": frame.sequence,
        "timestamp": frame.timestamp,
        "space": frame.space,
        "source": frame.source,
        "hands": [
            {
                "handedness": hand.handedness,
                "tracked": hand.tracked,
                "joints": [
                    {
                        "id": joint.id,
                        "tracked": joint.tracked,
                        "position": list(joint.position),
                        "rotation": list(joint.rotation),
                    }
                    for joint in hand.joints
                ],
            }
            for hand in frame.hands
        ],
    }


def normalize_source(value: Any, default: str = SOURCE_QUEST3) -> str:
    if value is None:
        return default
    source = str(value).strip().lower()
    return source or default


def parse_message(message: Any) -> HandSkeletonFrame:
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    if isinstance(message, str):
        try:
            payload = json.loads(message)
        except json.JSONDecodeError as exc:
            raise FrameParseError(f"JSON parse error: {exc}") from exc
    elif isinstance(message, Mapping):
        payload = message
    else:
        raise FrameParseError(f"Unsupported message type: {type(message).__name__}")

    return parse_frame_payload(payload)


def parse_frame_payload(payload: Mapping[str, Any]) -> HandSkeletonFrame:
    if payload.get("type") != CONTRACT_TYPE:
        raise FrameParseError(f"Ignoring unknown frame type: {payload.get('type')}")
    if int(payload.get("version", -1)) != CONTRACT_VERSION:
        raise FrameParseError(f"Unsupported frame version: {payload.get('version')}")

    hands_payload = payload.get("hands", [])
    if not isinstance(hands_payload, list):
        raise FrameParseError("Frame 'hands' must be an array")

    return HandSkeletonFrame(
        sequence=int(payload.get("sequence", 0)),
        timestamp=float(payload.get("timestamp", 0.0)),
        space=str(payload.get("space", WORLD_SPACE)),
        hands=tuple(_parse_hand(hand) for hand in hands_payload),
        source=normalize_source(payload.get("source")),
    )


def _parse_hand(payload: Mapping[str, Any]) -> HandSkeleton:
    joints_payload = payload.get("joints", [])
    if not isinstance(joints_payload, list):
        raise FrameParseError("Hand 'joints' must be an array")

    return HandSkeleton(
        handedness=str(payload.get("handedness", "")),
        tracked=bool(payload.get("tracked", False)),
        joints=tuple(_parse_joint(joint) for joint in joints_payload),
    )


def _parse_joint(payload: Mapping[str, Any]) -> JointPose:
    return JointPose(
        id=str(payload.get("id", "")),
        tracked=bool(payload.get("tracked", False)),
        position=_parse_vector3(payload.get("position")),
        rotation=_parse_quaternion(payload.get("rotation")),
    )


def _parse_vector3(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        return (0.0, 0.0, 0.0)
    return (float(value[0]), float(value[1]), float(value[2]))


def _parse_quaternion(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        return (0.0, 0.0, 0.0, 1.0)
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
