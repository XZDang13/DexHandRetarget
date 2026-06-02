from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .frames import HandSkeleton, HandSkeletonFrame
from .state import SharedState


COMMAND_TYPE = "retarget_command"
COMMAND_VERSION = 1
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
NON_THUMB_FINGERS = ("index", "middle", "ring", "pinky")
FINGER_DIRECTION_WEIGHT = 0.35
FINGER_DISTANCE_WEIGHT = 0.35
FINGER_BEND_WEIGHT = 3.0
THUMB_DIRECTION_WEIGHT = 1.35
THUMB_DISTANCE_WEIGHT = 0.2
THUMB_BEND_WEIGHT = 1.4
THUMB_LOOKUP_DIRECTION_WEIGHT = 1.8
THUMB_LOOKUP_DISTANCE_WEIGHT = 0.12
THUMB_LOOKUP_PITCH_WEIGHT = 2.5
CTRL_REGULARIZATION_WEIGHT = 0.03
DEFAULT_MAX_NFEV = 25
DEFAULT_EMA_ALPHA = 0.45
DEFAULT_MAX_CTRL_STEP = 0.16
DEFAULT_RELEASE_MAX_CTRL_STEP = 0.12
DEFAULT_FEATURE_ALPHA = 0.35
DEFAULT_FEATURE_DEADBAND = 0.015
VIEWER_DT = 1.0 / 60.0
THUMB_LOOKUP_YAW_SAMPLES = 31
THUMB_LOOKUP_PITCH_SAMPLES = 41
THUMB_DIRECTION_CALIBRATION = np.diag((1.0, 1.0, -1.0))
THUMB_BEND_TO_PITCH_SCALE = 0.55
NON_THUMB_DIRECTION_CALIBRATION = np.diag((-1.0, 1.0, -1.0))
NON_THUMB_BEND_TO_CTRL_SCALE = 0.75

QUEST_FINGER_JOINTS = {
    "thumb": ("ThumbMetacarpal", "ThumbTip"),
    "index": ("IndexProximal", "IndexTip"),
    "middle": ("MiddleProximal", "MiddleTip"),
    "ring": ("RingProximal", "RingTip"),
    "pinky": ("LittleProximal", "LittleTip"),
}

QUEST_FINGER_BEND_JOINTS = {
    "thumb": ("ThumbMetacarpal", "ThumbProximal", "ThumbTip"),
    "index": ("IndexProximal", "IndexIntermediate", "IndexTip"),
    "middle": ("MiddleProximal", "MiddleIntermediate", "MiddleTip"),
    "ring": ("RingProximal", "RingIntermediate", "RingTip"),
    "pinky": ("LittleProximal", "LittleIntermediate", "LittleTip"),
}

DFQ_FINGER_BASES = {
    "thumb": "thumb_proximal_base",
    "index": "index_proximal",
    "middle": "middle_proximal",
    "ring": "ring_proximal",
    "pinky": "pinky_proximal",
}

DFQ_FINGER_TIPS = {
    "thumb": "thumb_distal_tip",
    "index": "index_intermediate_tip",
    "middle": "middle_intermediate_tip",
    "ring": "ring_intermediate_tip",
    "pinky": "pinky_intermediate_tip",
}

RH56E2_FINGER_BASES = {
    "thumb": "thumb_1",
    "index": "index_1",
    "middle": "middle_1",
    "ring": "ring_1",
    "pinky": "pinky_1",
}

RH56E2_FINGER_TIPS = {
    "thumb": "thumb_tip",
    "index": "index_tip",
    "middle": "middle_tip",
    "ring": "ring_tip",
    "pinky": "pinky_tip",
}

RH56E2_ACTUATOR_JOINTS = (
    "thumb_joint1",
    "thumb_joint2",
    "index_joint",
    "middle_joint",
    "ring_joint",
    "pinky_joint",
)


@dataclass(frozen=True, slots=True)
class FingerFeatureSet:
    directions: dict[str, np.ndarray]
    distances: dict[str, float]
    bends: dict[str, float]


@dataclass(frozen=True, slots=True)
class ActuatorCommand:
    name: str
    joint: str
    value: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "joint": self.joint,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class RetargetCommand:
    sequence: int
    timestamp: float
    hand: str
    backend: str
    model: str
    mode: str
    ctrl: tuple[float, ...]
    actuators: tuple[ActuatorCommand, ...]
    loss: float | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": COMMAND_TYPE,
            "version": COMMAND_VERSION,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "hand": self.hand,
            "backend": self.backend,
            "model": self.model,
            "mode": self.mode,
            "ctrl": list(self.ctrl),
            "actuators": [actuator.to_payload() for actuator in self.actuators],
            "loss": self.loss,
        }


DfqCommand = RetargetCommand


def default_dfq_model_path(hand: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    side = normalize_hand_side(hand)
    return project_root / "assets" / "mjcf" / f"inspire_dfq_{side}" / "model.xml"


def default_rh56e2_model_path(hand: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    side = normalize_hand_side(hand)
    return project_root / "assets" / "mjcf" / f"inspire_rh56e2_{side}" / "model.xml"


def normalize_retarget_model(value: str) -> str:
    model = value.strip().lower()
    if model not in {"dfq", "rh56e2"}:
        raise ValueError(f"Unsupported retarget model: {value}")
    return model


def default_model_path(retarget_model: str, hand: str) -> Path:
    model = normalize_retarget_model(retarget_model)
    if model == "dfq":
        return default_dfq_model_path(hand)
    return default_rh56e2_model_path(hand)


def create_model_adapter(retarget_model: str, model_path: str | Path | None, hand: str):
    model = normalize_retarget_model(retarget_model)
    path = Path(model_path) if model_path else default_model_path(model, hand)
    if model == "dfq":
        return DfqModelAdapter(path, hand)
    return Rh56e2ModelAdapter(path, hand)


def normalize_hand_side(value: str) -> str:
    side = value.strip().lower()
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported hand side: {value}")
    return side


def canonical_handedness(value: str) -> str:
    return "Left" if normalize_hand_side(value) == "left" else "Right"


def extract_quest_hand_features(hand: HandSkeleton) -> FingerFeatureSet | None:
    if not hand.tracked:
        return None

    joints = hand.joints_by_id
    palm = _tracked_joint_position(joints, "Palm")
    wrist = _tracked_joint_position(joints, "Wrist")
    index_base = _tracked_joint_position(joints, "IndexProximal")
    middle_base = _tracked_joint_position(joints, "MiddleProximal")
    little_base = _tracked_joint_position(joints, "LittleProximal")
    if palm is None or wrist is None or index_base is None or middle_base is None or little_base is None:
        return None

    basis = _build_basis(
        lateral=index_base - little_base,
        forward_seed=middle_base - wrist,
    )
    if basis is None:
        return None

    scale = _norm(index_base - little_base)
    if scale <= 1e-8:
        return None

    directions: dict[str, np.ndarray] = {}
    distances: dict[str, float] = {}
    bends: dict[str, float] = {}
    for finger, (base_name, tip_name) in QUEST_FINGER_JOINTS.items():
        base = _tracked_joint_position(joints, base_name)
        tip = _tracked_joint_position(joints, tip_name)
        if base is None or tip is None:
            return None
        vector = tip - base
        direction = _normalize(basis.T @ vector)
        if direction is None:
            return None
        directions[finger] = direction
        distances[finger] = _norm(vector) / scale

    for finger, (base_name, middle_name, tip_name) in QUEST_FINGER_BEND_JOINTS.items():
        base = _tracked_joint_position(joints, base_name)
        middle = _tracked_joint_position(joints, middle_name)
        tip = _tracked_joint_position(joints, tip_name)
        if base is None or middle is None or tip is None:
            return None
        bend = _joint_bend_angle(base, middle, tip)
        if bend is None:
            return None
        bends[finger] = bend

    return FingerFeatureSet(directions=directions, distances=distances, bends=bends)


class DfqModelAdapter:
    def __init__(self, model_path: str | Path, hand: str) -> None:
        self.backend_name = "dfq"
        self.model_path = Path(model_path)
        self.hand_side = normalize_hand_side(hand)
        self.handedness = canonical_handedness(hand)
        self.prefix = "L" if self.hand_side == "left" else "R"
        self.model_name = self.model_path.parent.name
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.actuator_names = tuple(
            _required_name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(self.model.nu)
        )
        self.actuator_joint_names = tuple(
            _required_name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(self.model.actuator_trnid[index, 0]),
            )
            for index in range(self.model.nu)
        )
        self.ctrl_ranges = np.asarray(self.model.actuator_ctrlrange, dtype=float)
        if self.ctrl_ranges.shape != (6, 2):
            raise ValueError(f"Expected 6 DFQ actuators, got {self.model.nu}")
        self.ctrl_lower = self.ctrl_ranges[:, 0]
        self.ctrl_upper = self.ctrl_ranges[:, 1]
        self.open_ctrl = np.clip(np.zeros(self.model.nu, dtype=float), self.ctrl_lower, self.ctrl_upper)
        self.thumb_yaw_index = self.actuator_joint_names.index(f"{self.prefix}_thumb_proximal_yaw_joint")
        self.thumb_pitch_index = self.actuator_joint_names.index(f"{self.prefix}_thumb_proximal_pitch_joint")
        self.finger_ctrl_indices = {
            finger: self.actuator_joint_names.index(f"{self.prefix}_{finger}_proximal_joint")
            for finger in NON_THUMB_FINGERS
        }

        self._joint_ids = {
            name: _required_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.actuator_joint_names
        }
        for suffix in (
            "thumb_intermediate_joint",
            "thumb_distal_joint",
            "index_intermediate_joint",
            "middle_intermediate_joint",
            "ring_intermediate_joint",
            "pinky_intermediate_joint",
        ):
            joint_name = f"{self.prefix}_{suffix}"
            self._joint_ids[joint_name] = _required_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

        self._body_ids = {
            "hand_root": _required_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_root"),
            **{
                finger: _required_id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{self.prefix}_{DFQ_FINGER_BASES[finger]}",
                )
                for finger in FINGERS
            },
        }
        self._site_ids = {
            finger: _required_id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                f"{self.prefix}_{DFQ_FINGER_TIPS[finger]}",
            )
            for finger in FINGERS
        }
        self.apply_ctrl(self.open_ctrl)
        self._thumb_lookup_ctrls, self._thumb_lookup_features = self._build_thumb_lookup()

    def apply_ctrl(self, ctrl: np.ndarray | tuple[float, ...] | list[float]) -> np.ndarray:
        clipped = np.clip(np.asarray(ctrl, dtype=float), self.ctrl_lower, self.ctrl_upper)
        if clipped.shape != (self.model.nu,):
            raise ValueError(f"Expected ctrl shape {(self.model.nu,)}, got {clipped.shape}")

        self.data.ctrl[:] = clipped
        self.data.qpos[:] = 0.0
        for index, joint_name in enumerate(self.actuator_joint_names):
            self._set_joint_qpos(joint_name, clipped[index])

        prefix = self.prefix
        thumb_pitch = clipped[self.actuator_joint_names.index(f"{prefix}_thumb_proximal_pitch_joint")]
        index = clipped[self.actuator_joint_names.index(f"{prefix}_index_proximal_joint")]
        middle = clipped[self.actuator_joint_names.index(f"{prefix}_middle_proximal_joint")]
        ring = clipped[self.actuator_joint_names.index(f"{prefix}_ring_proximal_joint")]
        pinky = clipped[self.actuator_joint_names.index(f"{prefix}_pinky_proximal_joint")]

        self._set_joint_qpos(f"{prefix}_thumb_intermediate_joint", 1.6 * thumb_pitch)
        self._set_joint_qpos(f"{prefix}_thumb_distal_joint", 2.4 * thumb_pitch)
        self._set_joint_qpos(f"{prefix}_index_intermediate_joint", index)
        self._set_joint_qpos(f"{prefix}_middle_intermediate_joint", middle)
        self._set_joint_qpos(f"{prefix}_ring_intermediate_joint", ring)
        self._set_joint_qpos(f"{prefix}_pinky_intermediate_joint", pinky)

        mujoco.mj_forward(self.model, self.data)
        return clipped

    def finger_features_for_ctrl(self, ctrl: np.ndarray) -> FingerFeatureSet:
        self.apply_ctrl(ctrl)
        hand_root = self.data.xpos[self._body_ids["hand_root"]]
        index_base = self.data.xpos[self._body_ids["index"]]
        middle_base = self.data.xpos[self._body_ids["middle"]]
        pinky_base = self.data.xpos[self._body_ids["pinky"]]
        basis = _build_basis(
            lateral=index_base - pinky_base,
            forward_seed=middle_base - hand_root,
        )
        if basis is None:
            raise RuntimeError("DFQ hand frame is degenerate")

        scale = _norm(index_base - pinky_base)
        if scale <= 1e-8:
            raise RuntimeError("DFQ hand scale is degenerate")

        directions: dict[str, np.ndarray] = {}
        distances: dict[str, float] = {}
        for finger in FINGERS:
            base = self.data.xpos[self._body_ids[finger]]
            tip = self.data.site_xpos[self._site_ids[finger]]
            vector = tip - base
            direction = _normalize(basis.T @ vector)
            if direction is None:
                raise RuntimeError(f"DFQ {finger} feature is degenerate")
            directions[finger] = direction
            distances[finger] = _norm(vector) / scale
        bends = {
            "thumb": float(ctrl[self.actuator_joint_names.index(f"{self.prefix}_thumb_proximal_pitch_joint")]),
            "index": float(ctrl[self.actuator_joint_names.index(f"{self.prefix}_index_proximal_joint")]),
            "middle": float(ctrl[self.actuator_joint_names.index(f"{self.prefix}_middle_proximal_joint")]),
            "ring": float(ctrl[self.actuator_joint_names.index(f"{self.prefix}_ring_proximal_joint")]),
            "pinky": float(ctrl[self.actuator_joint_names.index(f"{self.prefix}_pinky_proximal_joint")]),
        }
        return FingerFeatureSet(directions=directions, distances=distances, bends=bends)

    def quest_bend_to_ctrl(self, target: FingerFeatureSet) -> np.ndarray:
        ctrl = self.open_ctrl.copy()
        thumb_ctrl = self.thumb_ctrl_guess(target)
        ctrl[self.thumb_yaw_index] = thumb_ctrl[0]
        ctrl[self.thumb_pitch_index] = thumb_ctrl[1]
        assignments = {
            f"{self.prefix}_index_proximal_joint": target.bends["index"],
            f"{self.prefix}_middle_proximal_joint": target.bends["middle"],
            f"{self.prefix}_ring_proximal_joint": target.bends["ring"],
            f"{self.prefix}_pinky_proximal_joint": target.bends["pinky"],
        }
        for index, joint_name in enumerate(self.actuator_joint_names):
            if joint_name in assignments:
                ctrl[index] = assignments[joint_name]
        return np.clip(ctrl, self.ctrl_lower, self.ctrl_upper)

    def retarget_target_features(self, target: FingerFeatureSet) -> FingerFeatureSet:
        directions = dict(target.directions)
        thumb_direction = _normalize(THUMB_DIRECTION_CALIBRATION @ target.directions["thumb"])
        if thumb_direction is None:
            thumb_direction = target.directions["thumb"]
        directions["thumb"] = thumb_direction
        for finger in NON_THUMB_FINGERS:
            calibrated = _normalize(NON_THUMB_DIRECTION_CALIBRATION @ target.directions[finger])
            if calibrated is not None:
                directions[finger] = calibrated
        bends = dict(target.bends)
        bends["thumb"] = self.thumb_bend_to_pitch(target.bends["thumb"])
        for finger in NON_THUMB_FINGERS:
            bends[finger] = self.non_thumb_bend_to_ctrl(finger, target.bends[finger])
        distances = dict(target.distances)
        distances["thumb"] = self.projected_thumb_distance(
            thumb_direction,
            target.distances["thumb"],
            bends["thumb"],
        )
        distances.update(self.projected_non_thumb_distances(bends))
        return FingerFeatureSet(
            directions=directions,
            distances=distances,
            bends=bends,
        )

    def thumb_ctrl_guess(self, target: FingerFeatureSet) -> tuple[float, float]:
        target_pitch = float(
            np.clip(
                target.bends["thumb"],
                self.ctrl_lower[self.thumb_pitch_index],
                self.ctrl_upper[self.thumb_pitch_index],
            )
        )
        index = self._thumb_lookup_index(
            target.directions["thumb"],
            target.distances["thumb"],
            target_pitch,
        )
        yaw, pitch = self._thumb_lookup_ctrls[index]
        return float(yaw), float(pitch)

    def projected_thumb_distance(self, direction: np.ndarray, distance: float, pitch: float) -> float:
        index = self._thumb_lookup_index(direction, distance, pitch)
        yaw, pitch = self._thumb_lookup_ctrls[index]
        ctrl = self.open_ctrl.copy()
        ctrl[self.thumb_yaw_index] = yaw
        ctrl[self.thumb_pitch_index] = pitch
        features = self.finger_features_for_ctrl(ctrl)
        return float(features.distances["thumb"])

    def _thumb_lookup_index(self, target_direction: np.ndarray, target_distance: float, target_pitch: float) -> int:
        direction_error = self._thumb_lookup_features["directions"] - target_direction
        distance_error = self._thumb_lookup_features["distances"] - float(target_distance)
        pitch_error = self._thumb_lookup_ctrls[:, 1] - target_pitch
        residual = np.column_stack(
            (
                direction_error * THUMB_LOOKUP_DIRECTION_WEIGHT,
                distance_error[:, None] * THUMB_LOOKUP_DISTANCE_WEIGHT,
                pitch_error[:, None] * THUMB_LOOKUP_PITCH_WEIGHT,
            )
        )
        return int(np.argmin(np.einsum("ij,ij->i", residual, residual)))

    def thumb_bend_to_pitch(self, bend: float) -> float:
        return float(
            np.clip(
                bend * THUMB_BEND_TO_PITCH_SCALE,
                self.ctrl_lower[self.thumb_pitch_index],
                self.ctrl_upper[self.thumb_pitch_index],
            )
        )

    def non_thumb_bend_to_ctrl(self, finger: str, bend: float) -> float:
        index = self.finger_ctrl_indices[finger]
        return float(
            np.clip(
                bend * NON_THUMB_BEND_TO_CTRL_SCALE,
                self.ctrl_lower[index],
                self.ctrl_upper[index],
            )
        )

    def projected_non_thumb_distances(self, bends: dict[str, float]) -> dict[str, float]:
        ctrl = self.open_ctrl.copy()
        for finger in NON_THUMB_FINGERS:
            ctrl[self.finger_ctrl_indices[finger]] = bends[finger]
        features = self.finger_features_for_ctrl(ctrl)
        return {finger: float(features.distances[finger]) for finger in NON_THUMB_FINGERS}

    def _build_thumb_lookup(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        yaw_values = np.linspace(
            self.ctrl_lower[self.thumb_yaw_index],
            self.ctrl_upper[self.thumb_yaw_index],
            THUMB_LOOKUP_YAW_SAMPLES,
        )
        pitch_values = np.linspace(
            self.ctrl_lower[self.thumb_pitch_index],
            self.ctrl_upper[self.thumb_pitch_index],
            THUMB_LOOKUP_PITCH_SAMPLES,
        )
        ctrl_pairs: list[tuple[float, float]] = []
        directions: list[np.ndarray] = []
        distances: list[float] = []
        bends: list[float] = []
        for yaw in yaw_values:
            for pitch in pitch_values:
                ctrl = self.open_ctrl.copy()
                ctrl[self.thumb_yaw_index] = yaw
                ctrl[self.thumb_pitch_index] = pitch
                features = self.finger_features_for_ctrl(ctrl)
                ctrl_pairs.append((float(yaw), float(pitch)))
                directions.append(features.directions["thumb"])
                distances.append(features.distances["thumb"])
                bends.append(features.bends["thumb"])
        self.apply_ctrl(self.open_ctrl)
        return (
            np.asarray(ctrl_pairs, dtype=float),
            {
                "directions": np.asarray(directions, dtype=float),
                "distances": np.asarray(distances, dtype=float),
                "bends": np.asarray(bends, dtype=float),
            },
        )

    def actuator_commands(self, ctrl: np.ndarray | tuple[float, ...]) -> tuple[ActuatorCommand, ...]:
        return tuple(
            ActuatorCommand(name=name, joint=joint, value=float(value))
            for name, joint, value in zip(self.actuator_names, self.actuator_joint_names, ctrl)
        )

    def _set_joint_qpos(self, joint_name: str, value: float) -> None:
        joint_id = self._joint_ids[joint_name]
        address = int(self.model.jnt_qposadr[joint_id])
        low, high = self.model.jnt_range[joint_id]
        self.data.qpos[address] = float(np.clip(value, low, high))


class Rh56e2ModelAdapter:
    def __init__(self, model_path: str | Path, hand: str) -> None:
        self.backend_name = "rh56e2"
        self.model_path = Path(model_path)
        self.hand_side = normalize_hand_side(hand)
        self.handedness = canonical_handedness(hand)
        self.model_name = self.model_path.parent.name
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.actuator_names = tuple(
            _required_name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(self.model.nu)
        )
        self.actuator_joint_names = tuple(
            _required_name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(self.model.actuator_trnid[index, 0]),
            )
            for index in range(self.model.nu)
        )
        if self.actuator_joint_names != RH56E2_ACTUATOR_JOINTS:
            raise ValueError(
                "Expected RH56E2 actuator joints "
                f"{RH56E2_ACTUATOR_JOINTS}, got {self.actuator_joint_names}"
            )
        self.ctrl_ranges = np.asarray(self.model.actuator_ctrlrange, dtype=float)
        if self.ctrl_ranges.shape != (6, 2):
            raise ValueError(f"Expected 6 RH56E2 actuators, got {self.model.nu}")
        self.ctrl_lower = self.ctrl_ranges[:, 0]
        self.ctrl_upper = self.ctrl_ranges[:, 1]
        self.open_ctrl = np.clip(np.zeros(self.model.nu, dtype=float), self.ctrl_lower, self.ctrl_upper)
        self.thumb_yaw_index = self.actuator_joint_names.index("thumb_joint1")
        self.thumb_pitch_index = self.actuator_joint_names.index("thumb_joint2")
        self.finger_ctrl_indices = {
            "index": self.actuator_joint_names.index("index_joint"),
            "middle": self.actuator_joint_names.index("middle_joint"),
            "ring": self.actuator_joint_names.index("ring_joint"),
            "pinky": self.actuator_joint_names.index("pinky_joint"),
        }

        mimic_joint_names = (
            "thumb_joint3",
            "thumb_joint4",
            "index_dip",
            "middle_dip",
            "ring_dip",
            "pinky_dip",
        )
        self._joint_ids = {
            name: _required_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in (*self.actuator_joint_names, *mimic_joint_names)
        }
        self._body_ids = {
            "hand_root": _required_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_root"),
            **{
                finger: _required_id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                for finger, body_name in RH56E2_FINGER_BASES.items()
            },
        }
        self._site_ids = {
            finger: _required_id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            for finger, site_name in RH56E2_FINGER_TIPS.items()
        }
        self.apply_ctrl(self.open_ctrl)
        self._thumb_lookup_ctrls, self._thumb_lookup_features = self._build_thumb_lookup()

    def apply_ctrl(self, ctrl: np.ndarray | tuple[float, ...] | list[float]) -> np.ndarray:
        clipped = np.clip(np.asarray(ctrl, dtype=float), self.ctrl_lower, self.ctrl_upper)
        if clipped.shape != (self.model.nu,):
            raise ValueError(f"Expected ctrl shape {(self.model.nu,)}, got {clipped.shape}")

        self.data.ctrl[:] = clipped
        self.data.qpos[:] = 0.0
        for index, joint_name in enumerate(self.actuator_joint_names):
            self._set_joint_qpos(joint_name, clipped[index])

        thumb_pitch = clipped[self.thumb_pitch_index]
        self._set_joint_qpos("thumb_joint3", 0.8024 * thumb_pitch)
        self._set_joint_qpos("thumb_joint4", 0.76123688 * thumb_pitch)
        for finger in NON_THUMB_FINGERS:
            self._set_joint_qpos(f"{finger}_dip", 1.0843 * clipped[self.finger_ctrl_indices[finger]])

        mujoco.mj_forward(self.model, self.data)
        return clipped

    def finger_features_for_ctrl(self, ctrl: np.ndarray) -> FingerFeatureSet:
        self.apply_ctrl(ctrl)
        hand_root = self.data.xpos[self._body_ids["hand_root"]]
        index_base = self.data.xpos[self._body_ids["index"]]
        middle_base = self.data.xpos[self._body_ids["middle"]]
        pinky_base = self.data.xpos[self._body_ids["pinky"]]
        basis = _build_basis(
            lateral=index_base - pinky_base,
            forward_seed=middle_base - hand_root,
        )
        if basis is None:
            raise RuntimeError("RH56E2 hand frame is degenerate")

        scale = _norm(index_base - pinky_base)
        if scale <= 1e-8:
            raise RuntimeError("RH56E2 hand scale is degenerate")

        directions: dict[str, np.ndarray] = {}
        distances: dict[str, float] = {}
        for finger in FINGERS:
            base = self.data.xpos[self._body_ids[finger]]
            tip = self.data.site_xpos[self._site_ids[finger]]
            vector = tip - base
            direction = _normalize(basis.T @ vector)
            if direction is None:
                raise RuntimeError(f"RH56E2 {finger} feature is degenerate")
            directions[finger] = direction
            distances[finger] = _norm(vector) / scale
        bends = {
            "thumb": float(ctrl[self.thumb_pitch_index]),
            "index": float(ctrl[self.finger_ctrl_indices["index"]]),
            "middle": float(ctrl[self.finger_ctrl_indices["middle"]]),
            "ring": float(ctrl[self.finger_ctrl_indices["ring"]]),
            "pinky": float(ctrl[self.finger_ctrl_indices["pinky"]]),
        }
        return FingerFeatureSet(directions=directions, distances=distances, bends=bends)

    def quest_bend_to_ctrl(self, target: FingerFeatureSet) -> np.ndarray:
        ctrl = self.open_ctrl.copy()
        thumb_ctrl = self.thumb_ctrl_guess(target)
        ctrl[self.thumb_yaw_index] = thumb_ctrl[0]
        ctrl[self.thumb_pitch_index] = thumb_ctrl[1]
        for finger in NON_THUMB_FINGERS:
            ctrl[self.finger_ctrl_indices[finger]] = target.bends[finger]
        return np.clip(ctrl, self.ctrl_lower, self.ctrl_upper)

    def retarget_target_features(self, target: FingerFeatureSet) -> FingerFeatureSet:
        directions = dict(target.directions)
        thumb_direction = _normalize(THUMB_DIRECTION_CALIBRATION @ target.directions["thumb"])
        if thumb_direction is None:
            thumb_direction = target.directions["thumb"]
        directions["thumb"] = thumb_direction
        for finger in NON_THUMB_FINGERS:
            calibrated = _normalize(NON_THUMB_DIRECTION_CALIBRATION @ target.directions[finger])
            if calibrated is not None:
                directions[finger] = calibrated
        bends = dict(target.bends)
        bends["thumb"] = self.thumb_bend_to_pitch(target.bends["thumb"])
        for finger in NON_THUMB_FINGERS:
            bends[finger] = self.non_thumb_bend_to_ctrl(finger, target.bends[finger])
        distances = dict(target.distances)
        distances["thumb"] = self.projected_thumb_distance(
            thumb_direction,
            target.distances["thumb"],
            bends["thumb"],
        )
        distances.update(self.projected_non_thumb_distances(bends))
        return FingerFeatureSet(
            directions=directions,
            distances=distances,
            bends=bends,
        )

    def thumb_ctrl_guess(self, target: FingerFeatureSet) -> tuple[float, float]:
        target_pitch = float(
            np.clip(
                target.bends["thumb"],
                self.ctrl_lower[self.thumb_pitch_index],
                self.ctrl_upper[self.thumb_pitch_index],
            )
        )
        index = self._thumb_lookup_index(
            target.directions["thumb"],
            target.distances["thumb"],
            target_pitch,
        )
        yaw, pitch = self._thumb_lookup_ctrls[index]
        return float(yaw), float(pitch)

    def projected_thumb_distance(self, direction: np.ndarray, distance: float, pitch: float) -> float:
        index = self._thumb_lookup_index(direction, distance, pitch)
        yaw, pitch = self._thumb_lookup_ctrls[index]
        ctrl = self.open_ctrl.copy()
        ctrl[self.thumb_yaw_index] = yaw
        ctrl[self.thumb_pitch_index] = pitch
        features = self.finger_features_for_ctrl(ctrl)
        return float(features.distances["thumb"])

    def _thumb_lookup_index(self, target_direction: np.ndarray, target_distance: float, target_pitch: float) -> int:
        direction_error = self._thumb_lookup_features["directions"] - target_direction
        distance_error = self._thumb_lookup_features["distances"] - float(target_distance)
        pitch_error = self._thumb_lookup_ctrls[:, 1] - target_pitch
        residual = np.column_stack(
            (
                direction_error * THUMB_LOOKUP_DIRECTION_WEIGHT,
                distance_error[:, None] * THUMB_LOOKUP_DISTANCE_WEIGHT,
                pitch_error[:, None] * THUMB_LOOKUP_PITCH_WEIGHT,
            )
        )
        return int(np.argmin(np.einsum("ij,ij->i", residual, residual)))

    def thumb_bend_to_pitch(self, bend: float) -> float:
        return float(
            np.clip(
                bend * THUMB_BEND_TO_PITCH_SCALE,
                self.ctrl_lower[self.thumb_pitch_index],
                self.ctrl_upper[self.thumb_pitch_index],
            )
        )

    def non_thumb_bend_to_ctrl(self, finger: str, bend: float) -> float:
        index = self.finger_ctrl_indices[finger]
        return float(
            np.clip(
                bend * NON_THUMB_BEND_TO_CTRL_SCALE,
                self.ctrl_lower[index],
                self.ctrl_upper[index],
            )
        )

    def projected_non_thumb_distances(self, bends: dict[str, float]) -> dict[str, float]:
        ctrl = self.open_ctrl.copy()
        for finger in NON_THUMB_FINGERS:
            ctrl[self.finger_ctrl_indices[finger]] = bends[finger]
        features = self.finger_features_for_ctrl(ctrl)
        return {finger: float(features.distances[finger]) for finger in NON_THUMB_FINGERS}

    def _build_thumb_lookup(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        yaw_values = np.linspace(
            self.ctrl_lower[self.thumb_yaw_index],
            self.ctrl_upper[self.thumb_yaw_index],
            THUMB_LOOKUP_YAW_SAMPLES,
        )
        pitch_values = np.linspace(
            self.ctrl_lower[self.thumb_pitch_index],
            self.ctrl_upper[self.thumb_pitch_index],
            THUMB_LOOKUP_PITCH_SAMPLES,
        )
        ctrl_pairs: list[tuple[float, float]] = []
        directions: list[np.ndarray] = []
        distances: list[float] = []
        bends: list[float] = []
        for yaw in yaw_values:
            for pitch in pitch_values:
                ctrl = self.open_ctrl.copy()
                ctrl[self.thumb_yaw_index] = yaw
                ctrl[self.thumb_pitch_index] = pitch
                features = self.finger_features_for_ctrl(ctrl)
                ctrl_pairs.append((float(yaw), float(pitch)))
                directions.append(features.directions["thumb"])
                distances.append(features.distances["thumb"])
                bends.append(features.bends["thumb"])
        self.apply_ctrl(self.open_ctrl)
        return (
            np.asarray(ctrl_pairs, dtype=float),
            {
                "directions": np.asarray(directions, dtype=float),
                "distances": np.asarray(distances, dtype=float),
                "bends": np.asarray(bends, dtype=float),
            },
        )

    def actuator_commands(self, ctrl: np.ndarray | tuple[float, ...]) -> tuple[ActuatorCommand, ...]:
        return tuple(
            ActuatorCommand(name=name, joint=joint, value=float(value))
            for name, joint, value in zip(self.actuator_names, self.actuator_joint_names, ctrl)
        )

    def _set_joint_qpos(self, joint_name: str, value: float) -> None:
        joint_id = self._joint_ids[joint_name]
        address = int(self.model.jnt_qposadr[joint_id])
        low, high = self.model.jnt_range[joint_id]
        self.data.qpos[address] = float(np.clip(value, low, high))


class Retargeter:
    def __init__(
        self,
        adapter: Any,
        hand: str,
        *,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        max_nfev: int = DEFAULT_MAX_NFEV,
        max_ctrl_step: float | None = DEFAULT_MAX_CTRL_STEP,
        release_max_ctrl_step: float | None = DEFAULT_RELEASE_MAX_CTRL_STEP,
        feature_alpha: float = DEFAULT_FEATURE_ALPHA,
        feature_deadband: float = DEFAULT_FEATURE_DEADBAND,
    ) -> None:
        self.adapter = adapter
        self.hand_side = normalize_hand_side(hand)
        self.handedness = canonical_handedness(hand)
        self.ema_alpha = float(ema_alpha)
        self.max_nfev = int(max_nfev)
        self.max_ctrl_step = None if max_ctrl_step is None or max_ctrl_step <= 0.0 else float(max_ctrl_step)
        self.release_max_ctrl_step = (
            None if release_max_ctrl_step is None or release_max_ctrl_step <= 0.0 else float(release_max_ctrl_step)
        )
        self.feature_alpha = float(np.clip(feature_alpha, 0.0, 0.999))
        self.feature_deadband = max(0.0, float(feature_deadband))
        self._last_smoothed_ctrl: np.ndarray | None = None
        self._last_successful_command: RetargetCommand | None = None
        self._last_filtered_features: FingerFeatureSet | None = None

    @property
    def current_ctrl(self) -> np.ndarray:
        if self._last_smoothed_ctrl is None:
            return self.adapter.open_ctrl.copy()
        return self._last_smoothed_ctrl.copy()

    def command_for_frame(self, frame: HandSkeletonFrame | None) -> RetargetCommand:
        if frame is None:
            return self._waiting_command(sequence=0, timestamp=time.time())

        hand = frame.hands_by_name.get(self.handedness)
        features = extract_quest_hand_features(hand) if hand is not None else None
        if features is None:
            if self._last_successful_command is not None:
                return self._command(
                    sequence=frame.sequence,
                    timestamp=frame.timestamp,
                    mode="hold",
                    ctrl=np.asarray(self._last_successful_command.ctrl, dtype=float),
                    loss=self._last_successful_command.loss,
                )
            return self._waiting_command(sequence=frame.sequence, timestamp=frame.timestamp)
        features = self._filter_features(features)

        raw_ctrl, loss = self._solve(features)
        if self._last_smoothed_ctrl is None:
            smoothed_ctrl = self._limit_ctrl_step(raw_ctrl, self.adapter.open_ctrl)
        else:
            smoothed_ctrl = self.ema_alpha * self._last_smoothed_ctrl + (1.0 - self.ema_alpha) * raw_ctrl
            smoothed_ctrl[self.adapter.thumb_yaw_index] = raw_ctrl[self.adapter.thumb_yaw_index]
            smoothed_ctrl[self.adapter.thumb_pitch_index] = raw_ctrl[self.adapter.thumb_pitch_index]
            smoothed_ctrl = self._limit_ctrl_step(smoothed_ctrl, self._last_smoothed_ctrl)
        smoothed_ctrl = np.clip(smoothed_ctrl, self.adapter.ctrl_lower, self.adapter.ctrl_upper)
        self._last_smoothed_ctrl = smoothed_ctrl
        command = self._command(
            sequence=frame.sequence,
            timestamp=frame.timestamp,
            mode="tracking",
            ctrl=smoothed_ctrl,
            loss=loss,
        )
        self._last_successful_command = command
        return command

    def _solve(self, target: FingerFeatureSet) -> tuple[np.ndarray, float]:
        target = self.adapter.retarget_target_features(target)
        bend_guess = self.adapter.quest_bend_to_ctrl(target)
        warm_start = self.current_ctrl
        if self._last_smoothed_ctrl is None:
            warm_start = bend_guess.copy()
        else:
            warm_start = np.maximum(warm_start, bend_guess * 0.25)
        warm_start = self._nudge_from_active_lower_bounds(warm_start, bend_guess)
        ctrl_range = np.maximum(self.adapter.ctrl_upper - self.adapter.ctrl_lower, 1e-6)

        def residual(ctrl: np.ndarray) -> np.ndarray:
            current = self.adapter.finger_features_for_ctrl(ctrl)
            values: list[float] = []
            for finger in FINGERS:
                direction_weight = THUMB_DIRECTION_WEIGHT if finger == "thumb" else FINGER_DIRECTION_WEIGHT
                distance_weight = THUMB_DISTANCE_WEIGHT if finger == "thumb" else FINGER_DISTANCE_WEIGHT
                bend_weight = THUMB_BEND_WEIGHT if finger == "thumb" else FINGER_BEND_WEIGHT
                values.extend(
                    ((current.directions[finger] - target.directions[finger]) * direction_weight).tolist()
                )
                values.append(
                    (current.distances[finger] - target.distances[finger]) * distance_weight
                )
                values.append((current.bends[finger] - target.bends[finger]) * bend_weight)
            values.extend(((ctrl - warm_start) / ctrl_range * CTRL_REGULARIZATION_WEIGHT).tolist())
            return np.asarray(values, dtype=float)

        result = least_squares(
            residual,
            warm_start,
            bounds=(self.adapter.ctrl_lower, self.adapter.ctrl_upper),
            max_nfev=self.max_nfev,
            ftol=1e-4,
            xtol=1e-4,
            gtol=1e-4,
        )
        ctrl = np.clip(np.asarray(result.x, dtype=float), self.adapter.ctrl_lower, self.adapter.ctrl_upper)
        ctrl[self.adapter.thumb_yaw_index] = bend_guess[self.adapter.thumb_yaw_index]
        ctrl[self.adapter.thumb_pitch_index] = bend_guess[self.adapter.thumb_pitch_index]
        loss = float(np.linalg.norm(residual(ctrl)))
        return ctrl, loss

    def _nudge_from_active_lower_bounds(self, ctrl: np.ndarray, bend_guess: np.ndarray) -> np.ndarray:
        nudged = np.clip(np.asarray(ctrl, dtype=float), self.adapter.ctrl_lower, self.adapter.ctrl_upper)
        spans = np.maximum(self.adapter.ctrl_upper - self.adapter.ctrl_lower, 1e-6)
        for index in range(len(nudged)):
            if bend_guess[index] > self.adapter.ctrl_lower[index] + 1e-5:
                lower_gap = nudged[index] - self.adapter.ctrl_lower[index]
                if lower_gap <= 1e-8:
                    nudged[index] = min(self.adapter.ctrl_upper[index], self.adapter.ctrl_lower[index] + spans[index] * 0.01)
        return nudged

    def _limit_ctrl_step(self, target_ctrl: np.ndarray, previous_ctrl: np.ndarray) -> np.ndarray:
        if self.max_ctrl_step is None:
            return target_ctrl
        lower = np.full_like(target_ctrl, -self.max_ctrl_step)
        upper = np.full_like(target_ctrl, self.max_ctrl_step)
        if self.release_max_ctrl_step is not None:
            release_step = min(self.max_ctrl_step, self.release_max_ctrl_step)
            for finger in NON_THUMB_FINGERS:
                index = self.adapter.finger_ctrl_indices[finger]
                lower[index] = -release_step
        delta = np.clip(target_ctrl - previous_ctrl, lower, upper)
        return previous_ctrl + delta

    def _filter_features(self, features: FingerFeatureSet) -> FingerFeatureSet:
        previous = self._last_filtered_features
        if previous is None:
            self._last_filtered_features = features
            return features

        directions: dict[str, np.ndarray] = {}
        distances: dict[str, float] = {}
        bends: dict[str, float] = {}
        for finger in FINGERS:
            directions[finger] = self._filter_direction(
                previous.directions[finger],
                features.directions[finger],
            )
            distances[finger] = self._filter_scalar(
                previous.distances[finger],
                features.distances[finger],
            )
            bends[finger] = self._filter_scalar(previous.bends[finger], features.bends[finger])

        filtered = FingerFeatureSet(directions=directions, distances=distances, bends=bends)
        self._last_filtered_features = filtered
        return filtered

    def _filter_direction(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        angle = _angle_between_unit_vectors(previous, current)
        if angle <= self.feature_deadband:
            return np.asarray(previous, dtype=float)
        if self.feature_alpha <= 0.0:
            return np.asarray(current, dtype=float)
        filtered = _normalize(self.feature_alpha * previous + (1.0 - self.feature_alpha) * current)
        return np.asarray(current, dtype=float) if filtered is None else filtered

    def _filter_scalar(self, previous: float, current: float) -> float:
        if abs(float(current) - float(previous)) <= self.feature_deadband:
            return float(previous)
        if self.feature_alpha <= 0.0:
            return float(current)
        return float(self.feature_alpha * previous + (1.0 - self.feature_alpha) * current)

    def _waiting_command(self, *, sequence: int, timestamp: float) -> RetargetCommand:
        return self._command(
            sequence=sequence,
            timestamp=timestamp,
            mode="waiting",
            ctrl=self.adapter.open_ctrl,
            loss=None,
        )

    def _command(
        self,
        *,
        sequence: int,
        timestamp: float,
        mode: str,
        ctrl: np.ndarray,
        loss: float | None,
    ) -> RetargetCommand:
        ctrl_tuple = tuple(float(value) for value in ctrl)
        return RetargetCommand(
            sequence=int(sequence),
            timestamp=float(timestamp),
            hand=self.handedness,
            backend=self.adapter.backend_name,
            model=self.adapter.model_name,
            mode=mode,
            ctrl=ctrl_tuple,
            actuators=self.adapter.actuator_commands(ctrl_tuple),
            loss=None if loss is None else float(loss),
        )


class CommandWriter:
    def __init__(self, destination: str, stream: TextIO | None = None) -> None:
        if destination not in {"stdout", "off"}:
            raise ValueError(f"Unsupported command output: {destination}")
        self.destination = destination
        self.stream = stream or sys.stdout

    def write(self, command: RetargetCommand) -> None:
        if self.destination == "off":
            return
        print(json.dumps(command.to_payload(), separators=(",", ":")), file=self.stream, flush=True)


class DfqRetargetRunner:
    def __init__(
        self,
        state: SharedState,
        *,
        retarget_model: str = "dfq",
        hand: str,
        model_path: str | Path | None,
        command_output: str,
        log_stream: TextIO,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        max_nfev: int = DEFAULT_MAX_NFEV,
        max_ctrl_step: float | None = DEFAULT_MAX_CTRL_STEP,
        release_max_ctrl_step: float | None = DEFAULT_RELEASE_MAX_CTRL_STEP,
        feature_alpha: float = DEFAULT_FEATURE_ALPHA,
        feature_deadband: float = DEFAULT_FEATURE_DEADBAND,
        live_log_interval: float = 0.0,
        real_hand=None,
    ) -> None:
        self.state = state
        self.retarget_model = normalize_retarget_model(retarget_model)
        self.model_path = Path(model_path) if model_path else default_model_path(self.retarget_model, hand)
        self.adapter = create_model_adapter(self.retarget_model, self.model_path, hand)
        self.retargeter = Retargeter(
            self.adapter,
            hand,
            ema_alpha=ema_alpha,
            max_nfev=max_nfev,
            max_ctrl_step=max_ctrl_step,
            release_max_ctrl_step=release_max_ctrl_step,
            feature_alpha=feature_alpha,
            feature_deadband=feature_deadband,
        )
        self.writer = CommandWriter(command_output)
        self.log_stream = log_stream
        self.live_log_interval = max(0.0, float(live_log_interval))
        self._last_processed_sequence: int | None = None
        self._last_live_log_at = time.monotonic()
        self._last_command: RetargetCommand | None = None
        self._last_command_ctrl: np.ndarray | None = None
        self._commands_since_log = 0
        self._max_ctrl_step_since_log = 0.0
        self._max_thumb_step_since_log = 0.0
        self._max_actuator_step_since_log = 0.0
        self.real_hand = real_hand

    def show(self) -> None:
        import mujoco.viewer

        print(
            f"{self.retarget_model} retarget: hand={self.retargeter.handedness} model={self.model_path}",
            file=self.log_stream,
            flush=True,
        )
        initial = self.retargeter.command_for_frame(None)
        self.adapter.apply_ctrl(initial.ctrl)
        if self.real_hand is not None:
            self.real_hand.send(initial.ctrl)
        self.writer.write(initial)

        with mujoco.viewer.launch_passive(self.adapter.model, self.adapter.data) as viewer:
            while viewer.is_running():
                snapshot = self.state.snapshot()
                frame = snapshot["latest_frame"]
                command = self._command_for_new_frame(frame)
                if command is not None:
                    with viewer.lock():
                        self.adapter.apply_ctrl(command.ctrl)
                    if self.real_hand is not None:
                        self.real_hand.send(command.ctrl)
                    self.writer.write(command)
                    self._record_live_command(command)
                self._maybe_log_live(snapshot)
                viewer.sync()
                time.sleep(VIEWER_DT)

    def _command_for_new_frame(self, frame: HandSkeletonFrame | None) -> RetargetCommand | None:
        if frame is None:
            return None
        if frame.sequence == self._last_processed_sequence:
            return None
        self._last_processed_sequence = frame.sequence
        return self.retargeter.command_for_frame(frame)

    def _record_live_command(self, command: RetargetCommand) -> None:
        ctrl = np.asarray(command.ctrl, dtype=float)
        if self._last_command_ctrl is not None:
            ctrl_step = float(np.linalg.norm(ctrl - self._last_command_ctrl))
            actuator_step = float(np.max(np.abs(ctrl - self._last_command_ctrl)))
            thumb_indices = [self.adapter.thumb_yaw_index, self.adapter.thumb_pitch_index]
            thumb_step = float(np.linalg.norm(ctrl[thumb_indices] - self._last_command_ctrl[thumb_indices]))
            self._max_ctrl_step_since_log = max(self._max_ctrl_step_since_log, ctrl_step)
            self._max_thumb_step_since_log = max(self._max_thumb_step_since_log, thumb_step)
            self._max_actuator_step_since_log = max(self._max_actuator_step_since_log, actuator_step)
        self._last_command = command
        self._last_command_ctrl = ctrl
        self._commands_since_log += 1

    def _maybe_log_live(self, snapshot: dict[str, Any]) -> None:
        if self.live_log_interval <= 0.0:
            return
        now = time.monotonic()
        if now - self._last_live_log_at < self.live_log_interval:
            return
        frame = snapshot["latest_frame"]
        frame_sequence = "-" if frame is None else frame.sequence
        last_rx_age = "-" if snapshot["last_message_at"] <= 0.0 else f"{now - snapshot['last_message_at']:.2f}s"
        command = self._last_command
        mode = "-" if command is None else command.mode
        loss = "-" if command is None or command.loss is None else f"{command.loss:.3g}"
        ctrl = self.retargeter.current_ctrl
        print(
            f"[{self.retarget_model}-live] "
            f"seq={frame_sequence} mode={mode} "
            f"peer={snapshot['peer_state']} channel={snapshot['channel_state']} "
            f"rx_fps={snapshot['rx_fps']:.1f} last_rx_age={last_rx_age} "
            f"cmds={self._commands_since_log} loss={loss} "
            f"ctrl_step_max={self._max_ctrl_step_since_log:.4f} "
            f"thumb_step_max={self._max_thumb_step_since_log:.4f} "
            f"actuator_step_max={self._max_actuator_step_since_log:.4f} "
            f"ctrl={[round(float(value), 3) for value in ctrl]}",
            file=self.log_stream,
            flush=True,
        )
        self._last_live_log_at = now
        self._commands_since_log = 0
        self._max_ctrl_step_since_log = 0.0
        self._max_thumb_step_since_log = 0.0
        self._max_actuator_step_since_log = 0.0


DfqRetargeter = Retargeter
RetargetRunner = DfqRetargetRunner


def _tracked_joint_position(joints: dict[str, Any], joint_name: str) -> np.ndarray | None:
    joint = joints.get(joint_name)
    if joint is None or not joint.tracked:
        return None
    return np.asarray(joint.position, dtype=float)


def _build_basis(*, lateral: np.ndarray, forward_seed: np.ndarray) -> np.ndarray | None:
    lateral_axis = _normalize(lateral)
    if lateral_axis is None:
        return None
    forward_axis = forward_seed - lateral_axis * float(np.dot(forward_seed, lateral_axis))
    forward_axis = _normalize(forward_axis)
    if forward_axis is None:
        return None
    normal_axis = _normalize(np.cross(lateral_axis, forward_axis))
    if normal_axis is None:
        return None
    forward_axis = _normalize(np.cross(normal_axis, lateral_axis))
    if forward_axis is None:
        return None
    return np.column_stack((lateral_axis, forward_axis, normal_axis))


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    length = _norm(vector)
    if length <= 1e-8:
        return None
    return np.asarray(vector, dtype=float) / length


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _angle_between_unit_vectors(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.arccos(cosine))


def _joint_bend_angle(base: np.ndarray, middle: np.ndarray, tip: np.ndarray) -> float | None:
    first = _normalize(middle - base)
    second = _normalize(tip - middle)
    if first is None or second is None:
        return None
    return _angle_between_unit_vectors(first, second)


def _required_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Missing MuJoCo object {name!r}")
    return int(object_id)


def _required_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    name = mujoco.mj_id2name(model, object_type, index)
    if name is None:
        raise ValueError(f"Missing MuJoCo object name at index {index}")
    return name
