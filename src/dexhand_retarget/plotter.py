from __future__ import annotations

import time
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .frames import XR_BONE_EDGES, HandSkeletonFrame
from .state import SharedState


HAND_COLORS = {
    "Left": "#26a6d9",
    "Right": "#f26b3a",
}
AXIS_COLORS = {
    "x": "#e53935",
    "y": "#43a047",
    "z": "#1e88e5",
}
AXIS_JOINTS = ("Wrist", "Palm")
AXIS_LENGTH = 0.055
PLOT_INTERVAL_MS = 20
MAX_BONE_EDGE_COUNT = len(XR_BONE_EDGES)


class SkeletonPlot:
    def __init__(self, state: SharedState, draw_axes: bool) -> None:
        self.state = state
        self.draw_axes = draw_axes
        self.fig = plt.figure("DexHandRetarget Hand Skeleton")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.status_text = self.ax.text2D(0.02, 0.96, "", transform=self.ax.transAxes)
        self.hand_artists: dict[str, dict[str, Any]] = {}
        self.axis_artists: dict[str, dict[tuple[str, str], Any]] = {}
        self.animation: FuncAnimation | None = None

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.view_init(elev=18, azim=-68)

        for handedness, color in HAND_COLORS.items():
            lines = []
            for _ in range(MAX_BONE_EDGE_COUNT):
                (line,) = self.ax.plot([], [], [], color=color, linewidth=2.4, alpha=0.9)
                lines.append(line)
            scatter = self.ax.scatter([], [], [], color=color, s=24, depthshade=True)
            self.hand_artists[handedness] = {
                "lines": lines,
                "scatter": scatter,
            }

            axis_lines: dict[tuple[str, str], Any] = {}
            if self.draw_axes:
                for joint_name in AXIS_JOINTS:
                    for axis_name, axis_color in AXIS_COLORS.items():
                        (axis_line,) = self.ax.plot(
                            [],
                            [],
                            [],
                            color=axis_color,
                            linewidth=1.3,
                            alpha=0.75,
                        )
                        axis_lines[(joint_name, axis_name)] = axis_line
            self.axis_artists[handedness] = axis_lines

        self._set_default_axes()

    def show(self) -> None:
        self.animation = FuncAnimation(
            self.fig,
            self.update,
            interval=PLOT_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()

    def update(self, _frame_index: int) -> Iterable[Any]:
        snapshot = self.state.snapshot()
        frame = snapshot["latest_frame"]
        self._draw_frame(frame)
        self._update_status(snapshot)
        return []

    def _draw_frame(self, frame: HandSkeletonFrame | None) -> None:
        all_points: list[np.ndarray] = []
        hands_by_name = {} if frame is None else frame.hands_by_name
        for handedness in HAND_COLORS:
            hand = hands_by_name.get(handedness)
            artists = self.hand_artists[handedness]
            if not hand or not hand.tracked:
                self._hide_hand(artists, handedness)
                continue

            joints = {
                joint.id: joint
                for joint in hand.joints
                if joint.tracked
            }
            points = []
            for joint in joints.values():
                points.append(np.asarray(joint.position, dtype=float))

            if points:
                point_array = np.vstack(points)
                artists["scatter"]._offsets3d = (
                    point_array[:, 0],
                    point_array[:, 1],
                    point_array[:, 2],
                )
                all_points.extend(points)
            else:
                artists["scatter"]._offsets3d = ([], [], [])

            for line_index, line in enumerate(artists["lines"]):
                if line_index >= len(XR_BONE_EDGES):
                    clear_line(line)
                    continue
                start_id, end_id = XR_BONE_EDGES[line_index]
                start = joints.get(start_id)
                end = joints.get(end_id)
                if start is None or end is None:
                    clear_line(line)
                    continue
                draw_line(line, np.asarray(start.position), np.asarray(end.position))

            self._draw_axes(handedness, joints)

        if all_points:
            self._set_axes_from_points(all_points)
        else:
            self._set_default_axes()

    def _hide_hand(self, artists: dict[str, Any], handedness: str) -> None:
        artists["scatter"]._offsets3d = ([], [], [])
        for line in artists["lines"]:
            clear_line(line)
        for line in self.axis_artists[handedness].values():
            clear_line(line)

    def _draw_axes(self, handedness: str, joints: dict[str, Any]) -> None:
        if not self.draw_axes:
            return

        axis_lines = self.axis_artists[handedness]
        for joint_name in AXIS_JOINTS:
            joint = joints.get(joint_name)
            if joint is None:
                for axis_name in AXIS_COLORS:
                    clear_line(axis_lines[(joint_name, axis_name)])
                continue

            origin = np.asarray(joint.position, dtype=float)
            rotation = np.asarray(joint.rotation, dtype=float)
            basis = quaternion_to_matrix(rotation)
            for axis_index, axis_name in enumerate(("x", "y", "z")):
                end = origin + basis[:, axis_index] * AXIS_LENGTH
                draw_line(axis_lines[(joint_name, axis_name)], origin, end)

    def _update_status(self, snapshot: dict[str, Any]) -> None:
        frame = snapshot["latest_frame"]
        hands = {} if frame is None else {hand.handedness: hand.tracked for hand in frame.hands}
        message_age = time.monotonic() - snapshot["last_message_at"] if snapshot["last_message_at"] else None
        connected = snapshot["connected"] and (message_age is None or message_age < 2.0)
        sequence = "-" if frame is None else frame.sequence
        source = "-" if frame is None else frame.source
        space = "-" if frame is None else frame.space
        left = "tracked" if hands.get("Left") else "lost"
        right = "tracked" if hands.get("Right") else "lost"
        status = (
            f"status: {'connected' if connected else 'waiting'} | "
            f"peer: {snapshot['peer_state']} | channel: {snapshot['channel_state']}\n"
            f"source: {source} | space: {space}\n"
            f"frame: {sequence} | rx: {snapshot['rx_fps']:.1f} fps | "
            f"left: {left} | right: {right}"
        )
        if snapshot["error"]:
            status += f"\nerror: {snapshot['error']}"
        self.status_text.set_text(status)

    def _set_default_axes(self) -> None:
        self.ax.set_xlim(-0.4, 0.4)
        self.ax.set_ylim(0.7, 1.5)
        self.ax.set_zlim(-0.4, 0.4)

    def _set_axes_from_points(self, points: list[np.ndarray]) -> None:
        point_array = np.vstack(points)
        center = point_array.mean(axis=0)
        size = point_array.max(axis=0) - point_array.min(axis=0)
        span = max(float(size.max()) * 1.4, 0.32)
        half = span / 2.0
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    norm = x * x + y * y + z * z + w * w
    if norm <= 1e-12:
        return np.identity(3)
    scale = 2.0 / norm
    xx = x * x * scale
    yy = y * y * scale
    zz = z * z * scale
    xy = x * y * scale
    xz = x * z * scale
    yz = y * z * scale
    wx = w * x * scale
    wy = w * y * scale
    wz = w * z * scale
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def clear_line(line: Any) -> None:
    line.set_data([], [])
    line.set_3d_properties([])


def draw_line(line: Any, start: np.ndarray, end: np.ndarray) -> None:
    line.set_data([start[0], end[0]], [start[1], end[1]])
    line.set_3d_properties([start[2], end[2]])
