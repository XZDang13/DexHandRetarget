from __future__ import annotations

import threading
import time
from typing import Any

from .frames import FrameParseError, HandSkeletonFrame, parse_message


class SharedState:
    """Thread-safe latest-frame state shared by WebRTC and visualization."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latest_frame: HandSkeletonFrame | None = None
        self.connected = False
        self.peer_state = "waiting"
        self.channel_state = "closed"
        self.channel_label = ""
        self.error = ""
        self.last_message_at = 0.0
        self.total_frames = 0
        self.rx_fps = 0.0
        self._fps_window_started_at = time.monotonic()
        self._fps_window_count = 0

    def record_message(self, message: Any) -> HandSkeletonFrame | None:
        try:
            frame = parse_message(message)
        except FrameParseError as exc:
            self.set_error(str(exc))
            return None

        now = time.monotonic()
        with self._lock:
            self.latest_frame = frame
            self.connected = True
            self.channel_state = "open"
            self.last_message_at = now
            self.total_frames += 1
            self._fps_window_count += 1
            elapsed = now - self._fps_window_started_at
            if elapsed >= 1.0:
                self.rx_fps = self._fps_window_count / elapsed
                self._fps_window_count = 0
                self._fps_window_started_at = now
            self.error = ""
        return frame

    def set_peer_state(self, peer_state: str) -> None:
        with self._lock:
            self.peer_state = peer_state
            if peer_state == "connected":
                self.connected = True
            elif peer_state in {"closed", "failed", "disconnected"}:
                self.connected = False

    def set_channel_state(self, label: str, channel_state: str) -> None:
        with self._lock:
            self.channel_label = label
            self.channel_state = channel_state
            self.connected = channel_state == "open"

    def set_error(self, error: str) -> None:
        with self._lock:
            self.error = error

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "latest_frame": self.latest_frame,
                "connected": self.connected,
                "peer_state": self.peer_state,
                "channel_state": self.channel_state,
                "channel_label": self.channel_label,
                "error": self.error,
                "last_message_at": self.last_message_at,
                "total_frames": self.total_frames,
                "rx_fps": self.rx_fps,
            }
