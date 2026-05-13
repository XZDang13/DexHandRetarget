from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .frames import (
    CONTRACT_TYPE,
    FrameParseError,
    HandSkeletonFrame,
    frame_to_payload,
    parse_frame_payload,
)
from .state import SharedState


REPLAY_RECORD_TYPE = "dexhand_replay_frame"
REPLAY_RECORD_VERSION = 1


class ReplayError(ValueError):
    """Raised when a replay file cannot be parsed."""


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    elapsed: float | None
    recorded_at: float | None
    payload: dict[str, Any]


class ReplayWriter:
    def __init__(self, path: str | Path, *, stream: TextIO | None = None) -> None:
        self.path = Path(path)
        self.stream = stream or sys.stdout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self._started_at = time.monotonic()
        self.count = 0
        print(f"[replay] Saving frames to {self.path}", file=self.stream, flush=True)

    def record_frame(self, frame: HandSkeletonFrame) -> None:
        record = {
            "type": REPLAY_RECORD_TYPE,
            "version": REPLAY_RECORD_VERSION,
            "recorded_at": time.time(),
            "elapsed": time.monotonic() - self._started_at,
            "frame": frame_to_payload(frame),
        }
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.count += 1

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
            print(
                f"[replay] Saved {self.count} frame(s) to {self.path}",
                file=self.stream,
                flush=True,
            )

    def __enter__(self) -> ReplayWriter:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def iter_replay_records(path: str | Path) -> Iterator[ReplayRecord]:
    replay_path = Path(path)
    with replay_path.open("r", encoding="utf-8") as replay_file:
        for line_number, line in enumerate(replay_file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield _parse_replay_line(line, line_number=line_number)


def iter_replay_frames(path: str | Path) -> Iterator[HandSkeletonFrame]:
    for record in iter_replay_records(path):
        try:
            yield parse_frame_payload(record.payload)
        except FrameParseError as exc:
            raise ReplayError(f"Invalid frame payload in replay: {exc}") from exc


class ReplayPlayer:
    def __init__(
        self,
        state: SharedState,
        path: str | Path,
        *,
        speed: float = 1.0,
        fps: float | None = None,
        loop: bool = False,
        log_stream: TextIO | None = None,
    ) -> None:
        if speed <= 0:
            raise ReplayError("--replay-speed must be greater than 0")
        if fps is not None and fps <= 0:
            raise ReplayError("--replay-fps must be greater than 0")

        self.state = state
        self.path = Path(path)
        self.speed = float(speed)
        self.fps = None if fps is None else float(fps)
        self.loop = loop
        self.log_stream = log_stream or sys.stdout
        self.records = list(iter_replay_records(self.path))
        if not self.records:
            raise ReplayError(f"Replay file is empty: {self.path}")

    def play(self, stop_event: threading.Event) -> None:
        print(
            f"[replay] Playing {len(self.records)} frame(s) from {self.path}",
            file=self.log_stream,
            flush=True,
        )
        self.state.set_peer_state("replay")
        self.state.set_channel_state("replay", "open")
        try:
            pass_index = 0
            while not stop_event.is_set():
                previous_elapsed: float | None = None
                for index, record in enumerate(self.records):
                    delay = self._delay_for_record(index, previous_elapsed, record.elapsed)
                    if _sleep_until_stopped(delay, stop_event):
                        return
                    frame = self.state.record_message(record.payload)
                    if frame is None:
                        snapshot = self.state.snapshot()
                        raise ReplayError(snapshot["error"] or "Replay frame could not be parsed")
                    previous_elapsed = record.elapsed
                pass_index += 1
                if not self.loop:
                    break
                print(f"[replay] Loop {pass_index} complete", file=self.log_stream, flush=True)
            print("[replay] Finished", file=self.log_stream, flush=True)
        finally:
            self.state.set_channel_state("replay", "closed")

    def _delay_for_record(
        self,
        index: int,
        previous_elapsed: float | None,
        elapsed: float | None,
    ) -> float:
        if index == 0:
            return 0.0
        if self.fps is not None:
            return 1.0 / self.fps
        if elapsed is None or previous_elapsed is None:
            return 0.0
        return max(0.0, (elapsed - previous_elapsed) / self.speed)


def _parse_replay_line(line: str, *, line_number: int) -> ReplayRecord:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ReplayError(f"Replay line {line_number} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ReplayError(f"Replay line {line_number} must be a JSON object")

    payload_type = payload.get("type")
    if payload_type == REPLAY_RECORD_TYPE:
        version = int(payload.get("version", -1))
        if version != REPLAY_RECORD_VERSION:
            raise ReplayError(f"Replay line {line_number} has unsupported version: {version}")
        frame_payload = payload.get("frame")
        if not isinstance(frame_payload, Mapping):
            raise ReplayError(f"Replay line {line_number} missing object field 'frame'")
        return ReplayRecord(
            elapsed=_optional_float(payload.get("elapsed")),
            recorded_at=_optional_float(payload.get("recorded_at")),
            payload=dict(frame_payload),
        )

    if payload_type == CONTRACT_TYPE:
        return ReplayRecord(elapsed=None, recorded_at=None, payload=dict(payload))

    raise ReplayError(f"Replay line {line_number} has unsupported type: {payload_type}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _sleep_until_stopped(delay: float, stop_event: threading.Event) -> bool:
    if delay <= 0:
        return stop_event.is_set()
    deadline = time.monotonic() + delay
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return stop_event.is_set()
        if stop_event.wait(min(remaining, 0.05)):
            return True
