from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dexhand_retarget.cli import build_parser, main  # noqa: E402
from dexhand_retarget.frames import (  # noqa: E402
    CONTRACT_TYPE,
    CONTRACT_VERSION,
    SOURCE_QUEST3,
    frame_to_payload,
    parse_frame_payload,
)
from dexhand_retarget.replay import (  # noqa: E402
    REPLAY_RECORD_TYPE,
    ReplayPlayer,
    ReplayWriter,
    iter_replay_frames,
    iter_replay_records,
)
from dexhand_retarget.state import SharedState  # noqa: E402


def frame_payload(sequence: int = 7) -> dict:
    return {
        "type": CONTRACT_TYPE,
        "version": CONTRACT_VERSION,
        "sequence": sequence,
        "timestamp": sequence * 0.01,
        "space": "unity_world_meters",
        "source": SOURCE_QUEST3,
        "hands": [],
    }


class ReplayTests(unittest.TestCase):
    def test_replay_writer_round_trips_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "quest.jsonl"
            frame = parse_frame_payload(frame_payload())
            writer = ReplayWriter(path, stream=io.StringIO())

            writer.record_frame(frame)
            writer.close()

            records = list(iter_replay_records(path))
            frames = list(iter_replay_frames(path))
            raw_record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].payload["sequence"], 7)
        self.assertEqual(frames[0], frame)
        self.assertEqual(raw_record["type"], REPLAY_RECORD_TYPE)
        self.assertEqual(raw_record["frame"], frame_to_payload(frame))

    def test_replay_reader_accepts_raw_frame_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw.jsonl"
            path.write_text(json.dumps(frame_payload(sequence=3)) + "\n", encoding="utf-8")

            frames = list(iter_replay_frames(path))

        self.assertEqual(frames[0].sequence, 3)
        self.assertEqual(frames[0].source, SOURCE_QUEST3)

    def test_replay_player_feeds_shared_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw.jsonl"
            path.write_text(
                json.dumps(frame_payload(sequence=1)) + "\n"
                + json.dumps(frame_payload(sequence=2)) + "\n",
                encoding="utf-8",
            )
            state = SharedState()
            player = ReplayPlayer(state, path, log_stream=io.StringIO())

            player.play(threading.Event())

        snapshot = state.snapshot()
        self.assertEqual(snapshot["total_frames"], 2)
        self.assertEqual(snapshot["latest_frame"].sequence, 2)
        self.assertEqual(snapshot["peer_state"], "replay")
        self.assertEqual(snapshot["channel_label"], "replay")
        self.assertEqual(snapshot["channel_state"], "closed")

    def test_replay_cli_args_parse(self) -> None:
        args = build_parser().parse_args(
            [
                "--save-replay",
                "recordings/session.jsonl",
                "--replay-speed",
                "2.0",
                "--replay-fps",
                "90",
            ]
        )

        self.assertEqual(args.save_replay, Path("recordings/session.jsonl"))
        self.assertEqual(args.replay_speed, 2.0)
        self.assertEqual(args.replay_fps, 90)

    def test_headless_replay_validates_file_without_viewer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw.jsonl"
            path.write_text(json.dumps(frame_payload(sequence=4)) + "\n", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                status = main(["--replay", str(path), "--replay-headless"])

        self.assertEqual(status, 0)
        self.assertIn("Parsed 1 frame", stdout.getvalue())

    def test_save_and_playback_are_mutually_exclusive(self) -> None:
        with redirect_stderr(io.StringIO()):
            status = main(
                [
                    "--save-replay",
                    "recordings/session.jsonl",
                    "--replay",
                    "recordings/session.jsonl",
                ]
            )

        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
