from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dexhand_retarget.frames import (  # noqa: E402
    CONTRACT_TYPE,
    CONTRACT_VERSION,
    SOURCE_QUEST3,
    parse_frame_payload,
)
from dexhand_retarget.state import SharedState  # noqa: E402


def frame_payload(source: str | None = None) -> dict:
    payload = {
        "type": CONTRACT_TYPE,
        "version": CONTRACT_VERSION,
        "sequence": 7,
        "timestamp": 1.25,
        "hands": [],
    }
    if source is not None:
        payload["source"] = source
    return payload


class FrameSourceTests(unittest.TestCase):
    def test_missing_source_defaults_to_quest3(self) -> None:
        frame = parse_frame_payload(frame_payload())

        self.assertEqual(frame.source, SOURCE_QUEST3)

    def test_quest3_source_is_parsed(self) -> None:
        frame = parse_frame_payload(frame_payload(SOURCE_QUEST3))

        self.assertEqual(frame.source, SOURCE_QUEST3)

    def test_shared_state_records_frame(self) -> None:
        state = SharedState()

        state.record_message(json.dumps(frame_payload()))
        snapshot = state.snapshot()
        self.assertIsNotNone(snapshot["latest_frame"])
        self.assertEqual(snapshot["latest_frame"].source, SOURCE_QUEST3)
        self.assertEqual(snapshot["total_frames"], 1)


if __name__ == "__main__":
    unittest.main()
