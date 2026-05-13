#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from dexhand_retarget.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
