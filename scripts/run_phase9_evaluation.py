#!/usr/bin/env python
"""Score the forest model's `ltrc_cif` against the frozen `minimal_cox` benchmark.

Detached launch, same pattern as the fitting run:

Start this runner through ``uv`` under ``nohup``, redirecting output to a log.
"""

from __future__ import annotations

import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from crc_folfox.forest.evaluation import run_forest_evaluation_workflow

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[{stamp}] phase 9 evaluation starting (OS, 2000 resamples, post_hoc_exploratory)")
    try:
        result = run_forest_evaluation_workflow(
            ROOT / "analysis" / "out" / "current", progress=True
        )
    except Exception:
        print(
            f"[{datetime.now(UTC).isoformat(timespec='seconds')}] FAILED\n{traceback.format_exc()}"
        )
        return 1
    print(
        f"[{datetime.now(UTC).isoformat(timespec='seconds')}] complete: "
        f"{len(result.performance)} performance rows, {len(result.complexity)} complexity rows"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
