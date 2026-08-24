#!/usr/bin/env python
"""Launch the delayed-entry conditional-inference forest fitting run.

Designed to be started **detached** so it survives the terminal, the session, and this window
closing:

Start this runner through ``uv`` under ``nohup`` and ``setsid``, redirecting output to a log.

The run is resumable by construction. Re-launching with the **same** `--run-id` reads the
per-scope checkpoint and skips every scope already recorded, so an interruption costs one scope
rather than the run. That is not a convenience: at ~40-63 hours this is a multi-day execution, and
an earlier explainability run lost ~38 hours twice to causes outside the code.

OS only, per the dated 2026-08-15 amendment. Everything published carries
`analysis_status = post_hoc_exploratory`.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from crc_folfox.forest.workflow import run_forest_training_workflow

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "analysis" / "out" / "current"


def stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="stable id; reuse it to resume")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--endpoints", default="OS", help="comma-separated; OS only by default")
    parser.add_argument("--scope-limit", type=int, default=None, help="smoke tests only")
    parser.add_argument("--inner-limit", type=int, default=None, help="smoke tests only")
    parser.add_argument("--grid-limit", type=int, default=None, help="smoke tests only")
    args = parser.parse_args()

    output = args.output or (args.input / "phase9")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.run_id}.pid").write_text(str(os.getpid()), encoding="utf-8")

    reduced = any(
        value is not None for value in (args.scope_limit, args.inner_limit, args.grid_limit)
    )
    print(f"[{stamp()}] phase 9 ltrccif run '{args.run_id}' starting (pid {os.getpid()})")
    print(f"[{stamp()}] input={args.input}  output={output}")
    print(f"[{stamp()}] endpoints={args.endpoints}  post_hoc_exploratory")
    if reduced:
        print(
            f"[{stamp()}] *** REDUCED INVENTORY -- NOT A REAL RUN *** "
            f"scope={args.scope_limit} inner={args.inner_limit} grid={args.grid_limit}"
        )
    try:
        result = run_forest_training_workflow(
            args.input,
            output,
            run_id=args.run_id,
            endpoints=tuple(e.strip() for e in args.endpoints.split(",") if e.strip()),
            scope_limit=args.scope_limit,
            inner_limit=args.inner_limit,
            grid_limit=args.grid_limit,
            progress=True,
        )
    except Exception:
        print(f"[{stamp()}] RUN FAILED\n{traceback.format_exc()}")
        return 1

    summary = result.fit_summary
    succeeded = int((summary["fit_status"] == "fit_succeeded").sum())
    print(f"[{stamp()}] complete: {succeeded}/{len(summary)} scopes fit_succeeded")
    print(f"[{stamp()}] artifacts in {result.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
