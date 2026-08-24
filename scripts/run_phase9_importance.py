#!/usr/bin/env python
"""Launch the forest-model WP4b attribution comparison.

Detachable, like the fitting and evaluation runners:

Start this runner through ``uv`` under ``nohup`` and ``setsid``, redirecting output to a log.

Unlike the fitting run this is not resumable, and does not need to be: it refits nothing. The cost
is 3 scopes x ~15 feature groups = ~45 batched R prediction calls, each carrying `R_PERM = 50`
permuted replicates of one held-out centre's assessment rows.

Reads the three already-fitted OS IECV forest bundles and the published explainability
`importance_summary.csv`; writes `importance_summary.csv` and `importance_agreement.csv` under
the forest-retry output directory. Everything published carries `analysis_status =
post_hoc_exploratory`.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from crc_folfox.explain.contracts import R_PERM
from crc_folfox.forest.importance import IMPORTANCE_ENDPOINT, run_forest_importance_workflow

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "analysis" / "out" / "current"


def stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--endpoint", default=IMPORTANCE_ENDPOINT)
    parser.add_argument("--r-perm", type=int, default=R_PERM, help="smoke tests only")
    parser.add_argument("--scope-limit", type=int, default=None, help="smoke tests only")
    args = parser.parse_args()

    output = args.output or (args.input / "phase9")
    reduced = args.r_perm != R_PERM or args.scope_limit is not None
    print(f"[{stamp()}] phase 9 WP4b attribution comparison starting")
    print(f"[{stamp()}] input={args.input}  output={output}  endpoint={args.endpoint}")
    if reduced:
        print(
            f"[{stamp()}] *** REDUCED INVENTORY -- NOT A REAL RUN *** "
            f"r_perm={args.r_perm} scope_limit={args.scope_limit}"
        )

    try:
        result = run_forest_importance_workflow(
            args.input,
            output,
            endpoint=args.endpoint,
            r_perm=args.r_perm,
            scope_limit=args.scope_limit,
            progress=True,
        )
    except Exception:
        print(f"[{stamp()}] RUN FAILED\n{traceback.format_exc()}")
        return 1

    importance, agreement = result.importance_summary, result.agreement_summary
    print(
        f"[{stamp()}] complete: {len(importance)} importance rows, {len(agreement)} agreement rows"
    )
    if not importance.empty:
        estimable = int((importance["reporting_status"] == "estimable").sum())
        print(f"[{stamp()}] {estimable}/{len(importance)} importance rows estimable")
    print(f"[{stamp()}] artifacts in {result.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
