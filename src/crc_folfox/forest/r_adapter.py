"""Private, schema-validated bridge to the forest-model-retry workflow delayed-entry
conditional-inference forest.

`LTRCforests::ltrccif` is reached through the same language-neutral pattern model-training and
optimization workflows use:
**R fits and predicts; R never scores.** Every metric, interval, and comparison stays in the
project's single validated Python implementation.

Two risk scales, from one fitted forest
---------------------------------------
`predictProb` returns *entry-conditional* survival, ``S(t | T > entry_i)``, not unconditional
``S(t)``. Because ``entry`` sits in the response and never among the predictors, it cannot affect
terminal-node assignment -- only the conditioning. So both scales come from the same fit:

* ``entry_conditional`` -- predict at each row's own entry;
* ``unconditional`` -- predict at ``entry = 0``, which is vacuous conditioning and reproduces an
  independent ``survival::survfit`` delayed-entry Kaplan-Meier to machine precision.

This mirrors the ``risk_scale`` column the project's performance artifacts already carry. An
adapter returning only one scale would silently compare different estimands against the Cox
models. See `implement.md`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crc_folfox.model_training.contracts import (
    ModelTrainingContractError,
    r_seed,
    validate_survival_inputs,
)

#: The two risk scales every forest-model-retry workflow prediction carries.
RISK_SCALES = ("unconditional", "entry_conditional")

#: Frozen forest size, mirrored in ``r/the conditional-inference-forest artifact
#: directory_adapter.R``. 100 is ``ltrccif``'s own package
#: default, adopted by the dated 2026-08-15 amendment in `forest-retry-spec.md`; model-training
#: workflow's 500 was
#: locked for the different ``ltrcrrf`` algorithm and was computationally infeasible here. Only
#: ``mtry`` and ``minbucket`` are tuned; every other control is fixed before any real fit.
CIF_NTREE = 100


@dataclass(frozen=True)
class RConditionalForestResult:
    """A fitted bundle plus its synthetic-fixture predictions on both risk scales."""

    bundle: Path
    survival_unconditional: np.ndarray
    survival_entry_conditional: np.ndarray
    runtime: dict[str, str]

    def survival(self, scale: str) -> np.ndarray:
        if scale not in RISK_SCALES:
            raise ModelTrainingContractError(f"Unknown risk scale: {scale}")
        return (
            self.survival_unconditional
            if scale == "unconditional"
            else self.survival_entry_conditional
        )


def _as_survival_array(payload: object, *, rows: int, times: int, label: str) -> np.ndarray:
    array = np.asarray(payload, dtype=float)
    # jsonlite collapses a 1-row matrix to a flat vector; restore the declared shape.
    if array.ndim == 1:
        array = array.reshape(rows, times)
    if array.shape != (rows, times):
        raise ModelTrainingContractError(
            f"R forest returned {label} with shape {array.shape}, expected {(rows, times)}"
        )
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise ModelTrainingContractError(f"R forest {label} is not a finite survival in [0, 1]")
    # Survival must be non-increasing in time along each row.
    if times > 1 and np.any(np.diff(array, axis=1) > 1e-9):
        raise ModelTrainingContractError(f"R forest {label} is not monotone non-increasing")
    return array


def sanitize_columns(columns: Sequence[str]) -> list[str]:
    """Map predictor names to R-formula-safe identifiers, deterministically.

    Real preprocessed core-matrix columns carry characters that are illegal in an R formula --
    ``race_ethnicity__AAAPI (Asian, Asian American, and Pacific Islander)``,
    ``race_ethnicity__Hispanic/Latinx``, ``metastatic_presentation__De novo Stage IV``. Pasting
    those into ``Surv(entry, exit, event) ~ ...`` produces invalid R syntax, and no synthetic
    ``x1/x2/x3`` fixture exercises it.

    The mapping depends only on the name and its position among prior collisions, so fit and
    predict produce identical safe names for identical column lists. A mismatch cannot corrupt a
    prediction silently: the fitted forest carries its own formula, so an unexpected name fails
    loudly inside R.
    """
    safe: list[str] = []
    seen: dict[str, int] = {}
    for index, name in enumerate(columns):
        cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in str(name))
        if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
            cleaned = f"v_{cleaned}"
        count = seen.get(cleaned, 0)
        seen[cleaned] = count + 1
        safe.append(cleaned if count == 0 else f"{cleaned}__{count + 1}")
        del index
    return safe


class RConditionalForestAdapter:
    """Invoke the project-owned forest-model-retry workflow R adapter without publishing exchange
    files."""

    def __init__(self, script: str | Path | None = None, rscript: str = "Rscript") -> None:
        self.script = Path(script or Path(__file__).parents[3] / "r" / "phase9" / "cif_adapter.R")
        self.rscript = rscript

    def _run(self, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
        # Invoke by filename with `cwd` set to the script's own directory, not by absolute path:
        # on this platform Rscript's self-reported `--file=` value mangles spaces in an absolute
        # path to `~+~`, which then fails `normalizePath(mustWork = TRUE)`. Regression from
        # Model-training Block C Step 7 -- see r_adapter.py in model_training for the original.
        return subprocess.run(
            [self.rscript, self.script.name, *arguments],
            cwd=self.script.parent,
            capture_output=True,
            text=True,
            check=False,
        )

    def preflight(self) -> dict[str, str]:
        if shutil.which(self.rscript) is None:
            raise ModelTrainingContractError("Required R runtime is unavailable")
        completed = self._run(["--preflight"])
        if completed.returncode != 0:
            raise ModelTrainingContractError("R conditional-forest preflight failed")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ModelTrainingContractError(
                "R conditional-forest preflight returned invalid metadata"
            ) from exc
        required = {"R", "survival", "LTRCforests", "partykit", "jsonlite"}
        if set(payload) != required or not all(isinstance(v, str) for v in payload.values()):
            raise ModelTrainingContractError(
                "R conditional-forest preflight metadata is incomplete"
            )
        return payload

    def fit(
        self,
        matrix: pd.DataFrame,
        metadata: pd.DataFrame,
        *,
        mtry: int,
        minbucket: int,
        seed: int,
        destination: Path,
    ) -> RConditionalForestResult:
        """Fit on the frozen configuration. `bootstrap` is deliberately not exposed."""
        return self._fit_impl(
            matrix,
            metadata,
            mtry=mtry,
            minbucket=minbucket,
            seed=seed,
            destination=destination,
            bootstrap=None,
        )

    def _fit_impl(
        self,
        matrix: pd.DataFrame,
        metadata: pd.DataFrame,
        *,
        mtry: int,
        minbucket: int,
        seed: int,
        destination: Path,
        bootstrap: str | None,
    ) -> RConditionalForestResult:
        values = matrix.to_numpy(float)
        entry = metadata["entry"].to_numpy(float)
        exit_ = metadata["exit"].to_numpy(float)
        event = metadata["event"].to_numpy(int)
        validate_survival_inputs(values, entry, exit_, event)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="crc-phase9-r-") as directory:
            root = Path(directory)
            renamed = matrix.copy()
            renamed.columns = sanitize_columns(matrix.columns)
            exchange = pd.concat(
                [
                    renamed.reset_index(drop=True),
                    metadata.loc[:, ["entry", "exit", "event"]].reset_index(drop=True),
                ],
                axis=1,
            )
            input_path, response_path = root / "fit.csv", root / "response.json"
            exchange.to_csv(input_path, index=False)
            completed = self._run(
                [
                    "--fit",
                    str(input_path),
                    str(destination),
                    str(response_path),
                    str(int(mtry)),
                    str(int(minbucket)),
                    str(r_seed(seed)),
                    *([bootstrap] if bootstrap is not None else []),
                ]
            )
            if completed.returncode != 0 or not response_path.is_file():
                raise ModelTrainingContractError(
                    "R conditional-forest fit failed: "
                    + (completed.stderr or "<no stderr>").strip()[-600:]
                )
            payload = json.loads(response_path.read_text(encoding="utf-8"))
        if not destination.is_file() or payload.get("trees") != CIF_NTREE:
            raise ModelTrainingContractError("R conditional-forest bundle is incomplete")
        unconditional = _as_survival_array(
            payload.get("survival_unconditional"), rows=1, times=2, label="unconditional survival"
        )
        conditional = _as_survival_array(
            payload.get("survival_entry_conditional"),
            rows=1,
            times=2,
            label="entry-conditional survival",
        )
        return RConditionalForestResult(
            bundle=destination,
            survival_unconditional=unconditional,
            survival_entry_conditional=conditional,
            runtime={k: str(v) for k, v in payload["runtime"].items()},
        )

    def _fit_degenerate(
        self,
        matrix: pd.DataFrame,
        metadata: pd.DataFrame,
        *,
        mtry: int,
        seed: int,
        destination: Path,
    ) -> RConditionalForestResult:
        """Fit a deliberately unsplittable forest over the **full** sample.

        Used only by the admission gate's hand-calculated-fixture check (gate 3). Every tree sees
        all training rows (``bootstrap="none"``) and no split can satisfy the terminal-node size,
        so the ensemble must reduce on the unconditional scale to the delayed-entry Kaplan-Meier
        of its training data -- which the gate compares against an independent Python
        implementation.

        Private and never reachable from `fit()`: a real fit must use the frozen
        ``bootstrap="by.root"`` configuration.
        """
        return self._fit_impl(
            matrix,
            metadata,
            mtry=mtry,
            minbucket=len(matrix) * 10,
            seed=seed,
            destination=destination,
            bootstrap="none",
        )

    def predict(
        self,
        bundle: Path,
        matrix: pd.DataFrame,
        metadata: pd.DataFrame,
        *,
        times: Sequence[float],
    ) -> dict[str, np.ndarray]:
        """Absolute survival at `times`, on both risk scales, for every row of `matrix`."""
        if not Path(bundle).is_file():
            raise ModelTrainingContractError("R conditional-forest bundle is missing")
        times = [float(t) for t in times]
        if not times:
            raise ModelTrainingContractError("Prediction time grid is empty")
        with tempfile.TemporaryDirectory(prefix="crc-phase9-r-predict-") as directory:
            root = Path(directory)
            renamed = matrix.copy()
            renamed.columns = sanitize_columns(matrix.columns)
            exchange = pd.concat(
                [
                    renamed.reset_index(drop=True),
                    metadata.loc[:, ["entry", "exit", "event"]].reset_index(drop=True),
                ],
                axis=1,
            )
            newdata_path = root / "newdata.csv"
            times_path = root / "times.csv"
            response_path = root / "response.json"
            exchange.to_csv(newdata_path, index=False)
            pd.DataFrame({"time": times}).to_csv(times_path, index=False)
            completed = self._run(
                ["--predict", str(bundle), str(newdata_path), str(response_path), str(times_path)]
            )
            if completed.returncode != 0 or not response_path.is_file():
                raise ModelTrainingContractError(
                    "R conditional-forest prediction failed: "
                    + (completed.stderr or "<no stderr>").strip()[-600:]
                )
            payload = json.loads(response_path.read_text(encoding="utf-8"))
        rows, n_times = len(matrix), len(times)
        return {
            "unconditional": _as_survival_array(
                payload.get("survival_unconditional"),
                rows=rows,
                times=n_times,
                label="unconditional survival",
            ),
            "entry_conditional": _as_survival_array(
                payload.get("survival_entry_conditional"),
                rows=rows,
                times=n_times,
                label="entry-conditional survival",
            ),
        }


__all__ = [
    "CIF_NTREE",
    "sanitize_columns",
    "RISK_SCALES",
    "RConditionalForestAdapter",
    "RConditionalForestResult",
]
