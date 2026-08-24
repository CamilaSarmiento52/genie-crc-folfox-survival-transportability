"""Private, schema-validated bridge to the required R LTRC forest runtime."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crc_folfox.model_training.contracts import (
    ModelTrainingContractError,
    r_seed,
    validate_survival_inputs,
)


def _stderr_excerpt(text: str | None, *, limit: int = 600) -> str:
    """Bound R's stderr for an error message while keeping **both** ends.

    A tail-only excerpt is the wrong choice for the failure this exists to capture: R prints
    ``*** caught segfault *** address 0x1, cause 'invalid permissions'`` at the **top** of stderr
    and the traceback below it, so keeping only the last N characters discards the signature and
    retains the least informative part.
    """
    cleaned = (text or "<no stderr>").strip()
    if len(cleaned) <= 2 * limit:
        return cleaned
    elided = len(cleaned) - 2 * limit
    return f"{cleaned[:limit]}\n...[{elided} characters elided]...\n{cleaned[-limit:]}"


@dataclass(frozen=True)
class RForestResult:
    bundle: Path
    survival: np.ndarray
    runtime: dict[str, str]


class RForestAdapter:
    """Invoke the project-owned R adapter without publishing exchange files."""

    def __init__(self, script: str | Path | None = None, rscript: str = "Rscript") -> None:
        self.script = Path(
            script or Path(__file__).parents[3] / "r" / "phase4" / "forest_adapter.R"
        )
        self.rscript = rscript

    def preflight(self) -> dict[str, str]:
        if shutil.which(self.rscript) is None:
            raise ModelTrainingContractError("Required R runtime is unavailable")
        # Invoke by filename with `cwd` set to the script's own directory, not by absolute path:
        # on this platform, Rscript's self-reported `--file=` value (used only for renv
        # activation) mangles spaces in an absolute path to the literal sequence `~+~`, which then
        # fails `normalizePath(mustWork = TRUE)`. A relative invocation never goes through that
        # reconstruction. Ordinary trailing arguments (the exchange file paths below) are
        # unaffected either way -- `commandArgs(trailingOnly = TRUE)` passes them through exactly.
        completed = subprocess.run(
            [self.rscript, self.script.name, "--preflight"],
            cwd=self.script.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ModelTrainingContractError(
                "Required R forest compatibility gate failed: " + _stderr_excerpt(completed.stderr)
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ModelTrainingContractError(
                "R forest preflight returned invalid metadata"
            ) from exc
        required = {"R", "survival", "LTRCforests", "jsonlite"}
        if set(payload) != required or not all(
            isinstance(value, str) for value in payload.values()
        ):
            raise ModelTrainingContractError("R forest preflight metadata is incomplete")
        return payload

    def fit(
        self,
        matrix: pd.DataFrame,
        metadata: pd.DataFrame,
        *,
        mtry: int,
        nodesize: int,
        seed: int,
        destination: Path,
    ) -> RForestResult:
        values = matrix.to_numpy(float)
        entry = metadata["entry"].to_numpy(float)
        exit_ = metadata["exit"].to_numpy(float)
        event = metadata["event"].to_numpy(int)
        validate_survival_inputs(values, entry, exit_, event)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="crc-phase4-r-") as directory:
            root = Path(directory)
            exchange = pd.concat(
                [
                    matrix.reset_index(drop=True),
                    metadata.loc[:, ["entry", "exit", "event"]].reset_index(drop=True),
                ],
                axis=1,
            )
            input_path, response_path = root / "fit.csv", root / "response.json"
            exchange.to_csv(input_path, index=False)
            completed = subprocess.run(
                [
                    self.rscript,
                    self.script.name,
                    "--fit",
                    str(input_path),
                    str(destination),
                    str(response_path),
                    str(mtry),
                    str(nodesize),
                    str(r_seed(seed)),
                ],
                cwd=self.script.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0 or not response_path.is_file():
                # Carry R's own stderr into the error. Without it a segfault arrives as a bare
                # "fit failed" and the crash signature has to be reproduced by hand -- which is
                # exactly how an unverified crash mechanism survived in this project's docs.
                raise ModelTrainingContractError(
                    "R forest fit failed: " + _stderr_excerpt(completed.stderr)
                )
            payload = json.loads(response_path.read_text(encoding="utf-8"))
        if not destination.is_file() or payload.get("trees") != 500:
            raise ModelTrainingContractError("R forest bundle is incomplete")
        survival = np.asarray(payload.get("survival", []), dtype=float)
        if (
            survival.shape != (1, 2)
            or not np.isfinite(survival).all()
            or np.any(survival < 0)
            or np.any(survival > 1)
        ):
            raise ModelTrainingContractError("R forest survival fixture is invalid")
        return RForestResult(
            destination, survival, {key: str(value) for key, value in payload["runtime"].items()}
        )
