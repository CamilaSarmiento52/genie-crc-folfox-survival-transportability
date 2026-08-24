"""Private, schema-validated bridges to the optimization workflow glmnet and flexsurv R runtimes.

R fits and predicts; every validation score, lambda/knot-count selection, and comparison stays in
Python (tech-stack.md, "Software and the R boundary"). Mirrors
`crc_folfox.model_training.r_adapter`'s subprocess-exchange pattern: a private temporary directory
carries the exchange CSVs and response JSON, and nothing survives the call.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crc_folfox.model_training.contracts import r_seed, validate_survival_inputs
from crc_folfox.optimize.contracts import Phase5AContractError


def _horizons_arg(horizons: tuple[float, ...]) -> str:
    return ",".join(str(float(h)) for h in horizons)


def _run_preflight(
    rscript: str, script: Path, required_keys: set[str], label: str
) -> dict[str, str]:
    if shutil.which(rscript) is None:
        raise Phase5AContractError(f"Required R runtime is unavailable for {label}")
    # Invoke by filename with `cwd` set to the script's own directory, not by absolute path: on
    # this platform, Rscript's self-reported `--file=` value (used only for renv activation)
    # mangles spaces in an absolute path to the literal sequence `~+~`, which then fails
    # `normalizePath(mustWork = TRUE)`. A relative invocation never goes through that
    # reconstruction. Ordinary trailing arguments (the exchange file paths below) are unaffected
    # either way -- `commandArgs(trailingOnly = TRUE)` passes them through exactly.
    completed = subprocess.run(
        [rscript, script.name, "--preflight"],
        cwd=script.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise Phase5AContractError(f"{label} preflight failed: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise Phase5AContractError(f"{label} preflight returned invalid metadata") from exc
    if set(payload) != required_keys or not all(isinstance(v, str) for v in payload.values()):
        raise Phase5AContractError(f"{label} preflight metadata is incomplete")
    return payload


def _validate_columns_match(train_matrix: pd.DataFrame, assessment_matrix: pd.DataFrame) -> None:
    if not assessment_matrix.columns.equals(train_matrix.columns):
        raise Phase5AContractError("Assessment matrix columns must match the training matrix")


def _train_exchange(matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    values = matrix.to_numpy(float)
    entry = metadata["entry"].to_numpy(float)
    exit_ = metadata["exit"].to_numpy(float)
    event = metadata["event"].to_numpy(int)
    validate_survival_inputs(values, entry, exit_, event)
    return pd.concat(
        [
            matrix.reset_index(drop=True),
            metadata.loc[:, ["entry", "exit", "event"]].reset_index(drop=True),
        ],
        axis=1,
    )


# --- glmnet: B2 `glmnet_ridge_core`, B3 `glmnet_enet_core` ------------------------------------


@dataclass(frozen=True)
class GlmnetPathResult:
    bundle: Path
    lambda_values: np.ndarray  # (n_lambda,)
    survival: np.ndarray  # (n_lambda, n_assessment_rows, n_horizons)
    horizons: np.ndarray
    runtime: dict[str, str]
    # Populated only when the bundle was fit at exactly one lambda (the final development-scope
    # refit, or any bundle reached only via `predict`) -- explainability workflow's
    # `explain.coefficients` reads
    # this; a "path" fit (more than one lambda) leaves it `None` (see glmnet_adapter.R,
    # `.coefficients_if_single_lambda`).
    coefficients: dict[str, float] | None = None
    # The opposite condition: populated only for a multi-lambda "path" fit, one
    # `{predictor_column: coefficient}` dict per entry of `lambda_values` (same order) --
    # `explain.coefficient_paths` reads this for `coefficient_paths.csv`
    # (`explainability-spec.md`'s 2026-08-11 amendment). `None` for a single-lambda bundle.
    coefficient_path: list[dict[str, float]] | None = None


class GlmnetAdapter:
    """Invoke the project-owned R glmnet Cox adapter without publishing exchange files."""

    _REQUIRED_KEYS = {"R", "survival", "glmnet", "jsonlite"}

    def __init__(self, script: str | Path | None = None, rscript: str = "Rscript") -> None:
        self.script = Path(
            script or Path(__file__).parents[3] / "r" / "phase5a" / "glmnet_adapter.R"
        )
        self.rscript = rscript

    def preflight(self) -> dict[str, str]:
        return _run_preflight(self.rscript, self.script, self._REQUIRED_KEYS, "glmnet adapter")

    def fit(
        self,
        train_matrix: pd.DataFrame,
        train_metadata: pd.DataFrame,
        assessment_matrix: pd.DataFrame,
        *,
        alpha: float,
        lambda_value: float | tuple[float, ...] | None,
        seed: int,
        horizons: tuple[float, ...],
        destination: Path,
    ) -> GlmnetPathResult:
        """Fit one glmnet Cox path (or one or more explicit lambda values) and predict at
        `horizons`.

        `lambda_value=None` computes the frozen `nlambda=50`/`lambda.min.ratio=0.01` path fresh on
        `train_matrix` -- used once per parent development scope to fix a reference lambda
        sequence. A tuple of explicit values forces the fit onto exactly those lambdas -- used for
        every inner fold, so every fold scores the same candidate set on the parent's shared
        reference sequence rather than each fold's own data-dependent path (see
        `the glmnet adapter script`'s `--fit` docstring). A single float refits at that one
        previously selected value only -- the final development-scope refit.
        """
        _validate_columns_match(train_matrix, assessment_matrix)
        train_exchange = _train_exchange(train_matrix, train_metadata)

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="crc-phase5a-glmnet-") as directory:
            root = Path(directory)
            train_path, val_path = root / "train.csv", root / "assessment.csv"
            response_path = root / "response.json"
            train_exchange.to_csv(train_path, index=False)
            assessment_matrix.reset_index(drop=True).to_csv(val_path, index=False)

            if lambda_value is None:
                lambda_spec = "path"
            elif isinstance(lambda_value, tuple):
                lambda_spec = ",".join(str(float(v)) for v in lambda_value)
            else:
                lambda_spec = str(float(lambda_value))
            completed = subprocess.run(
                [
                    self.rscript,
                    self.script.name,
                    "--fit",
                    str(train_path),
                    str(val_path),
                    str(destination),
                    str(response_path),
                    str(float(alpha)),
                    lambda_spec,
                    str(r_seed(seed)),
                    _horizons_arg(horizons),
                ],
                cwd=self.script.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0 or not response_path.is_file():
                raise Phase5AContractError(f"glmnet adapter fit failed: {completed.stderr.strip()}")
            payload = json.loads(response_path.read_text(encoding="utf-8"))

        if not destination.is_file():
            raise Phase5AContractError("glmnet adapter bundle is missing")
        lambda_values = np.atleast_1d(np.asarray(payload.get("lambda", []), dtype=float))
        survival = np.asarray(payload.get("survival", []), dtype=float)
        returned_horizons = np.asarray(payload.get("horizons", []), dtype=float)
        expected_shape = (len(lambda_values), len(assessment_matrix), len(horizons))
        if (
            lambda_values.size == 0
            or survival.shape != expected_shape
            or not np.isfinite(survival).all()
            or np.any(survival < 0)
            or np.any(survival > 1)
            or not np.allclose(returned_horizons, np.asarray(horizons, dtype=float))
        ):
            raise Phase5AContractError("glmnet adapter survival payload is invalid")
        return GlmnetPathResult(
            destination,
            lambda_values,
            survival,
            returned_horizons,
            {key: str(value) for key, value in payload["runtime"].items()},
            coefficients=_parse_coefficients(payload),
            coefficient_path=_parse_coefficient_path(payload),
        )

    def predict(
        self,
        bundle: Path,
        assessment_matrix: pd.DataFrame,
        *,
        horizons: tuple[float, ...],
    ) -> GlmnetPathResult:
        """Predict from an already-fitted bundle (saved by `fit`) with no refitting -- the
        evaluation-time path. Predicts at every lambda the bundle was fit with (one, for a
        final-refit bundle)."""
        if not bundle.is_file():
            raise Phase5AContractError("glmnet adapter bundle is missing")
        with tempfile.TemporaryDirectory(prefix="crc-phase5a-glmnet-predict-") as directory:
            root = Path(directory)
            val_path = root / "assessment.csv"
            response_path = root / "response.json"
            assessment_matrix.reset_index(drop=True).to_csv(val_path, index=False)
            completed = subprocess.run(
                [
                    self.rscript,
                    self.script.name,
                    "--predict",
                    str(bundle),
                    str(val_path),
                    str(response_path),
                    _horizons_arg(horizons),
                ],
                cwd=self.script.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0 or not response_path.is_file():
                raise Phase5AContractError(
                    f"glmnet adapter predict failed: {completed.stderr.strip()}"
                )
            payload = json.loads(response_path.read_text(encoding="utf-8"))

        lambda_values = np.atleast_1d(np.asarray(payload.get("lambda", []), dtype=float))
        survival = np.asarray(payload.get("survival", []), dtype=float)
        returned_horizons = np.asarray(payload.get("horizons", []), dtype=float)
        expected_shape = (len(lambda_values), len(assessment_matrix), len(horizons))
        if (
            lambda_values.size == 0
            or survival.shape != expected_shape
            or not np.isfinite(survival).all()
            or np.any(survival < 0)
            or np.any(survival > 1)
            or not np.allclose(returned_horizons, np.asarray(horizons, dtype=float))
        ):
            raise Phase5AContractError("glmnet adapter survival payload is invalid")
        return GlmnetPathResult(
            bundle,
            lambda_values,
            survival,
            returned_horizons,
            {key: str(value) for key, value in payload["runtime"].items()},
            coefficients=_parse_coefficients(payload),
        )


def _parse_coefficients(payload: dict) -> dict[str, float] | None:
    """`None` for a multi-lambda "path" response; a `{predictor_column: coefficient}` mapping for
    a single-lambda response (see glmnet_adapter.R, `.coefficients_if_single_lambda`)."""
    raw = payload.get("coefficients")
    if raw is None:
        return None
    return {str(key): float(value) for key, value in raw.items()}


def _parse_coefficient_path(payload: dict) -> list[dict[str, float]] | None:
    """`None` for a single-lambda response; one `{predictor_column: coefficient}` dict per lambda,
    in `lambda_values` order, for a multi-lambda "path" response (see glmnet_adapter.R,
    `.coefficient_path_if_multi_lambda`)."""
    raw = payload.get("coefficient_path")
    if raw is None:
        return None
    return [{str(key): float(value) for key, value in entry.items()} for entry in raw]


# --- flexsurv: B6 `flexsurv_rp_spline` ---------------------------------------------------------


@dataclass(frozen=True)
class FlexsurvResult:
    bundle: Path
    survival: np.ndarray  # (n_assessment_rows, n_horizons)
    horizons: np.ndarray
    runtime: dict[str, str]


class FlexsurvAdapter:
    """Invoke the project-owned R flexsurv Royston-Parmar adapter without publishing exchange
    files."""

    _REQUIRED_KEYS = {"R", "survival", "flexsurv", "jsonlite"}

    def __init__(self, script: str | Path | None = None, rscript: str = "Rscript") -> None:
        self.script = Path(
            script or Path(__file__).parents[3] / "r" / "phase5a" / "flexsurv_adapter.R"
        )
        self.rscript = rscript

    def preflight(self) -> dict[str, str]:
        return _run_preflight(self.rscript, self.script, self._REQUIRED_KEYS, "flexsurv adapter")

    def fit(
        self,
        train_matrix: pd.DataFrame,
        train_metadata: pd.DataFrame,
        assessment_matrix: pd.DataFrame,
        *,
        k: int,
        seed: int,
        horizons: tuple[float, ...],
        destination: Path,
    ) -> FlexsurvResult:
        _validate_columns_match(train_matrix, assessment_matrix)
        train_exchange = _train_exchange(train_matrix, train_metadata)

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="crc-phase5a-flexsurv-") as directory:
            root = Path(directory)
            train_path, val_path = root / "train.csv", root / "assessment.csv"
            response_path = root / "response.json"
            train_exchange.to_csv(train_path, index=False)
            assessment_matrix.reset_index(drop=True).to_csv(val_path, index=False)

            completed = subprocess.run(
                [
                    self.rscript,
                    self.script.name,
                    "--fit",
                    str(train_path),
                    str(val_path),
                    str(destination),
                    str(response_path),
                    str(int(k)),
                    str(r_seed(seed)),
                    _horizons_arg(horizons),
                ],
                cwd=self.script.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0 or not response_path.is_file():
                raise Phase5AContractError(
                    f"flexsurv adapter fit failed: {completed.stderr.strip()}"
                )
            payload = json.loads(response_path.read_text(encoding="utf-8"))

        if not destination.is_file():
            raise Phase5AContractError("flexsurv adapter bundle is missing")
        survival = np.asarray(payload.get("survival", []), dtype=float)
        returned_horizons = np.asarray(payload.get("horizons", []), dtype=float)
        expected_shape = (len(assessment_matrix), len(horizons))
        if (
            payload.get("k") != int(k)
            or survival.shape != expected_shape
            or not np.isfinite(survival).all()
            or np.any(survival < 0)
            or np.any(survival > 1)
            or not np.allclose(returned_horizons, np.asarray(horizons, dtype=float))
        ):
            raise Phase5AContractError("flexsurv adapter survival payload is invalid")
        return FlexsurvResult(
            destination,
            survival,
            returned_horizons,
            {key: str(value) for key, value in payload["runtime"].items()},
        )

    def predict(
        self,
        bundle: Path,
        assessment_matrix: pd.DataFrame,
        *,
        horizons: tuple[float, ...],
    ) -> FlexsurvResult:
        """Predict from an already-fitted bundle (saved by `fit`) with no refitting -- the
        evaluation-time path."""
        if not bundle.is_file():
            raise Phase5AContractError("flexsurv adapter bundle is missing")
        with tempfile.TemporaryDirectory(prefix="crc-phase5a-flexsurv-predict-") as directory:
            root = Path(directory)
            val_path = root / "assessment.csv"
            response_path = root / "response.json"
            assessment_matrix.reset_index(drop=True).to_csv(val_path, index=False)
            completed = subprocess.run(
                [
                    self.rscript,
                    self.script.name,
                    "--predict",
                    str(bundle),
                    str(val_path),
                    str(response_path),
                    _horizons_arg(horizons),
                ],
                cwd=self.script.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0 or not response_path.is_file():
                raise Phase5AContractError(
                    f"flexsurv adapter predict failed: {completed.stderr.strip()}"
                )
            payload = json.loads(response_path.read_text(encoding="utf-8"))

        survival = np.asarray(payload.get("survival", []), dtype=float)
        returned_horizons = np.asarray(payload.get("horizons", []), dtype=float)
        expected_shape = (len(assessment_matrix), len(horizons))
        if (
            survival.shape != expected_shape
            or not np.isfinite(survival).all()
            or np.any(survival < 0)
            or np.any(survival > 1)
            or not np.allclose(returned_horizons, np.asarray(horizons, dtype=float))
        ):
            raise Phase5AContractError("flexsurv adapter survival payload is invalid")
        return FlexsurvResult(
            bundle,
            survival,
            returned_horizons,
            {key: str(value) for key, value in payload["runtime"].items()},
        )
