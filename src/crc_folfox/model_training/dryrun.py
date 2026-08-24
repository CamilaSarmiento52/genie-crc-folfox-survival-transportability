"""Synthetic computational dry run (Block C, Step 8).

Measures per-fit wall time and incremental peak memory on a small synthetic Cox fit -- the unit
both the 7,800 spec-locked elastic-net inner fits and the frozen final-fit inventory are built
from -- then extrapolates linearly to project total wall time, peak memory, and temporary storage
for the complete inventory. Must run after Step 7's admission decision: the forest was not
admitted (`technically_ineligible`, see `admission.py`/`implement.md`), so no forest inner-fit
budget is included here. Never touches restricted data or restricted outcomes; the projection is a
planning artifact against a synthetic fixture, not a performance claim about the real release.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, replace
from pathlib import Path

from crc_folfox.model_training.bundles import extract_cox_bundle_state, save_bundle
from crc_folfox.model_training.contracts import REQUIRED_FINAL_FIT_COUNT
from crc_folfox.model_training.cox import fit_cox
from crc_folfox.preprocessing.contracts import build_endpoint_view
from crc_folfox.preprocessing.transform import FittedPreprocessor

# Spec-locked (model-training-spec.md, "Prespecified computational workload gate"): 7,800
# elastic-net inner fits across both endpoints under the full (unreduced) resampling topology.
# Independent of whether Step 6/8's own tests use a reduced topology for speed.
ELASTIC_NET_INNER_FIT_COUNT = 7800


@dataclass(frozen=True)
class DryRunMeasurement:
    """One representative Cox fit's measured cost -- the unit both inner and final fits use."""

    sample_size: int
    mean_fit_seconds: float
    peak_incremental_bytes: int
    bundle_bytes: int


@dataclass(frozen=True)
class DryRunProjection:
    measurement: DryRunMeasurement
    forest_admitted: bool
    inner_fit_count: int
    final_fit_count: int
    projected_wall_time_seconds: float
    projected_peak_memory_mb: float
    projected_temporary_storage_mb: float

    def to_json(self) -> dict[str, object]:
        return {
            "sample_size": self.measurement.sample_size,
            "mean_fit_seconds": self.measurement.mean_fit_seconds,
            "forest_admitted": self.forest_admitted,
            "inner_fit_count": self.inner_fit_count,
            "final_fit_count": self.final_fit_count,
            "projected_wall_time_seconds": self.projected_wall_time_seconds,
            "projected_peak_memory_mb": self.projected_peak_memory_mb,
            "projected_temporary_storage_mb": self.projected_temporary_storage_mb,
        }


def measure_one_cox_fit(frame_builder, *, repeats: int = 20, tmp_dir: Path) -> DryRunMeasurement:
    """Time and memory-profile one representative preprocess-fit-serialize cycle.

    `frame_builder()` must return a fresh synthetic OS-endpoint-shaped frame each call. The timed
    region includes `FittedPreprocessor.fit`/`.transform`, not just `fit_cox` -- deliberately
    conservative: the real elastic-net inner loop refits its preprocessor once per inner fold and
    reuses it across all 12 grid candidates (`selection.tune_elastic_net_cox`), so charging every
    one of the 7,800 counted "fits" a full preprocessing cost overestimates true preprocessing
    load by roughly 12x rather than modeling that amortization precisely -- a safe upper bound for
    a resource budget, not a claim that the real run needs a full preprocessing pass per fit.
    `tracemalloc` measures *incremental* allocation during the cycle, not total process RSS --
    appropriate for extrapolating N sequential fits, since Python's allocator reclaims memory
    between fits rather than accumulating it; treating each fit's peak as independent is
    intentionally the conservative (upper-bound) assumption for a resource budget, not a claim
    that N concurrent fits would need N times the memory.
    """
    durations: list[float] = []
    peak_bytes = 0
    bundle_bytes = 0
    for index in range(repeats):
        frame = build_endpoint_view(frame_builder(), "OS").data
        tracemalloc.start()
        started = time.perf_counter()
        preprocessor = FittedPreprocessor.fit(
            frame, endpoint="OS", feature_set="core", fit_scope="dry-run"
        )
        result = preprocessor.transform(frame)
        fitter = fit_cox(result.matrix, result.metadata, penalizer=0.01, l1_ratio=0.0)
        state = extract_cox_bundle_state(
            fitter,
            model_id=f"dry-run-{index}",
            endpoint="OS",
            design="pooled",
            held_out_institution=None,
            repeat=0,
            fold=0,
            model_name="core_cox",
            feature_set="core",
            penalizer=0.01,
            l1_ratio=0.0,
            preprocessor=replace(preprocessor, fit_patient_keys=frozenset()),
            spline_state=None,
            development_patient_count=len(frame),
            development_event_count=int(frame["event"].sum()),
        )
        bundle_path = tmp_dir / f"dry-run-{index}.joblib"
        save_bundle(state, bundle_path)
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        durations.append(elapsed)
        peak_bytes = max(peak_bytes, peak)
        bundle_bytes = bundle_path.stat().st_size

    return DryRunMeasurement(
        sample_size=repeats,
        mean_fit_seconds=sum(durations) / len(durations),
        peak_incremental_bytes=peak_bytes,
        bundle_bytes=bundle_bytes,
    )


def project_computational_budget(
    measurement: DryRunMeasurement,
    *,
    forest_admitted: bool = False,
    inner_fit_count: int = ELASTIC_NET_INNER_FIT_COUNT,
    final_fit_count: int = REQUIRED_FINAL_FIT_COUNT,
) -> DryRunProjection:
    """Extrapolate one fit's measured cost to the complete frozen inventory.

    Linear in fit count: no concurrency/parallelism is assumed, matching this codebase's current
    strictly sequential fit loop (`workflow._fit_required_model`/`selection.tune_elastic_net_cox`).
    """
    if forest_admitted:
        raise NotImplementedError(
            "Forest inner-fit budgeting is out of scope: ltrc_rrf is technically_ineligible "
            "(Step 7); re-derive this projection only if a dated amendment later admits it."
        )
    total_fits = inner_fit_count + final_fit_count
    wall_time = total_fits * measurement.mean_fit_seconds
    peak_memory_mb = measurement.peak_incremental_bytes / (1024 * 1024)
    storage_mb = (final_fit_count * measurement.bundle_bytes) / (1024 * 1024)
    return DryRunProjection(
        measurement=measurement,
        forest_admitted=forest_admitted,
        inner_fit_count=inner_fit_count,
        final_fit_count=final_fit_count,
        projected_wall_time_seconds=wall_time,
        projected_peak_memory_mb=peak_memory_mb,
        projected_temporary_storage_mb=storage_mb,
    )


__all__ = [
    "ELASTIC_NET_INNER_FIT_COUNT",
    "DryRunMeasurement",
    "DryRunProjection",
    "measure_one_cox_fit",
    "project_computational_budget",
]
