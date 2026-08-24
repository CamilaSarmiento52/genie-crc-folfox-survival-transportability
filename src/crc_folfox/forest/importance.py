"""WP4b: grouped delayed-entry permutation importance for the admitted `ltrc_cif`, and its rank
agreement with the Cox models explainability workflow already explained (`forest-retry-spec.md` §9).

forest-model-retry workflow answered whether the forest predicts better than `minimal_cox`. It does
not. This module
answers the follow-up question — *why* — by placing the forest inside the explanation set
explainability workflow
built, rather than by adding a new explanation method. If the forest recovers the same drivers on
the same delayed-entry loss scale, the null has a mechanism: flexibility found nothing extra because
there was nothing extra to find.

Everything metric-shaped here is **imported**:

* the delayed-entry IPCW Brier estimator, through `explain.importance` (which itself imports
  primary-evaluation workflow's Gate-3-validated `evaluate.prediction_error`);
* `permutation_seed`, `R_PERM`, `IBS_GRID`, `rank_agreement`, `label_scope_basis`, and
  explainability workflow's
  `_reporting_status` rule, each used as the object explainability workflow uses, never re-declared.

The forest reaches that machinery through one *additive* hook —
`explain.loading.register_matrix_predictor` — because explainability workflow's nine-model register
is frozen with a
hard `== 9` inventory assertion, and `ltrc_cif` is a forest model borrowing explainability code
rather
than a tenth explainability model. Registering the hook cannot change how any of the nine resolves;
the
registry is consulted only after every explainability name has already been dispatched.

Comparator importances are **read** from the published explainability workflow
`importance_summary.csv`, never
recomputed. Recomputing a frozen published number to make a forest-model-retry workflow comparison
convenient is exactly
the failure mode `mission.md` non-negotiable 9 exists to prevent.

No bootstrap intervals. explainability workflow's interval machinery is `B_EXPLAIN = 200` model
**refits** per
scope; at the measured ~12.5 minutes per forest fit that is ~125 hours against forest-model-retry
workflow's entire
11h02m fitting run. Every ranking here is a point ranking, so only the *agreement* between two
rankings is interpreted — never "group A matters more than group B".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crc_folfox.explain.comparability import label_scope_basis, rank_agreement
from crc_folfox.explain.contracts import R_PERM, ExplainContractError
from crc_folfox.explain.feature_groups import build_feature_groups
from crc_folfox.explain.importance import _fit_km_safe, compute_group_importance
from crc_folfox.explain.loading import (
    RUNTIME_R,
    LoadedModel,
    RiskPrediction,
    loaded_feature_columns,
    register_matrix_predictor,
)
from crc_folfox.explain.workflow import _reporting_status
from crc_folfox.forest.r_adapter import RConditionalForestAdapter
from crc_folfox.forest.workflow import ANALYSIS_STATUS, CANDIDATE, FEATURE_SET, _scope_id
from crc_folfox.preprocessing.contracts import EndpointView, SplitDefinition, build_endpoint_view
from crc_folfox.preprocessing.splits import build_iecv_splits
from crc_folfox.preprocessing.transform import FittedPreprocessor

#: The forest's `explanation_tier` in its own artifact. It is deliberately *not* one of
#: explainability workflow's
#: four tiers: the forest never entered explainability workflow's availability hierarchy, and
#: borrowing a tier label
#: would imply it did.
FOREST_TIER = "post_hoc_forest"

#: explainability workflow explains the three leave-one-centre-out scopes and nothing else. Matching
#: that exactly is
#: the point — a different scope set would make the two sides of the comparison different
#: quantities. `pooled` is the within-mixture comparator, not the held-out quantity.
IMPORTANCE_ENDPOINT = "OS"

#: The core-feature-set Cox model with the identical group inventory. This, not `minimal_cox`,
#: carries the rank comparison: `minimal_cox` uses the four-variable baseline set, whose
#: intersection with the core set is only three groups (`number_metastatic_sites` is baseline-only),
#: and a Kendall tau-b over three points is not evidence.
PRIMARY_COMPARATOR = "spline_cox"

#: Reported because it is the frozen forest-model-retry workflow benchmark, and labelled
#: `restricted_scopes` so its
#: three-group tau can never be read as the primary comparison.
BENCHMARK_COMPARATOR = "minimal_cox"

#: The remaining core-feature-set models explainability workflow explained. `forest-retry-spec.md`
#: §9's comparator
#: table lists these as "Supporting. Free -- their importances are already published", and the first
#: implementation shipped without them: only `spline_cox` and `minimal_cox` were compared, so the
#: frozen spec promised rows the artifact did not contain. Added 2026-08-16.
#:
#: They are genuinely free. Every comparison is `rank_agreement` over two already-published
#: rankings; nothing is refit, re-permuted, or re-scored. Their value is that one primary
#: comparator cannot distinguish "the forest agrees with the Cox family" from "the forest agrees
#: with `spline_cox` specifically", and seven comparators can.
SUPPORTING_COMPARATORS = (
    "elastic_net_cox",
    "glmnet_ridge_core",
    "glmnet_enet_core",
    "penalized_nonlinear_cox",
    "reduced_core_penalized_cox",
)

#: Ordered `(comparator, role)`. `role` is what keeps the thin `minimal_cox` comparison from being
#: read as the primary one, and is carried into the artifact rather than left to the reader.
COMPARATORS: tuple[tuple[str, str], ...] = (
    (PRIMARY_COMPARATOR, "primary"),
    (BENCHMARK_COMPARATOR, "descriptive"),
    *((name, "supporting") for name in SUPPORTING_COMPARATORS),
)

METRICS = ("365", "730", "ibs")

IMPORTANCE_ARTIFACT = "importance_summary.csv"
AGREEMENT_ARTIFACT = "importance_agreement.csv"

_ADAPTER = RConditionalForestAdapter()


def _forest_matrix_prediction(
    loaded: LoadedModel,
    matrix: pd.DataFrame,
    entry: np.ndarray,
    horizons: tuple[float, ...],
) -> RiskPrediction:
    """Predict absolute risk on both scales from an already-transformed (possibly permuted) design.

    `predict_from_matrix` supplies only `entry`, while the R exchange CSV carries the full
    `Surv(entry, exit, event)` response triple. `exit`/`event` are therefore **synthesized** here.
    That is sound because `--predict` passes `newdata` to `predictProb`, which reads predictors and
    `entry` alone; the outcome columns are inert on the prediction path. Two tests hold that
    invariant in place — one asserting predictions are unchanged under a different synthesized
    outcome vector, and its complement asserting `entry` still changes the entry-conditional scale.

    Permuting a predictor block never touches this synthesis: the *scoring* side always uses the
    caller's own unpermuted `entry`/`exit`/`event`, so risk-set membership, the censoring model, and
    the IPCW weights are recomputed unchanged on every repeat.
    """
    if loaded.r_bundle_path is None:
        raise ExplainContractError("Forest LoadedModel carries no bundle path")
    entry = np.asarray(entry, dtype=float)
    if len(entry) != len(matrix):
        raise ExplainContractError("Forest entry vector does not match the design matrix")
    metadata = pd.DataFrame(
        {"entry": entry, "exit": entry + 1.0, "event": np.zeros(len(entry), dtype=int)}
    )
    survival = _ADAPTER.predict(loaded.r_bundle_path, matrix, metadata, times=horizons)
    return RiskPrediction(
        unconditional=1.0 - survival["unconditional"],
        entry_conditional=1.0 - survival["entry_conditional"],
        linear_predictor=np.full(len(entry), np.nan),
        coefficients=None,
    )


register_matrix_predictor(CANDIDATE, _forest_matrix_prediction)


@dataclass(frozen=True)
class ForestImportanceResult:
    importance_summary: pd.DataFrame
    agreement_summary: pd.DataFrame
    output: Path


def _load_registry(phase9: Path) -> dict[str, dict]:
    summary = pd.read_csv(phase9 / "fit_summary.csv")
    return {str(row["scope_id"]): dict(row) for _, row in summary.iterrows()}


def _published_phase6_importance(
    explainability: Path,
) -> dict[tuple[str, str, str, str], dict[str, float]]:
    """`(endpoint, centre, model, metric) -> {feature_group: importance}` from the **published**
    explainability artifact. Rows with a missing importance are dropped rather than zero-filled: a
    group
    whose importance was not computable has no rank, and inventing one would silently move it to the
    bottom of every ranking it appears in.
    """
    frame = pd.read_csv(explainability / "importance_summary.csv")
    lookup: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        if pd.isna(row.importance):
            continue
        key = (str(row.endpoint), str(row.centre), str(row.model), str(row.metric))
        lookup.setdefault(key, {})[str(row.feature_group)] = float(row.importance)
    if not lookup:
        raise ExplainContractError("Published Phase 6 importance summary is empty")
    return lookup


def _forest_loaded_model(
    view: EndpointView,
    split: SplitDefinition,
    record: dict,
    models: Path,
) -> tuple[LoadedModel, pd.DataFrame, pd.DataFrame]:
    """Rebuild the scope's own transform and wrap the already-fitted bundle. Nothing is refitted.

    The preprocessor is fit on this scope's *development* frame with the identical `feature_set` and
    `fit_scope` `forest/evaluation.py` uses, so the design matrix the forest is explained on is the
    same one it was fitted and evaluated on.
    """
    scope_id = _scope_id(split)
    development, assessment = split.frames(view)
    preprocessor = FittedPreprocessor.fit(
        development, endpoint=split.endpoint, feature_set=FEATURE_SET, fit_scope=scope_id
    )
    transformed = preprocessor.transform(assessment)
    loaded = LoadedModel(
        model_name=CANDIDATE,
        runtime=RUNTIME_R,
        development_patient_count=int(record["development_patients"]),
        development_event_count=int(record["development_events"]),
        r_bundle_path=models / Path(str(record["bundle"])).name,
        r_preprocessor=preprocessor,
    )
    return loaded, transformed.matrix, transformed.metadata


def _scope_importance_rows(
    loaded: LoadedModel,
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    endpoint: str,
    centre: str,
    r_perm: int,
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    """One scope's importance rows, plus `{metric: {group: importance}}` for the agreement pass."""
    event = metadata["event"].to_numpy(dtype=int)
    n_events, n_event_free = int(event.sum()), int((event == 0).sum())

    # Canonical inventory from the model's own frozen schema, never from the assessment matrix --
    # explainability workflow's 2026-08-11 amendment. The assertion catches drift rather than
    # absorbing it.
    canonical = build_feature_groups(loaded_feature_columns(loaded))
    realized = build_feature_groups(tuple(matrix.columns))
    if set(canonical) != set(realized):
        raise ExplainContractError(
            "Forest assessment matrix group inventory differs from the fitted schema"
        )

    km = _fit_km_safe(
        metadata["entry"].to_numpy(dtype=float),
        metadata["exit"].to_numpy(dtype=float),
        event,
    )

    rows: list[dict] = []
    by_metric: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    for group_name, columns in canonical.items():
        result = compute_group_importance(
            loaded,
            matrix,
            metadata,
            columns,
            endpoint=endpoint,
            centre=centre,
            group_name=group_name,
            r_perm=r_perm,
            km=km,
        )
        measured = {
            "365": result.importance_365(),
            "730": result.importance_730(),
            "ibs": result.importance_ibs(),
        }
        for metric in METRICS:
            value, finite = measured[metric]
            if value is not None:
                by_metric[metric][group_name] = value
            rows.append(
                {
                    "endpoint": endpoint,
                    "centre": centre,
                    "model": CANDIDATE,
                    "explanation_tier": FOREST_TIER,
                    "analysis_status": ANALYSIS_STATUS,
                    "feature_group": group_name,
                    "n_columns": result.n_columns,
                    "metric": metric,
                    "importance": value,
                    "finite_fraction": finite,
                    "reporting_status": _reporting_status(n_events, n_event_free, finite),
                }
            )
    return rows, by_metric


def _agreement_row(
    *,
    endpoint: str,
    centre: str,
    metric: str,
    comparator: str,
    role: str,
    forest_importance: dict[str, float],
    comparator_importance: dict[str, float],
) -> dict:
    """One forest-vs-Cox rank-agreement row.

    forest-model-retry workflow builds this itself rather than calling
    `comparability.build_cross_model_row`, which
    reads `MODEL_REGISTER_BY_NAME[model_name].tier` — `ltrc_cif` is not in that frozen register and
    must not be added to it. The statistics themselves (`rank_agreement`, `label_scope_basis`) are
    the imported explainability functions, unchanged.
    """
    shared = set(forest_importance) & set(comparator_importance)
    is_restricted = shared != set(forest_importance) or shared != set(comparator_importance)
    agreement = rank_agreement(
        {group: forest_importance[group] for group in shared},
        {group: comparator_importance[group] for group in shared},
    )
    # Every comparison here is cross-tier by construction: the forest is outside explainability
    # workflow's hierarchy
    # entirely, which is a stronger version of the condition R4 was written for.
    label = label_scope_basis(is_restricted=is_restricted, is_cross_tier=True)
    return {
        "endpoint": endpoint,
        "centre": centre,
        "metric": metric,
        "model_a": CANDIDATE,
        "model_b": comparator,
        "role": role,
        "scope_basis": label.scope_basis,
        "availability_confounded": label.availability_confounded,
        "availability_artifact_possible": label.availability_artifact_possible,
        "n_shared_groups": len(shared),
        "n_groups_a": len(forest_importance),
        "n_groups_b": len(comparator_importance),
        "kendall_tau": agreement.kendall_tau,
        "top_k": agreement.top_k,
        "topk_overlap": agreement.topk_overlap,
        "jaccard": agreement.jaccard,
        "top_k_boundary_tied": agreement.top_k_boundary_tied,
        "analysis_status": ANALYSIS_STATUS,
    }


def _agreement_rows_for_scope(
    *,
    endpoint: str,
    centre: str,
    by_metric: dict[str, dict[str, float]],
    published: dict[tuple[str, str, str, str], dict[str, float]],
) -> list[dict]:
    """Every comparator's rank-agreement row for one centre.

    A comparator explainability workflow did not explain at this centre is skipped rather than
    reported as a zero
    agreement — an absent ranking has no rank correlation with anything.
    """
    rows: list[dict] = []
    for metric in METRICS:
        forest_importance = by_metric.get(metric) or {}
        if not forest_importance:
            continue
        for comparator, role in COMPARATORS:
            comparator_importance = published.get((endpoint, centre, comparator, metric))
            if not comparator_importance:
                continue
            rows.append(
                _agreement_row(
                    endpoint=endpoint,
                    centre=centre,
                    metric=metric,
                    comparator=comparator,
                    role=role,
                    forest_importance=forest_importance,
                    comparator_importance=comparator_importance,
                )
            )
    return rows


def _forest_importance_by_centre(
    published_forest: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    """`centre -> metric -> {group: importance}` from forest-model-retry workflow's **own**
    published importance.

    Lets the agreement pass be rebuilt without repeating the ~14-minute permutation run: the
    forest's importances are a published forest-retry artifact, and rank agreement is a pure
    function of
    two already-published rankings. Rows with a missing importance are dropped, matching how the
    explainability workflow comparator side is read.
    """
    by_centre: dict[str, dict[str, dict[str, float]]] = {}
    for row in published_forest.itertuples(index=False):
        if pd.isna(row.importance) or str(row.model) != CANDIDATE:
            continue
        centre = by_centre.setdefault(str(row.centre), {metric: {} for metric in METRICS})
        metric = str(row.metric)
        if metric in centre:
            centre[metric][str(row.feature_group)] = float(row.importance)
    return by_centre


def run_forest_importance_workflow(
    input_dir: Path,
    output_dir: Path,
    *,
    endpoint: str = IMPORTANCE_ENDPOINT,
    r_perm: int = R_PERM,
    scope_limit: int | None = None,
    reuse_published_importance: bool = False,
    progress: bool = False,
) -> ForestImportanceResult:
    """Compute `ltrc_cif` importance on the three OS IECV scopes; compare its ranking to
    explainability workflow's.

    `reuse_published_importance` rebuilds only the agreement artifact from the already-published
    `importance_summary.csv`, skipping the permutation run. Use it when the comparator set or the
    agreement statistics change but the forest's own importances have not — the permutation numbers
    are then carried over byte-for-byte rather than recomputed, which also makes it impossible for a
    re-run to silently perturb a published importance.

    `scope_limit` and a reduced `r_perm` exist for smoke tests only; a real run uses neither.
    """
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    published = _published_phase6_importance(input_dir / "explainability")

    if reuse_published_importance:
        existing = output_dir / IMPORTANCE_ARTIFACT
        if not existing.is_file():
            raise ExplainContractError(
                f"'{IMPORTANCE_ARTIFACT}' does not exist; there is no published importance to reuse"
            )
        importance = pd.read_csv(existing)
        agreement_rows: list[dict] = []
        for centre, by_metric in _forest_importance_by_centre(importance).items():
            agreement_rows.extend(
                _agreement_rows_for_scope(
                    endpoint=endpoint, centre=centre, by_metric=by_metric, published=published
                )
            )
        # Deliberately republishes only the agreement artifact. A read-then-write round trip through
        # pandas is value-preserving but *not* byte-preserving — float repr can change — so
        # rewriting `importance_summary.csv` here would churn a published artifact's checksum for no
        # reason and make "did the permutation numbers move?" unanswerable by hash. Leaving the file
        # untouched makes the answer trivially yes.
        agreement = pd.DataFrame(agreement_rows)
        _publish(output_dir, None, agreement)
        return ForestImportanceResult(importance, agreement, output_dir)

    registry = _load_registry(output_dir)
    models = output_dir / "models"

    dataset = pd.read_csv(input_dir / "analysis_dataset.csv")
    view = build_endpoint_view(dataset, endpoint)
    splits = build_iecv_splits(view)
    if scope_limit is not None:
        splits = splits[:scope_limit]

    importance_rows: list[dict] = []
    agreement_rows = []

    for split in splits:
        scope_id = _scope_id(split)
        record = registry.get(scope_id)
        if record is None or record.get("fit_status") != "fit_succeeded":
            raise ExplainContractError(f"'{scope_id}' has no fit_succeeded forest to explain")
        centre = str(split.held_out_institution)
        if progress:
            print(f"[importance] {centre}: transforming and predicting", flush=True)

        loaded, matrix, metadata = _forest_loaded_model(view, split, record, models)
        rows, by_metric = _scope_importance_rows(
            loaded, matrix, metadata, endpoint=endpoint, centre=centre, r_perm=r_perm
        )
        importance_rows.extend(rows)
        if progress:
            print(f"[importance] {centre}: {len(by_metric['730'])} groups scored", flush=True)

        agreement_rows.extend(
            _agreement_rows_for_scope(
                endpoint=endpoint, centre=centre, by_metric=by_metric, published=published
            )
        )

    importance = pd.DataFrame(importance_rows)
    agreement = pd.DataFrame(agreement_rows)
    _publish(output_dir, importance, agreement)
    return ForestImportanceResult(importance, agreement, output_dir)


def _publish(output_dir: Path, importance: pd.DataFrame | None, agreement: pd.DataFrame) -> None:
    """Atomic-ish publish: write a temp file per artifact, then replace. A partially written CSV is
    never left where a reader could mistake it for a complete result.

    `importance=None` leaves the published importance artifact untouched — used by the
    agreement-only rebuild, so a comparator change cannot churn the permutation artifact's bytes.
    """
    for name, frame in ((IMPORTANCE_ARTIFACT, importance), (AGREEMENT_ARTIFACT, agreement)):
        if frame is None:
            continue
        destination = output_dir / name
        staging = destination.with_suffix(".csv.tmp")
        frame.to_csv(staging, index=False)
        staging.replace(destination)


__all__ = [
    "AGREEMENT_ARTIFACT",
    "BENCHMARK_COMPARATOR",
    "COMPARATORS",
    "FOREST_TIER",
    "IMPORTANCE_ARTIFACT",
    "IMPORTANCE_ENDPOINT",
    "METRICS",
    "PRIMARY_COMPARATOR",
    "SUPPORTING_COMPARATORS",
    "ForestImportanceResult",
    "run_forest_importance_workflow",
]
