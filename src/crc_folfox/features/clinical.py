"""Materialize the locked baseline and core clinical predictors."""

from __future__ import annotations

import pandas as pd

from crc_folfox.cohort import CohortResult
from crc_folfox.features.manifest import FeatureContractError, FeatureManifest

SITE_FLAGS = (
    "dmets_abdomen",
    "dmets_bone",
    "dmets_brain",
    "dmets_liver",
    "dmets_pelvis",
    "dmets_thorax",
)


def materialize_clinical_features(
    cohort: CohortResult | pd.DataFrame, manifest: FeatureManifest | None = None
) -> pd.DataFrame:
    """Return identifiers plus raw locked predictors without fitted preprocessing."""
    source = cohort.cohort if isinstance(cohort, CohortResult) else cohort
    active = manifest or FeatureManifest.load()
    active.validate()
    required = {
        "record_id",
        "institution",
        "age_at_regimen",
        *active.core[1:],
        *SITE_FLAGS,
    }
    required.discard("age_at_folfox")
    missing = required - set(source)
    if missing:
        raise FeatureContractError(f"Clinical materialization is missing {len(missing)} columns")
    if not source["record_id"].is_unique:
        raise FeatureContractError("Clinical materialization requires unique patients")
    sites = source.loc[:, SITE_FLAGS].apply(pd.to_numeric, errors="coerce")
    invalid_sites = int((~sites.isin([0, 1]) | sites.isna()).sum().sum())
    if invalid_sites:
        raise FeatureContractError(
            f"Metastatic-site indicators contain {invalid_sites} invalid values"
        )
    output = source.loc[:, ["record_id", "institution", *active.core[1:]]].copy(deep=True)
    output.insert(2, "age_at_folfox", pd.to_numeric(source["age_at_regimen"], errors="coerce"))
    output["number_metastatic_sites"] = sites.sum(axis=1).astype("int64")
    ordered = ["record_id", "institution", *active.baseline]
    ordered.extend(column for column in active.core if column not in active.baseline)
    return output.loc[:, ordered]


def validate_design_columns(columns: list[str] | tuple[str, ...]) -> None:
    """Prevent the deterministic site count and all six flags entering one Cox design."""
    values = set(columns)
    if "number_metastatic_sites" in values and set(SITE_FLAGS).issubset(values):
        raise FeatureContractError(
            "Unpenalized design cannot contain site count and all site indicators"
        )
