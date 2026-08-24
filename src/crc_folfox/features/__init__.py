"""Leakage-audited clinical features and delayed-entry datasets."""

from crc_folfox.features.clinical import (
    SITE_FLAGS,
    materialize_clinical_features,
    validate_design_columns,
)
from crc_folfox.features.datasets import (
    AnalysisDatasetResult,
    AnalysisDatasets,
    build_analysis_datasets,
    validate_analysis_dataset,
)
from crc_folfox.features.entry import (
    SequencingEntryResult,
    build_sequencing_entry,
    horizon_entry_support,
)
from crc_folfox.features.extended import (
    SPECS,
    baseline_availability_profile,
    extract_extended_candidates,
)
from crc_folfox.features.lookback import (
    LookbackResult,
    LookbackSpec,
    closest_prior_record,
    validate_lookback_result,
)
from crc_folfox.features.manifest import (
    LOCKED_BASELINE,
    LOCKED_CORE,
    FeatureContractError,
    FeatureDefinition,
    FeatureManifest,
)
from crc_folfox.features.profiles import category_frequency_profile, missingness_profile

__all__ = [
    "LOCKED_BASELINE",
    "LOCKED_CORE",
    "SITE_FLAGS",
    "SPECS",
    "AnalysisDatasetResult",
    "AnalysisDatasets",
    "FeatureContractError",
    "FeatureDefinition",
    "FeatureManifest",
    "LookbackResult",
    "LookbackSpec",
    "SequencingEntryResult",
    "build_analysis_datasets",
    "build_sequencing_entry",
    "baseline_availability_profile",
    "category_frequency_profile",
    "closest_prior_record",
    "extract_extended_candidates",
    "horizon_entry_support",
    "materialize_clinical_features",
    "missingness_profile",
    "validate_design_columns",
    "validate_analysis_dataset",
    "validate_lookback_result",
]
