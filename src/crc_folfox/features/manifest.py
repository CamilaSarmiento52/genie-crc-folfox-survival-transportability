"""Machine-readable feature definitions and locked-set validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

LOCKED_BASELINE = (
    "age_at_folfox",
    "metastatic_presentation",
    "ca_d_site",
    "number_metastatic_sites",
)
LOCKED_CORE = (
    "age_at_folfox",
    "naaccr_sex_code",
    "race_ethnicity",
    "metastatic_presentation",
    "crc_type",
    "ca_d_site",
    "ca_grade",
    "dmets_abdomen",
    "dmets_bone",
    "dmets_brain",
    "dmets_liver",
    "dmets_pelvis",
    "dmets_thorax",
    "prior_oxaliplatin",
    "biologic_with_folfox",
)


class FeatureContractError(ValueError):
    """A privacy-safe feature or analysis-dataset contract failure."""


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    role: str
    source_table: str
    source_columns: tuple[str, ...]
    derivation: str
    dtype: str
    observation_day_rule: str
    missing_allowed: bool
    reference_rule: str
    encoding: str
    effective_df_rule: str
    sets: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict) -> FeatureDefinition:
        required = {
            "name",
            "role",
            "source_table",
            "source_columns",
            "derivation",
            "dtype",
            "observation_day_rule",
            "missing_allowed",
            "reference_rule",
            "encoding",
            "effective_df_rule",
            "sets",
        }
        if set(value) != required:
            raise FeatureContractError("Feature definition has incomplete or unknown metadata")
        return cls(
            **{
                **value,
                "source_columns": tuple(value["source_columns"]),
                "sets": tuple(value["sets"]),
            }
        )


@dataclass(frozen=True)
class FeatureManifest:
    version: str
    baseline: tuple[str, ...]
    core: tuple[str, ...]
    features: tuple[FeatureDefinition, ...]

    @classmethod
    def load(cls, path: str | Path | None = None) -> FeatureManifest:
        resource = files("crc_folfox.features").joinpath("feature_manifest.json")
        text = Path(path).read_text() if path is not None else resource.read_text()
        raw = json.loads(text)
        if set(raw) != {"version", "baseline", "core", "features"}:
            raise FeatureContractError("Feature manifest has invalid top-level fields")
        manifest = cls(
            version=str(raw["version"]),
            baseline=tuple(raw["baseline"]),
            core=tuple(raw["core"]),
            features=tuple(FeatureDefinition.from_dict(item) for item in raw["features"]),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.baseline != LOCKED_BASELINE or self.core != LOCKED_CORE:
            raise FeatureContractError("Feature-set membership differs from the locked contract")
        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise FeatureContractError("Feature manifest contains duplicate names")
        if set(self.baseline) | set(self.core) != set(names):
            raise FeatureContractError("Feature definitions do not match locked feature sets")
        for feature in self.features:
            expected_sets = {
                name
                for name, members in (("baseline", self.baseline), ("core", self.core))
                if feature.name in members
            }
            if set(feature.sets) != expected_sets or not feature.source_columns:
                raise FeatureContractError("Feature definition has inconsistent set membership")

    def by_name(self) -> dict[str, FeatureDefinition]:
        return {feature.name: feature for feature in self.features}
