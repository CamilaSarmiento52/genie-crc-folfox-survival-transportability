"""Manifest-driven, fold-fitted preprocessing workflow preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from crc_folfox.features import FeatureDefinition, FeatureManifest
from crc_folfox.preprocessing.contracts import PreprocessingContractError

RARE_LEVEL = "__RARE__"
RARE_MIN_COUNT = 10
MISSING_INDICATORS = {"ca_grade"}


@dataclass(frozen=True)
class TransformResult:
    matrix: pd.DataFrame
    metadata: pd.DataFrame
    unseen_level_counts: dict[str, int]


@dataclass(frozen=True)
class FittedPreprocessor:
    endpoint: str
    feature_set: str
    fit_scope: str
    feature_names: tuple[str, ...]
    definitions: tuple[FeatureDefinition, ...]
    imputation_values: dict[str, str]
    missing_indicators: tuple[str, ...]
    references: dict[str, str]
    level_maps: dict[str, dict[str, str]]
    encoded_levels: dict[str, tuple[str, ...]]
    scaling: dict[str, dict[str, float]]
    raw_missing_counts: dict[str, int]
    raw_category_counts: dict[str, dict[str, int]]
    all_output_columns: tuple[str, ...]
    removed_zero_variance: tuple[str, ...]
    output_columns: tuple[str, ...]
    fit_patient_keys: frozenset[str]

    @classmethod
    def fit(
        cls,
        training: pd.DataFrame,
        *,
        endpoint: str,
        feature_set: str,
        fit_scope: str,
        manifest: FeatureManifest | None = None,
    ) -> FittedPreprocessor:
        manifest = manifest or FeatureManifest.load()
        if feature_set not in {"baseline", "core"}:
            raise PreprocessingContractError("Unknown feature set")
        feature_names = tuple(getattr(manifest, feature_set))
        definitions_by_name = manifest.by_name()
        definitions = tuple(definitions_by_name[name] for name in feature_names)
        required = {"patient_key", "institution", "entry", "exit", "event", *feature_names}
        if missing := required - set(training.columns):
            raise PreprocessingContractError(
                f"Training frame is missing {len(missing)} required columns"
            )
        if training.empty or training["patient_key"].duplicated().any():
            raise PreprocessingContractError("Training rows must be nonempty and patient-unique")

        imputation_values: dict[str, str] = {}
        references: dict[str, str] = {}
        level_maps: dict[str, dict[str, str]] = {}
        encoded_levels: dict[str, tuple[str, ...]] = {}
        scaling: dict[str, dict[str, float]] = {}
        raw_missing_counts: dict[str, int] = {}
        raw_category_counts: dict[str, dict[str, int]] = {}
        all_columns: list[str] = []
        indicators: list[str] = []

        for definition in definitions:
            name = definition.name
            raw_missing_counts[name] = int(training[name].isna().sum())
            if _is_categorical(definition):
                values = _normalized_strings(training[name], name=name, allow_missing=True)
                raw_category_counts[name] = {
                    str(level): int(count)
                    for level, count in values.fillna("<MISSING>")
                    .value_counts()
                    .sort_index()
                    .items()
                }
                if definition.missing_allowed:
                    mode = _deterministic_mode(values.dropna(), name=name)
                    imputation_values[name] = mode
                    values = values.fillna(mode)
                elif values.isna().any():
                    raise PreprocessingContractError(
                        "Unexpected missingness in a nonmissing predictor"
                    )
                if name in MISSING_INDICATORS:
                    indicator = f"{name}__missing"
                    indicators.append(indicator)
                    all_columns.append(indicator)

                counts = values.value_counts()
                if definition.reference_rule == "training_mode_then_lexical":
                    reference = _deterministic_mode(values, name=name)
                elif definition.reference_rule not in {"none", "0", "false"}:
                    reference = definition.reference_rule
                else:
                    raise PreprocessingContractError("Categorical reference rule is invalid")
                if int(counts.get(reference, 0)) < RARE_MIN_COUNT:
                    raise PreprocessingContractError(
                        "Declared categorical reference lacks required training support"
                    )
                references[name] = reference
                rare = {
                    str(level) for level, count in counts.items() if int(count) < RARE_MIN_COUNT
                }
                rare.discard(reference)
                mapping = {
                    str(level): RARE_LEVEL if str(level) in rare else str(level)
                    for level in counts.index
                }
                level_maps[name] = mapping
                pooled = values.astype(str).map(mapping)
                levels = tuple(sorted(set(pooled.astype(str))))
                encoded_levels[name] = levels
                all_columns.extend(f"{name}__{level}" for level in levels if level != reference)
            elif definition.role == "continuous":
                values = _numeric(training[name], name=name, allow_missing=False)
                mean = float(values.mean())
                std = float(values.std(ddof=0))
                scaling[name] = {"mean": mean, "population_std": std}
                all_columns.append(name)
            else:
                _binary(training[name], name=name)
                all_columns.append(name)

        preliminary = cls(
            endpoint=endpoint,
            feature_set=feature_set,
            fit_scope=fit_scope,
            feature_names=feature_names,
            definitions=definitions,
            imputation_values=imputation_values,
            missing_indicators=tuple(indicators),
            references=references,
            level_maps=level_maps,
            encoded_levels=encoded_levels,
            scaling=scaling,
            raw_missing_counts=raw_missing_counts,
            raw_category_counts=raw_category_counts,
            all_output_columns=tuple(all_columns),
            removed_zero_variance=(),
            output_columns=tuple(all_columns),
            fit_patient_keys=frozenset(training["patient_key"].astype(str)),
        )
        encoded, _ = preliminary._encode(training)
        removed = tuple(
            column for column in encoded.columns if encoded[column].nunique(dropna=False) <= 1
        )
        output = tuple(column for column in all_columns if column not in set(removed))
        if not output:
            raise PreprocessingContractError("Preprocessing removed every design column")
        fitted = replace(
            preliminary,
            removed_zero_variance=removed,
            output_columns=output,
        )
        fitted.transform(training)
        return fitted

    def transform(self, frame: pd.DataFrame) -> TransformResult:
        """Apply frozen training transformations without learning from assessment rows."""
        required = {"patient_key", "institution", "entry", "exit", "event", *self.feature_names}
        if missing := required - set(frame.columns):
            raise PreprocessingContractError(
                f"Transform frame is missing {len(missing)} required columns"
            )
        encoded, unseen = self._encode(frame)
        matrix = encoded.loc[:, list(self.output_columns)].copy()
        numeric = matrix.to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise PreprocessingContractError("Transformed matrix contains nonfinite values")
        metadata = frame.loc[:, ["patient_key", "institution", "entry", "exit", "event"]].copy(
            deep=True
        )
        return TransformResult(matrix=matrix, metadata=metadata, unseen_level_counts=unseen)

    def aggregate_state(self) -> dict[str, object]:
        """Return serializable learned state without patient keys or rows."""
        return {
            "endpoint": self.endpoint,
            "feature_set": self.feature_set,
            "fit_scope": self.fit_scope,
            "training_patients": len(self.fit_patient_keys),
            "input_features": list(self.feature_names),
            "output_columns": list(self.output_columns),
            "removed_zero_variance": list(self.removed_zero_variance),
            "imputation_values": self.imputation_values,
            "missing_indicators": list(self.missing_indicators),
            "references": self.references,
            "level_maps": self.level_maps,
            "encoded_levels": {name: list(levels) for name, levels in self.encoded_levels.items()},
            "scaling": self.scaling,
            "raw_missing_counts": self.raw_missing_counts,
            "raw_category_counts": self.raw_category_counts,
        }

    def _encode(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        columns: dict[str, pd.Series] = {}
        unseen_counts: dict[str, int] = {}
        for definition in self.definitions:
            name = definition.name
            if _is_categorical(definition):
                raw = _normalized_strings(frame[name], name=name, allow_missing=True)
                if f"{name}__missing" in self.missing_indicators:
                    columns[f"{name}__missing"] = raw.isna().astype(float)
                if raw.isna().any():
                    if name not in self.imputation_values:
                        raise PreprocessingContractError(
                            "Transform has missingness without a fitted imputation"
                        )
                    raw = raw.fillna(self.imputation_values[name])
                mapping = self.level_maps[name]
                supported_rare = RARE_LEVEL in self.encoded_levels[name]
                unseen_mask = ~raw.astype(str).isin(mapping)
                unseen_counts[name] = int(unseen_mask.sum())
                mapped = raw.astype(str).map(mapping)
                fallback = RARE_LEVEL if supported_rare else self.references[name]
                mapped = mapped.fillna(fallback)
                for level in self.encoded_levels[name]:
                    if level != self.references[name]:
                        columns[f"{name}__{level}"] = mapped.eq(level).astype(float)
            elif definition.role == "continuous":
                values = _numeric(frame[name], name=name, allow_missing=False)
                parameters = self.scaling[name]
                std = parameters["population_std"]
                if std == 0:
                    columns[name] = pd.Series(0.0, index=frame.index)
                else:
                    columns[name] = (values - parameters["mean"]) / std
            else:
                columns[name] = _binary(frame[name], name=name).astype(float)
        encoded = pd.DataFrame(columns, index=frame.index)
        if tuple(encoded.columns) != self.all_output_columns:
            raise PreprocessingContractError("Encoded column order differs from fitted schema")
        return encoded, unseen_counts


def _is_categorical(definition: FeatureDefinition) -> bool:
    return definition.role == "categorical" or definition.encoding == "training_only_one_hot"


def _normalized_strings(series: pd.Series, *, name: str, allow_missing: bool) -> pd.Series:
    values = series.astype("string")
    present = values.dropna()
    if present.str.strip().eq("").any() or present.ne(present.str.strip()).any():
        raise PreprocessingContractError(f"Categorical predictor {name} is not normalized")
    if not allow_missing and values.isna().any():
        raise PreprocessingContractError("Unexpected missingness in a nonmissing predictor")
    return values


def _deterministic_mode(values: pd.Series, *, name: str) -> str:
    if values.empty:
        raise PreprocessingContractError(f"Categorical predictor {name} has no observed values")
    counts = values.astype(str).value_counts()
    maximum = int(counts.max())
    return sorted(str(level) for level, count in counts.items() if int(count) == maximum)[0]


def _numeric(series: pd.Series, *, name: str, allow_missing: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if not allow_missing and values.isna().any():
        raise PreprocessingContractError(f"Numeric predictor {name} is missing or invalid")
    return values


def _binary(series: pd.Series, *, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise PreprocessingContractError(f"Binary predictor {name} must contain only 0/1")
    return values.astype(int)


__all__ = [
    "MISSING_INDICATORS",
    "RARE_LEVEL",
    "RARE_MIN_COUNT",
    "FittedPreprocessor",
    "TransformResult",
]
