"""Core preprocessing workflow endpoint and partition contracts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crc_folfox.features import FeatureManifest

INSTITUTIONS = ("DFCI", "MSK", "VICC")


class PreprocessingContractError(ValueError):
    """Privacy-safe failure of a preprocessing workflow preprocessing contract."""


@dataclass(frozen=True)
class EndpointDefinition:
    name: str
    eligibility_column: str
    exit_column: str
    event_column: str


ENDPOINTS = {
    "OS": EndpointDefinition("OS", "os_eligible", "os_time", "os_event"),
    "TTNT-D": EndpointDefinition("TTNT-D", "ttnt_d_eligible", "ttnt_d_time", "ttnt_d_event"),
}


@dataclass(frozen=True)
class EndpointView:
    endpoint: str
    data: pd.DataFrame


@dataclass(frozen=True)
class SplitDefinition:
    split_id: str
    endpoint: str
    design: str
    development_keys: tuple[str, ...]
    assessment_keys: tuple[str, ...]
    held_out_institution: str | None = None
    repeat: int | None = None
    fold: int | None = None
    seed: int | None = None

    def frames(self, view: EndpointView) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return caller-independent development and assessment frames."""
        indexed = view.data.set_index("patient_key", drop=False)
        development = indexed.loc[list(self.development_keys)].reset_index(drop=True)
        assessment = indexed.loc[list(self.assessment_keys)].reset_index(drop=True)
        return development.copy(deep=True), assessment.copy(deep=True)

    def aggregate_row(self, view: EndpointView) -> dict[str, object]:
        development, assessment = self.frames(view)
        return {
            "endpoint": self.endpoint,
            "design": self.design,
            "split_id": self.split_id,
            "held_out_institution": self.held_out_institution or "",
            "repeat": self.repeat if self.repeat is not None else "",
            "fold": self.fold if self.fold is not None else "",
            "seed": self.seed if self.seed is not None else "",
            "development_patients": len(development),
            "development_events": int(development["event"].sum()),
            "assessment_patients": len(assessment),
            "assessment_events": int(assessment["event"].sum()),
            "development_institutions": ",".join(
                sorted(development["institution"].astype(str).unique())
            ),
            "assessment_institutions": ",".join(
                sorted(assessment["institution"].astype(str).unique())
            ),
        }


def build_endpoint_view(frame: pd.DataFrame, endpoint: str) -> EndpointView:
    """Select one endpoint population and preserve delayed-entry fields unchanged."""
    if endpoint not in ENDPOINTS:
        raise PreprocessingContractError("Unknown endpoint")
    definition = ENDPOINTS[endpoint]
    manifest = FeatureManifest.load()
    features = list(dict.fromkeys((*manifest.baseline, *manifest.core)))
    required = {
        "patient_key",
        "institution",
        "entry",
        definition.eligibility_column,
        definition.exit_column,
        definition.event_column,
        *features,
    }
    if missing := required - set(frame.columns):
        raise PreprocessingContractError(
            f"Canonical analysis dataset is missing {len(missing)} required columns"
        )
    if frame["patient_key"].isna().any() or frame["patient_key"].duplicated().any():
        raise PreprocessingContractError("Patient keys must be complete and unique")
    observed_institutions = set(frame["institution"].dropna().astype(str))
    if observed_institutions != set(INSTITUTIONS):
        raise PreprocessingContractError("Endpoint source has unknown or missing institutions")

    eligible = frame[definition.eligibility_column]
    if eligible.isna().any() or not eligible.isin([True, False]).all():
        raise PreprocessingContractError("Endpoint eligibility must be complete and boolean")
    selected = frame.loc[
        eligible.astype(bool),
        [
            "patient_key",
            "institution",
            "entry",
            definition.exit_column,
            definition.event_column,
            *features,
        ],
    ].copy()
    selected = selected.rename(
        columns={definition.exit_column: "exit", definition.event_column: "event"}
    )
    structural = selected[["patient_key", "institution", "entry", "exit", "event"]]
    if structural.isna().any().any():
        raise PreprocessingContractError("Endpoint view contains missing structural fields")
    selected["entry"] = pd.to_numeric(selected["entry"], errors="coerce")
    selected["exit"] = pd.to_numeric(selected["exit"], errors="coerce")
    selected["event"] = pd.to_numeric(selected["event"], errors="coerce")
    if selected[["entry", "exit", "event"]].isna().any().any():
        raise PreprocessingContractError("Endpoint view contains invalid numeric fields")
    if not selected["event"].isin([0, 1]).all():
        raise PreprocessingContractError("Endpoint events must be binary")
    if selected["entry"].lt(0).any() or selected["entry"].ge(selected["exit"]).any():
        raise PreprocessingContractError("Endpoint view violates delayed-entry ordering")
    if set(selected["institution"].astype(str)) != set(INSTITUTIONS):
        raise PreprocessingContractError("Eligible endpoint view must contain all institutions")
    selected["event"] = selected["event"].astype(int)
    selected = selected.sort_values("patient_key", kind="stable").reset_index(drop=True)
    return EndpointView(endpoint=endpoint, data=selected)


__all__ = [
    "ENDPOINTS",
    "INSTITUTIONS",
    "EndpointDefinition",
    "EndpointView",
    "PreprocessingContractError",
    "SplitDefinition",
    "build_endpoint_view",
]
