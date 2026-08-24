"""Shared aggregate endpoint result contract."""

from dataclasses import dataclass, field

import pandas as pd


class EndpointContractError(ValueError):
    """A privacy-safe endpoint construction failure."""


@dataclass(frozen=True)
class EndpointResult:
    data: pd.DataFrame
    validation: pd.DataFrame
    by_institution: pd.DataFrame
    event_sources: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def discrepancy_count(self) -> int:
        row = self.validation.loc[self.validation["check"].str.contains("differs")]
        return int(row["patients"].sum())
