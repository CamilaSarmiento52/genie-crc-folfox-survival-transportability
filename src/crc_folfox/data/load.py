"""Boundary loader for the restricted analytic release."""

from pathlib import Path

import pandas as pd

from crc_folfox.data.contracts import (
    TABLE_SPECS,
    ReleaseContractError,
    ReleaseTables,
)
from crc_folfox.data.integrity import validate_manifest

STRING_DTYPES = {"cohort": "string", "record_id": "string", "institution": "string"}


def load_release(root: Path, *, check_integrity: bool = True) -> ReleaseTables:
    """Load independent frames after validating files, columns, and boundary dtypes."""
    root = Path(root)
    if check_integrity:
        validate_manifest(root)

    frames: dict[str, pd.DataFrame] = {}
    for name, spec in TABLE_SPECS.items():
        path = root / spec.filename
        if not path.is_file():
            raise ReleaseContractError(f"Required analytic table is missing: {name}")
        frame = pd.read_csv(path, dtype=STRING_DTYPES, low_memory=False)
        missing = set(spec.required_columns) - set(frame.columns)
        if missing:
            raise ReleaseContractError(
                f"Analytic table {name} is missing {len(missing)} required columns"
            )
        for column in STRING_DTYPES:
            if not isinstance(frame[column].dtype, pd.StringDtype):
                raise ReleaseContractError(f"Analytic table {name} has invalid dtype: {column}")
        for column in spec.numeric:
            original_missing = frame[column].isna()
            converted = pd.to_numeric(frame[column], errors="coerce")
            invalid = int((~original_missing & converted.isna()).sum())
            if invalid:
                raise ReleaseContractError(
                    f"Analytic table {name} has {invalid} nonnumeric values in column {column}"
                )
            frame[column] = converted
        frames[name] = frame
    return ReleaseTables(frames)
