"""Manifest and payload integrity checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crc_folfox.data.contracts import ReleaseContractError

MANIFEST_COLUMNS = {"name", "dataFileMD5Hex", "dataFileSizeBytes"}


@dataclass(frozen=True)
class IntegrityReport:
    local_files: int
    manifest_payloads: int
    matched_payloads: int
    name_matches: int
    md5_matches: int

    @property
    def passed(self) -> bool:
        return self.manifest_payloads == self.matched_payloads


def calculate_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(root: Path) -> IntegrityReport:
    """Validate complete one-to-one manifest coverage by name or content hash."""
    root = Path(root)
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        raise ReleaseContractError("Release manifest is missing")

    manifest = pd.read_csv(manifest_path, low_memory=False)
    missing_columns = MANIFEST_COLUMNS - set(manifest.columns)
    if missing_columns:
        raise ReleaseContractError(f"Manifest is missing {len(missing_columns)} required columns")
    if (
        manifest["dataFileMD5Hex"].isna().any()
        or manifest["dataFileMD5Hex"].duplicated().any()
        or manifest["name"].isna().any()
        or manifest["name"].duplicated().any()
    ):
        raise ReleaseContractError("Manifest names and hashes must be complete and unique")

    local_paths = sorted(path for path in root.iterdir() if path.is_file())
    payload_paths = [path for path in local_paths if path.name != "manifest.csv"]
    hashes = {path: calculate_md5(path) for path in payload_paths}
    paths_by_hash: dict[str, list[Path]] = {}
    for path, digest in hashes.items():
        paths_by_hash.setdefault(digest, []).append(path)

    used: set[Path] = set()
    name_matches = 0
    md5_matches = 0
    for row in manifest.itertuples(index=False):
        expected_name = str(row.name)
        expected_hash = str(row.dataFileMD5Hex).lower()
        named_path = root / expected_name
        matched: Path | None = None
        method = ""
        if named_path in hashes and hashes[named_path].lower() == expected_hash:
            matched = named_path
            method = "name"
        else:
            candidates = [path for path in paths_by_hash.get(expected_hash, []) if path not in used]
            if len(candidates) == 1:
                matched = candidates[0]
                method = "md5"
        if matched is None or matched in used:
            raise ReleaseContractError("Manifest payload matching failed for 1 or more entries")
        if matched.stat().st_size != int(row.dataFileSizeBytes):
            raise ReleaseContractError("Manifest payload size mismatch detected")
        used.add(matched)
        name_matches += method == "name"
        md5_matches += method == "md5"

    unmatched = len(set(payload_paths) - used)
    if unmatched:
        raise ReleaseContractError(f"Release contains {unmatched} unmatched payload files")
    return IntegrityReport(
        local_files=len(local_paths),
        manifest_payloads=len(manifest),
        matched_payloads=len(used),
        name_matches=name_matches,
        md5_matches=md5_matches,
    )
