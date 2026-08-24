"""Manuscript-reporting canonical-input loader: hash-verify every bundle before reading any value
from it.

`manuscript-spec.md` ("Canonical inputs") requires each bundle to be verified against its own
manifest *before* any value is extracted, because a stale or silently overwritten bundle would be
transcribed into the paper and no existing gate would catch it -- prose is the one artifact no test
has ever checked.

Two bundles cannot meet that bar and are handled explicitly rather than waved through: the
exploratory-optimization bundle and the forest-retry bundle publish no `artifact_hashes` in their
manifests,
so there is nothing to verify *against*. For those, this module records a first-observation baseline
and reports them as `baseline_pinned` rather than `verified`. The distinction is preserved all the
way into the build manifest, because "we checked it against its publisher's record" and "we recorded
what it looked like today" are different claims and only one of them is verification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from crc_folfox.report.contracts import BundleVerificationError

VERIFIED = "verified"
BASELINE_PINNED = "baseline_pinned"


@dataclass(frozen=True)
class CanonicalBundle:
    """One frozen input bundle, its manifest, and how its integrity can be established."""

    name: str
    relative_root: str
    manifest_name: str | None
    #: manifest key holding a {relative path -> sha256} map, when the publisher recorded one
    hash_map_key: str | None = None
    #: single-file bundles record their hash under a scalar key instead
    scalar_hash_key: str | None = None
    scalar_hash_target: str | None = None
    supplies: str = ""


#: The eight canonical inputs of `manuscript-spec.md`, plus the data-cleaning workflow dataset that
#: manuscript-reporting workflow
#: computes its two descriptive frames from, plus the revision-era fairness bundle published by
#: `scripts/generate_sex_stratified_performance.py` (absent until that script has been run; its
#: absence is pinned rather than fatal, and only reading from it then fails).
CANONICAL_BUNDLES: tuple[CanonicalBundle, ...] = (
    CanonicalBundle(
        name="phase2_dataset",
        relative_root=".",
        manifest_name="run_manifest.json",
        scalar_hash_key="dataset_sha256",
        scalar_hash_target="analysis_dataset.csv",
        supplies="cohort-flow counts, Table 1 descriptives, missingness by institution",
    ),
    CanonicalBundle(
        name="phase3_preprocessing",
        relative_root="preprocessing",
        manifest_name="preprocessing_manifest.json",
        hash_map_key="artifact_hashes",
        supplies="split definitions, fold-fitted transformation description, feature schema",
    ),
    CanonicalBundle(
        name="phase4_training",
        relative_root="model_training",
        manifest_name="training_manifest.json",
        hash_map_key="artifact_hashes",
        supplies="fit/failure inventory, tuning summary, failure diagnostics",
    ),
    CanonicalBundle(
        name="phase5_evaluation",
        relative_root="evaluation",
        manifest_name="evaluation_manifest.json",
        hash_map_key="artifact_hashes",
        supplies="performance, transportability gap, complexity value, calibration, entry support",
    ),
    CanonicalBundle(
        name="phase5a_correction",
        relative_root="phase5a/correction/evaluation",
        manifest_name="evaluation_manifest.json",
        hash_map_key="artifact_hashes",
        supplies="corrected elastic_net_cox reconciliation",
    ),
    CanonicalBundle(
        name="phase5a_optimization",
        relative_root="phase5a/optimization",
        manifest_name="optimization_development_manifest.json",
        hash_map_key=None,
        supplies="exploratory optimization results (publisher recorded no artifact hashes)",
    ),
    CanonicalBundle(
        name="phase6_explainability",
        relative_root="explainability",
        manifest_name="explainability_manifest.json",
        hash_map_key="artifact_hashes",
        supplies="coefficients, importance, attribution/selection/prediction stability",
    ),
    CanonicalBundle(
        name="phase7_sensitivity",
        relative_root="sensitivity",
        manifest_name="sensitivity_manifest.json",
        hash_map_key="artifact_hashes",
        supplies="six sensitivities and their eight figures",
    ),
    CanonicalBundle(
        name="phase9_forest",
        relative_root="phase9",
        manifest_name="phase9_manifest.json",
        hash_map_key=None,
        supplies="delayed-entry ML comparison (publisher recorded no artifact hashes)",
    ),
    CanonicalBundle(
        name="fairness_subgroups",
        relative_root="fairness",
        manifest_name="fairness_manifest.json",
        hash_map_key="artifact_hashes",
        supplies=(
            "post hoc sex-stratified held-out concordance and sociodemographic subgroup "
            "support (scripts/generate_sex_stratified_performance.py)"
        ),
    ),
)

#: The corrected explainability workflow bundle is the canonical one. Its quarantined predecessor
#: must never be
#: read; if a directory matching this prefix is ever passed as the explainability workflow root, the
#: load fails.
QUARANTINED_EXPLAINABILITY_PREFIX = "explainability.invalidated"


def sha256_of(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks so a large artifact does not load twice."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BundleVerification:
    """The verification outcome for one bundle, carried into the build manifest verbatim."""

    name: str
    root: str
    status: str
    hashes_checked: int
    hashes_matched: int
    baseline_recorded: int = 0
    reason: str = ""
    file_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.hashes_checked == self.hashes_matched


@dataclass
class CanonicalInputs:
    """A verified view over the frozen bundles. Frames are read only through this object."""

    current_root: Path
    verifications: dict[str, BundleVerification]
    _frames: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict, repr=False)

    def bundle_root(self, bundle_name: str) -> Path:
        bundle = _bundle_by_name(bundle_name)
        return (self.current_root / bundle.relative_root).resolve()

    def path(self, bundle_name: str, filename: str) -> Path:
        return self.bundle_root(bundle_name) / filename

    def frame(self, bundle_name: str, filename: str) -> pd.DataFrame:
        """Read one published CSV from a bundle that has already passed verification."""

        key = (bundle_name, filename)
        if key in self._frames:
            return self._frames[key].copy()

        verification = self.verifications.get(bundle_name)
        if verification is None:
            raise BundleVerificationError(
                f"bundle {bundle_name!r} was not verified; Phase 8 reads no unverified input"
            )
        if not verification.ok:
            raise BundleVerificationError(
                f"bundle {bundle_name!r} failed verification "
                f"({verification.hashes_matched}/{verification.hashes_checked} hashes matched); "
                "restore it from analysis/out/archive/ and re-verify before extracting any value"
            )
        if key not in self._frames:
            target = self.path(bundle_name, filename)
            if not target.exists():
                raise BundleVerificationError(f"{target} does not exist in a verified bundle")
            self._frames[key] = pd.read_csv(target)
        return self._frames[key].copy()

    def register_generated_frame(self, filename: str, frame: pd.DataFrame) -> None:
        """Register an aggregate manuscript-reporting workflow frame for token resolution in this
        build only."""

        self._frames[("phase8_generated", filename)] = frame.copy()

    @property
    def all_verified(self) -> bool:
        return all(v.ok for v in self.verifications.values())

    @property
    def unverifiable(self) -> tuple[str, ...]:
        return tuple(name for name, v in self.verifications.items() if v.status == BASELINE_PINNED)


def _bundle_by_name(name: str) -> CanonicalBundle:
    for bundle in CANONICAL_BUNDLES:
        if bundle.name == name:
            return bundle
    raise BundleVerificationError(f"{name!r} is not a canonical Phase 8 input")


def _verify_one(bundle: CanonicalBundle, current_root: Path) -> BundleVerification:
    root = (current_root / bundle.relative_root).resolve()
    if QUARANTINED_EXPLAINABILITY_PREFIX in root.name:
        raise BundleVerificationError(
            f"{root} is the quarantined Phase 6 bundle; the corrected bundle at "
            "analysis/out/current/explainability/ is the only admissible Phase 6 input"
        )
    if not root.exists():
        return BundleVerification(
            name=bundle.name,
            root=str(root),
            status=BASELINE_PINNED,
            hashes_checked=0,
            hashes_matched=0,
            reason="bundle root does not exist",
        )

    manifest: dict = {}
    if bundle.manifest_name:
        manifest_path = root / bundle.manifest_name
        if not manifest_path.exists():
            raise BundleVerificationError(
                f"{manifest_path} is missing; cannot verify {bundle.name}"
            )
        manifest = json.loads(manifest_path.read_text())

    # Case 1 -- the publisher recorded a per-artifact hash map. This is real verification.
    if bundle.hash_map_key and bundle.hash_map_key in manifest:
        recorded: dict[str, str] = manifest[bundle.hash_map_key]
        matched = 0
        observed: dict[str, str] = {}
        mismatches: list[str] = []
        for relative, expected in recorded.items():
            target = root / relative
            if not target.exists():
                mismatches.append(f"missing:{relative}")
                continue
            actual = sha256_of(target)
            observed[relative] = actual
            if actual == expected:
                matched += 1
            else:
                mismatches.append(f"differs:{relative}")
        return BundleVerification(
            name=bundle.name,
            root=str(root),
            status=VERIFIED,
            hashes_checked=len(recorded),
            hashes_matched=matched,
            reason="; ".join(mismatches[:5]),
            file_hashes=observed,
        )

    # Case 2 -- a single-file bundle with a scalar hash key.
    if bundle.scalar_hash_key and bundle.scalar_hash_target:
        expected = manifest.get(bundle.scalar_hash_key)
        target = root / bundle.scalar_hash_target
        if expected is None or not target.exists():
            raise BundleVerificationError(
                f"{bundle.name}: cannot verify {bundle.scalar_hash_target} against "
                f"{bundle.scalar_hash_key}"
            )
        actual = sha256_of(target)
        return BundleVerification(
            name=bundle.name,
            root=str(root),
            status=VERIFIED,
            hashes_checked=1,
            hashes_matched=1 if actual == expected else 0,
            reason="" if actual == expected else f"differs:{bundle.scalar_hash_target}",
            file_hashes={bundle.scalar_hash_target: actual},
        )

    # Case 3 -- the publisher recorded no artifact hashes. Pin a first-observation baseline and say
    # so. This detects drift from now on; it does not establish agreement with publication, and it
    # is never reported as verification.
    observed = {
        item.name: sha256_of(item)
        for item in sorted(root.iterdir())
        if item.is_file() and item.suffix in {".csv", ".json", ".md"}
    }
    return BundleVerification(
        name=bundle.name,
        root=str(root),
        status=BASELINE_PINNED,
        hashes_checked=0,
        hashes_matched=0,
        baseline_recorded=len(observed),
        reason=(
            "publisher recorded no artifact_hashes; a first-observation baseline was pinned, which "
            "detects later drift but does not verify the bundle against its publication"
        ),
        file_hashes=observed,
    )


def verify_canonical_inputs(
    current_root: str | Path, *, bundles: tuple[str, ...] | None = None
) -> CanonicalInputs:
    """Verify every canonical bundle and return the object through which frames may be read.

    `bundles` narrows verification to a named subset. The full build never narrows -- it must prove
    every input it might read -- but a test exercising one artifact should not have to fabricate the
    other eight bundles to do it.
    """

    root = Path(current_root).expanduser().resolve()
    if not root.exists():
        raise BundleVerificationError(f"{root} does not exist")
    selected = (
        CANONICAL_BUNDLES if bundles is None else tuple(_bundle_by_name(name) for name in bundles)
    )
    verifications = {bundle.name: _verify_one(bundle, root) for bundle in selected}
    failed = [v for v in verifications.values() if not v.ok and v.status == VERIFIED]
    if failed:
        detail = "; ".join(f"{v.name} ({v.reason})" for v in failed)
        raise BundleVerificationError(
            "canonical input verification failed before any value was extracted: "
            f"{detail}. Restore from analysis/out/archive/ and re-verify."
        )
    return CanonicalInputs(current_root=root, verifications=verifications)
