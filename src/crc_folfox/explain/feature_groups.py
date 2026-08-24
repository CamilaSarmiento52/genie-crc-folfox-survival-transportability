"""Group design-matrix columns into source-variable groups (`explainability-spec.md`, "Feature
groups").

Every design column produced by `FittedPreprocessor`/`fit_spline_state`/
`add_presentation_age_spline_interaction` follows one of three naming shapes:

- a bare source-variable name (continuous or binary features, e.g. `age_at_folfox`,
  `prior_oxaliplatin`);
- `{source}__{level}` or `{source}__missing` (one-hot / missing-indicator columns) or
  `{source}__spline_{index}` (the age spline basis) -- the group is the `{source}` prefix; or
- `{presentation_column}__x__{spline_column}` (the one admitted B4 interaction) -- its own group,
  distinct from either source variable, because assigning it to one would misstate both.

Grouping is therefore always "prefix before the first `__`, unless the column contains the
literal `__x__` interaction marker, which gets its own dedicated group name."
"""

from __future__ import annotations

from dataclasses import dataclass

from crc_folfox.explain.contracts import ExplainContractError

ZERO_COEFFICIENT_THRESHOLD = 1e-10

INTERACTION_MARKER = "__x__"
INTERACTION_GROUP_NAME = "metastatic_presentation:age_spline"
_PRESENTATION_PREFIX = "metastatic_presentation__"

# `tech-stack.md` section 6: `number_metastatic_sites` is a group only in the baseline set; the six
# `dmets_*` flags are six groups only in the core set. They never appear together.
BASELINE_GROUPS = (
    "age_at_folfox",
    "metastatic_presentation",
    "ca_d_site",
    "number_metastatic_sites",
)
CORE_GROUPS = (
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

# `explainability-spec.md`, 2026-08-10 amendment: `metastatic_presentation` is required in every
# explained scope (it is in both the baseline and core sets); `prior_oxaliplatin` is required only
# for models using the core set -- `minimal_cox`'s four-variable baseline set never carried it.
ALWAYS_REPORTED_GROUPS = ("metastatic_presentation",)
ALWAYS_REPORTED_GROUPS_CORE_ONLY = ("prior_oxaliplatin",)
CORE_FEATURE_SET_MODELS = (
    "spline_cox",
    "elastic_net_cox",
    "glmnet_ridge_core",
    "glmnet_enet_core",
    "penalized_nonlinear_cox",
    "reduced_core_penalized_cox",
)

# Expected group counts under the registered manifests, per `explainability-spec.md`'s "Feature
# groups" table -- asserted, not assumed, by `assert_group_inventory`.
EXPECTED_GROUP_COUNT = {
    "minimal_cox": 4,
    "spline_cox": 15,
    "elastic_net_cox": 15,
    "glmnet_ridge_core": 15,
    "glmnet_enet_core": 15,
    "reduced_core_penalized_cox": 15,
    "penalized_nonlinear_cox": 16,
}


def group_for_column(column: str) -> str:
    """Map one design-matrix column to its source-variable (or interaction) group name."""
    if INTERACTION_MARKER in column:
        return INTERACTION_GROUP_NAME
    return column.split("__", 1)[0]


def build_feature_groups(columns: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Group an ordered design-matrix column tuple, preserving first-seen group order and the
    original within-group column order (needed so a permutation's shared row index is applied
    consistently to every column of a group)."""
    if not columns:
        raise ExplainContractError("Cannot build feature groups from an empty design matrix")
    groups: dict[str, list[str]] = {}
    for column in columns:
        groups.setdefault(group_for_column(column), []).append(column)
    return {name: tuple(members) for name, members in groups.items()}


def assert_group_inventory(model_name: str, groups: dict[str, tuple[str, ...]]) -> None:
    """Assert the realized group count against the frozen manifest-derived expectation."""
    expected = EXPECTED_GROUP_COUNT.get(model_name)
    if expected is None:
        raise ExplainContractError(f"No expected group count registered for '{model_name}'")
    if len(groups) != expected:
        raise ExplainContractError(
            f"'{model_name}' produced {len(groups)} feature groups, expected {expected}"
        )
    required_groups = ALWAYS_REPORTED_GROUPS
    if model_name in CORE_FEATURE_SET_MODELS:
        required_groups = required_groups + ALWAYS_REPORTED_GROUPS_CORE_ONLY
    for required in required_groups:
        if required not in groups:
            raise ExplainContractError(
                f"'{model_name}' is missing the always-reported group '{required}'"
            )


def dominant_group_coefficient(
    columns: tuple[str, ...], coefficients: dict[str, float]
) -> tuple[str, float] | None:
    """The group's largest-`|coefficient|` column among `columns` that has an entry in
    `coefficients`, ties broken by first-encountered (`columns`' own order). `None` if none of
    `columns` are present in `coefficients` at all.

    Used both for E1's cross-scope dominant-sign selection (`sign_agreement`) and E5's within-scope
    bootstrap sign-stability (both `explainability-spec.md`'s 2026-08-11 amendment) -- the prior
    code instead silently kept "the first design column encountered" in a group, which is wrong for
    every multi-column group (categorical variables, the age spline block, the admitted
    interaction).
    """
    best: tuple[str, float] | None = None
    for column in columns:
        if column not in coefficients:
            continue
        value = coefficients[column]
        if best is None or abs(value) > abs(best[1]):
            best = (column, value)
    return best


@dataclass(frozen=True)
class GroupAlignment:
    feature_group: str
    canonical_columns: tuple[str, ...]
    active_columns: tuple[str, ...]
    group_active: bool  # True iff active_columns is nonempty


def align_groups_to_present_columns(
    canonical_groups: dict[str, tuple[str, ...]], present_columns: tuple[str, ...]
) -> dict[str, GroupAlignment]:
    """Per canonical group, which of its columns exist in `present_columns`.

    A bootstrap refit's own transformed design can drop a canonical group's columns entirely (the
    group was zero-variance in that resample and removed) or partially (a rare level pooled away).
    `group_active=False` exactly when none of the group's canonical columns are present; the caller
    then assigns that refit's importance for the group as exactly `0.0` and still ranks it among
    every other canonical group, rather than dropping it from the ranking
    (`explainability-spec.md`'s 2026-08-11 bootstrap-importance amendment).
    """
    present = set(present_columns)
    return {
        group_name: GroupAlignment(
            feature_group=group_name,
            canonical_columns=columns,
            active_columns=tuple(c for c in columns if c in present),
            group_active=any(c in present for c in columns),
        )
        for group_name, columns in canonical_groups.items()
    }


__all__ = [
    "ALWAYS_REPORTED_GROUPS",
    "ALWAYS_REPORTED_GROUPS_CORE_ONLY",
    "BASELINE_GROUPS",
    "CORE_FEATURE_SET_MODELS",
    "CORE_GROUPS",
    "EXPECTED_GROUP_COUNT",
    "INTERACTION_GROUP_NAME",
    "INTERACTION_MARKER",
    "ZERO_COEFFICIENT_THRESHOLD",
    "GroupAlignment",
    "align_groups_to_present_columns",
    "assert_group_inventory",
    "build_feature_groups",
    "dominant_group_coefficient",
    "group_for_column",
]
