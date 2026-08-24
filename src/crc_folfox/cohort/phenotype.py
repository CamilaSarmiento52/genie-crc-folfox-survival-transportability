"""Locked FOLFOX phenotype definitions and precedence."""

from crc_folfox.cohort.drugs import ALLOWED_BIOLOGICS, FOLFOX_BACKBONE, contains_drug

EXACT_FOLFOX = "Exact FOLFOX"
FOLFOX_PLUS_BIOLOGIC = "FOLFOX plus biologic"
FOLFOX_PLUS_NON_BIOLOGIC = "FOLFOX plus non-biologic agent"
FOLFOXIRI = "FOLFOXIRI (excluded)"
CAPOX = "CAPOX/XELOX (excluded)"
OTHER = "Other first regimen"


def classify_first_regimen(drug_set: frozenset[str]) -> str:
    """Classify a normalized drug set using the frozen precedence order."""
    has_backbone = FOLFOX_BACKBONE.issubset(drug_set)
    has_capecitabine = contains_drug(drug_set, "capecitabine")
    has_irinotecan = contains_drug(drug_set, "irinotecan")
    extras = drug_set - FOLFOX_BACKBONE
    if has_backbone and has_irinotecan:
        return FOLFOXIRI
    if has_capecitabine and contains_drug(drug_set, "oxaliplatin"):
        return CAPOX
    if has_backbone and not extras:
        return EXACT_FOLFOX
    if has_backbone and not has_capecitabine and not has_irinotecan:
        if extras.issubset(ALLOWED_BIOLOGICS):
            return FOLFOX_PLUS_BIOLOGIC
        return FOLFOX_PLUS_NON_BIOLOGIC
    return OTHER


def is_broad_folfox(drug_set: frozenset[str]) -> bool:
    return (
        FOLFOX_BACKBONE.issubset(drug_set)
        and not contains_drug(drug_set, "capecitabine")
        and not contains_drug(drug_set, "irinotecan")
    )
