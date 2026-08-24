"""Regimen timing, deduplication, and prior-treatment derivations."""

import pandas as pd

from crc_folfox.cohort.drugs import (
    DRUG_COLUMNS,
    DRUG_START_COLUMNS,
    contains_drug,
    regimen_drug_set,
)


def prepare_regimens(regimens: pd.DataFrame) -> pd.DataFrame:
    result = regimens.copy(deep=True)
    result["regimen_origin_day"] = result[list(DRUG_START_COLUMNS)].min(axis=1)
    result["drug_set"] = result[list(DRUG_COLUMNS)].apply(regimen_drug_set, axis=1)
    return result


def first_post_metastatic_regimen(
    regimens: pd.DataFrame, advanced_cancers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cancer_columns = [
        "record_id",
        "ca_seq",
        "advanced_origin_day",
        "metastatic_presentation",
        "age_dx",
        "crc_type",
        "ca_d_site",
        "ca_grade",
        "dmets_abdomen",
        "dmets_bone",
        "dmets_brain",
        "dmets_liver",
        "dmets_pelvis",
        "dmets_thorax",
    ]
    candidates = regimens.merge(
        advanced_cancers[cancer_columns],
        on=["record_id", "ca_seq"],
        how="inner",
        validate="many_to_one",
    )
    post = candidates.loc[
        candidates["regimen_origin_day"].notna()
        & candidates["advanced_origin_day"].notna()
        & candidates["regimen_origin_day"].ge(candidates["advanced_origin_day"])
    ].copy()
    post["advanced_associations_per_physical_regimen"] = post.groupby(
        ["record_id", "regimen_number"]
    )["ca_seq"].transform("nunique")
    first = (
        post.sort_values(["record_id", "regimen_origin_day", "regimen_number", "ca_seq"])
        .drop_duplicates(["record_id", "regimen_number"])
        .drop_duplicates("record_id")
        .copy()
    )
    first["age_at_regimen"] = first["regimen_origin_day"] / 365.25
    return post, first


def add_prior_oxaliplatin(cohort: pd.DataFrame, prepared_regimens: pd.DataFrame) -> pd.DataFrame:
    result = cohort.copy(deep=True)
    physical = (
        prepared_regimens.sort_values(
            ["record_id", "regimen_number", "regimen_origin_day", "ca_seq"]
        )
        .drop_duplicates(["record_id", "regimen_number"])
        .copy()
    )
    physical["has_oxaliplatin"] = physical["drug_set"].map(
        lambda drugs: contains_drug(drugs, "oxaliplatin")
    )
    pairs = result[["record_id", "regimen_origin_day"]].merge(
        physical[["record_id", "regimen_origin_day", "has_oxaliplatin"]],
        on="record_id",
        suffixes=("_folfox", "_other"),
        how="left",
    )
    pairs["prior_oxaliplatin"] = pairs["has_oxaliplatin"].fillna(False) & pairs[
        "regimen_origin_day_other"
    ].lt(pairs["regimen_origin_day_folfox"])
    prior = pairs.groupby("record_id")["prior_oxaliplatin"].any()
    result["prior_oxaliplatin"] = result["record_id"].map(prior).fillna(False)
    return result
