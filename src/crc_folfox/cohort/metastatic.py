"""Metastatic-presentation rules for the index colorectal cancer."""

import numpy as np
import pandas as pd


def derive_metastatic_presentation(cancer_index: pd.DataFrame) -> pd.DataFrame:
    """Return an independent frame with frozen presentation labels and day origins."""
    result = cancer_index.copy(deep=True)
    result["is_de_novo"] = result["stage_dx_iv"].eq("Stage IV")
    result["is_recurrent"] = result["stage_dx_iv"].eq("Stage I-III") & result[
        "dmets_stage_i_iii"
    ].eq(1)
    result["metastatic_presentation"] = np.select(
        [result["is_de_novo"], result["is_recurrent"]],
        ["De novo Stage IV", "Recurrent/metachronous"],
        default="Not documented as metastatic",
    )
    result["advanced_origin_day"] = np.select(
        [result["is_de_novo"], result["is_recurrent"]],
        [result["dob_ca_dx_days"], result["dob_ca_dx_days"] + result["dx_to_dmets_days"]],
        default=np.nan,
    )
    return result
