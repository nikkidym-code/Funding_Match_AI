"""
startup_risk_tool.py (v2.1)
==========================
Fix: Align the `check_eligibility` function signature with the Pydantic
schema so that **CheckEligibilityTool** works without a TypeError.

* renamed positional arg from **features** → **company_info**
* adjusted the body to reference the new name
* kept everything else unchanged (including `as_langchain_tool` if you
  still want it for other contexts)
"""
from __future__ import annotations

import json
import pathlib
import warnings
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import shap
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).resolve().parent
MODEL_PATH = HERE / "risk_model.joblib"
FEATURES_PATH = HERE / "model_feature_names.joblib"

_model = None            # sklearn model, lazy-loaded
_feature_names: List[str] | None = None
_explainer = None        # shap.Explainer, lazy-loaded

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_eligibility(company_info: Dict[str, Any]) -> Dict[str, Any]:
    """Predict shutdown probability + SHAP drivers for a single company.

    Parameters
    ----------
    company_info : dict
        Must contain the five raw fields used during model training:
        `country_code`, `category_list`, `founded_at`, `funding_total_usd`,
        `funding_rounds`.
    """
    _lazy_load()

    X_processed = _preprocess_single(company_info)
    prob = float(_model.predict_proba(X_processed)[:, 1][0])

    # SHAP explanation (best-effort — silently skips on unsupported models)
    try:
        _ensure_explainer()
        shap_values = _explainer(X_processed)
        contrib = shap_values.values[0]
        df = (
            pd.DataFrame({
                "feature": _feature_names,
                "value": X_processed.iloc[0].values,
                "contribution": contrib,
            })
            .assign(abs_contrib=lambda d: d.contribution.abs())
            .sort_values("abs_contrib", ascending=False)
            .head(10)
            .drop(columns="abs_contrib")
        )
        top_drivers = df.to_dict(orient="records")
    except Exception:
        top_drivers = []
    print("probability",round(prob, 6), "top_drivers",top_drivers)
    return {"probability": round(prob, 6), "top_drivers": top_drivers}

# ---------------------------------------------------------------------------
# LangChain helper (optional)
# ---------------------------------------------------------------------------

def as_langchain_tool():  # pragma: no cover
    from langchain.tools import Tool

    def _wrapped(json_input: str) -> str:
        data = json.loads(json_input)
        result = check_eligibility(data)
        return json.dumps(result)

    return Tool(
        name="startup_risk_score",
        description=(
            "Assess shutdown probability for a startup based on country, "
            "category list, founding date, total funding, and round count. "
            "Returns JSON with probability + SHAP driver list."),
        func=_wrapped,
    )

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lazy_load():
    global _model, _feature_names
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _feature_names is None:
        _feature_names = joblib.load(FEATURES_PATH)


def _ensure_explainer():
    global _explainer
    if _explainer is None:
        _explainer = shap.Explainer(_model)  # let shap decide best explainer


def _preprocess_single(raw: Dict[str, Any]) -> pd.DataFrame:
    """Replicate the exact feature pipeline from `StartupRiskPredictor`."""
    data = dict(raw)  # shallow copy so we don't mutate caller input

    # founded_year
    founded_at = pd.to_datetime(data.get("founded_at"))
    data["founded_year"] = int(founded_at.year) if not pd.isna(founded_at) else 2010

    # funding_total_usd
    ft = str(data.get("funding_total_usd", "0"))
    ft_clean = ft.replace("$", "").replace(",", "")
    try:
        funding_val = float(ft_clean)
    except ValueError:
        funding_val = np.nan
    if np.isnan(funding_val):
        funding_val = 1_000_000.0
    data["funding_total_usd"] = funding_val

    # funding_rounds
    fr = data.get("funding_rounds", 0)
    try:
        data["funding_rounds"] = int(fr)
    except Exception:
        data["funding_rounds"] = 0

    # main_category
    clist = str(data.get("category_list", "Software"))
    data["main_category"] = clist.split("|")[0] if "|" in clist else clist

    # country_grouped
    country = data.get("country_code", "Other")
    top_countries = {"USA", "GBR", "CAN", "IND", "DEU", "FRA", "ISR", "CHN", "AUS", "SWE"}
    data["country_grouped"] = country if country in top_countries else "Other"

    # one-hot encode categoricals
    df_cat = pd.get_dummies(pd.DataFrame([data]),
                            columns=["country_grouped", "main_category"],
                            prefix=["country", "category"])

    df_num = pd.DataFrame([{k: data[k] for k in ["founded_year", "funding_total_usd", "funding_rounds"]}])
    df_final = pd.concat([df_num, df_cat], axis=1)

    # ensure full feature set & order
    for col in _feature_names:
        if col not in df_final:
            df_final[col] = 0.0
    df_final = df_final[_feature_names].astype(float)
    return df_final

# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------
# if __name__ == "__main__":  # pragma: no cover
#     demo = {
#         "country_code": "USA",
#         "category_list": "Software|SaaS",
#         "founded_at": "2020-01-15",
#         "funding_total_usd": "$3,500,000",
#         "funding_rounds": 1,
#     }
#     print(json.dumps(check_eligibility(demo), indent=2))

# ---------------------------------------------------------------------------
# Pydantic input schema & StructuredTool wrapper (kept for backward-compat)
# ---------------------------------------------------------------------------
class EligibilityQuery(BaseModel):
    company_info: Dict = Field(description="Company information (5 raw fields)")

CheckEligibilityTool = StructuredTool.from_function(
    name="CheckEligibilityTool",  # keep original name for external references
    description="Startup risk assessment returning probability + SHAP drivers",
    func=check_eligibility,
    args_schema=EligibilityQuery,
)