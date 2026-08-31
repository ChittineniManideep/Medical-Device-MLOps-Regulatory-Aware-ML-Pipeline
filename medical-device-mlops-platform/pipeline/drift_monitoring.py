"""
Drift monitoring — framed against the FDA PCCP performance-monitoring
component: drift beyond a pre-specified threshold is a defined trigger
for review, not just a dashboard alert.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# PCCP-style pre-specified monitoring thresholds — defined BEFORE deployment,
# per FDA PCCP guidance, not decided reactively after drift is observed
DRIFT_THRESHOLDS = {
    "feature_ks_pvalue_min": 0.05,      # below this, feature distribution has meaningfully shifted
    "auc_degradation_max_pct": 5.0,     # >5% relative AUC drop triggers review
}

def check_feature_drift(baseline: pd.Series, current: pd.Series, feature_name: str):
    stat, p_value = ks_2samp(baseline, current)
    drifted = p_value < DRIFT_THRESHOLDS["feature_ks_pvalue_min"]
    return {
        "feature": feature_name,
        "ks_statistic": round(stat, 4),
        "p_value": round(p_value, 4),
        "drift_detected": drifted,
        "action_required": "Escalate for PCCP review" if drifted else "None",
    }

def check_performance_drift(baseline_auc: float, current_auc: float):
    pct_change = (baseline_auc - current_auc) / baseline_auc * 100
    breach = pct_change > DRIFT_THRESHOLDS["auc_degradation_max_pct"]
    return {
        "baseline_auc": baseline_auc,
        "current_auc": current_auc,
        "pct_degradation": round(pct_change, 2),
        "threshold_breached": breach,
        "action_required": "Trigger PCCP-scoped retrain review" if breach else "None — within monitored performance band",
    }

if __name__ == "__main__":
    np.random.seed(9)
    baseline_age = np.random.normal(64, 16, 3000)
    current_age = np.random.normal(67, 17, 3000)  # simulated mild population shift, e.g. seasonal case-mix change

    result = check_feature_drift(pd.Series(baseline_age), pd.Series(current_age), "age")
    perf_result = check_performance_drift(baseline_auc=0.715, current_auc=0.693)

    print("Feature drift check:", result)
    print("Performance drift check:", perf_result)
