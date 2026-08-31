"""
Training pipeline for the 30-day readmission risk model, with MLflow
experiment tracking — logging not just metrics, but the regulatory
metadata (model version, training data snapshot date, PCCP change type)
that a SaMD deployment needs alongside standard ML tracking.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

np.random.seed(3)
N = 6000

# Simulated de-identified discharge record features
age = np.random.normal(64, 16, N).clip(18, 95)
prior_admissions_12mo = np.random.poisson(1.2, N)
length_of_stay_days = np.random.gamma(2, 2, N).clip(1, 40)
comorbidity_count = np.random.poisson(2.5, N)
discharge_disposition = np.random.choice(
    ["Home", "Home Health", "Skilled Nursing Facility", "Rehab"], N, p=[0.55, 0.20, 0.15, 0.10]
)
admission_type = np.random.choice(["Emergency", "Elective", "Urgent"], N, p=[0.55, 0.25, 0.20])

# Latent readmission risk logic
risk_logit = (
    -3.5 + 0.015 * age + 0.55 * prior_admissions_12mo + 0.04 * length_of_stay_days
    + 0.30 * comorbidity_count
    + np.where(discharge_disposition == "Skilled Nursing Facility", 0.4, 0)
    + np.where(admission_type == "Emergency", 0.3, 0)
)
readmit_prob = 1 / (1 + np.exp(-risk_logit))
readmitted_30d = (np.random.rand(N) < readmit_prob).astype(int)

df = pd.DataFrame({
    "age": age, "prior_admissions_12mo": prior_admissions_12mo,
    "length_of_stay_days": length_of_stay_days, "comorbidity_count": comorbidity_count,
    "discharge_disposition": discharge_disposition, "admission_type": admission_type,
    "readmitted_30d": readmitted_30d,
})
df = pd.get_dummies(df, columns=["discharge_disposition", "admission_type"], drop_first=True)

X = df.drop(columns=["readmitted_30d"])
y = df.readmitted_30d
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)

mlflow.set_experiment("readmission-risk-samd")

with mlflow.start_run(run_name="v1_xgboost_baseline"):
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss")
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    # Standard ML tracking
    mlflow.log_params({"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05})
    mlflow.log_metric("auc_roc", auc)

    # Regulatory metadata — tags that wouldn't exist in a non-SaMD MLOps run
    mlflow.set_tags({
        "samd_classification": "EU_MDR_Rule11_ClassIIa",
        "eu_ai_act_risk_tier": "high_risk_annex_iii",
        "training_data_snapshot": "simulated_2026-08-01",
        "pccp_change_type": "in_scope_performance_preserving_retrain",
        "human_oversight_required": "true",
    })
    mlflow.xgboost.log_model(model, "model")

    print(f"AUC-ROC: {auc:.3f}")
    print("Run logged with SaMD regulatory tags — see MLflow UI for full metadata")
