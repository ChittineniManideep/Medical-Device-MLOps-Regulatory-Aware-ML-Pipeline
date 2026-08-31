# Model Card: 30-Day Hospital Readmission Risk Model

*Format follows the FDA/EU AI Act transparency documentation convention — the standard structure for documenting a clinical decision-support model's intended use, performance, and limitations.*

## Intended use

**Primary intended use:** Flag patients at elevated risk of unplanned readmission within 30 days of discharge, to support (not replace) clinical discharge-planning decisions — e.g., prioritising follow-up call scheduling or care coordination resources.

**Intended users:** Discharge planning / care coordination clinical staff, reviewed alongside standard clinical judgement — not intended as a standalone diagnostic or automated decision tool.

**Out of scope:** Not validated for use as a sole determinant of discharge timing, insurance/resource-denial decisions, or any use case beyond risk-flagging for care-coordination prioritisation.

## Model details

- **Model type:** Gradient-boosted tree classifier (XGBoost)
- **Training data:** Simulated de-identified discharge records — demographics, prior admission history, length of stay, comorbidity count, discharge disposition
- **Output:** Risk score (0-1) + risk tier (Low/Medium/High), not a binary readmit/no-readmit prediction — deliberately framed as a triage signal, not a diagnosis

## Performance

| Metric | Value (simulated validation set) |
|---|---|
| AUC-ROC | 0.78 |
| Sensitivity at High-risk threshold | 0.71 |
| Specificity at High-risk threshold | 0.74 |
| Calibration | Reviewed via reliability diagram — predicted probabilities tracked against observed readmission rate by decile |

## Fairness & subgroup performance

Performance reviewed across simulated age bands and admission type — flagged as a required check under EU AI Act high-risk data governance requirements, not an optional nice-to-have. (In a real deployment this would extend to protected characteristics per the organisation's actual patient population and applicable anti-discrimination requirements.)

## Limitations

- Trained on simulated data — real deployment requires validation on the target hospital system's actual population, which may differ materially in case mix
- Does not account for post-discharge factors outside the EHR (e.g., social determinants of health, home support availability) unless those fields are explicitly captured upstream
- Model requires periodic retraining as care pathways and patient population shift — governed under the change control process in `governance/audit_trail_log.csv`

## Human oversight

Output is a decision-support signal reviewed by discharge planning staff, not an automated action trigger — consistent with the EU AI Act's human oversight requirement for high-risk systems.
