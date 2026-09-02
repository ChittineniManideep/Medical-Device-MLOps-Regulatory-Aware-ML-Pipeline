# Medical Device MLOps — Regulatory-Aware ML Pipeline
### A clinical decision-support model, deployed with the MLOps stack already in production use, wrapped in the regulatory layer specific to Software as a Medical Device (SaMD)


## The simulated use case

A **30-day hospital readmission risk model** — a common, well-understood clinical decision-support use case, and a realistic example of Software as a Medical Device (SaMD) under both FDA and EU MDR frameworks, since it directly informs a clinical decision.

## What's here

| Area | File |
|---|---|
| Regulatory classification (why this model is SaMD, its risk tier) | `docs/regulatory_framework_mapping.md` |
| Model card (FDA/EU AI Act transparency documentation standard) | `docs/model_card.md` |
| Training pipeline + CI/CD | `pipeline/train_model.py`, `pipeline/.github_workflow_ci.yml` |
| Containerised deployment | `pipeline/Dockerfile` |
| Drift monitoring | `pipeline/drift_monitoring.py` |
| Health data classification catalogue (Purview-style, PHI-tagged) | `governance/data_classification_catalog.yaml` |
| Audit trail (algorithm change log — FDA PCCP-style) | `governance/audit_trail_log.csv` |

## Repo structure

```
medical-device-mlops-platform/
├── README.md
├── pipeline/
│   ├── train_model.py
│   ├── drift_monitoring.py
│   ├── Dockerfile
│   └── ci_workflow.yml
├── governance/
│   ├── data_classification_catalog.yaml
│   └── audit_trail_log.csv
└── docs/
    ├── regulatory_framework_mapping.md
    └── model_card.md
```
