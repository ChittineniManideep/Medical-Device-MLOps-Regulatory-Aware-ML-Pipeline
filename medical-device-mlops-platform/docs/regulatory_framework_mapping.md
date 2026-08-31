# Regulatory Framework Mapping — Why This Isn't Just a Standard ML Deployment

## 1. Is this Software as a Medical Device (SaMD)?

A 30-day hospital readmission risk model that informs a clinical decision (e.g., discharge planning, follow-up intensity) meets both frameworks' working definition of SaMD:

- **FDA:** Software intended for one or more medical purposes, without being part of a hardware device — this model's output directly informs a clinical management decision, which places it in scope.
- **EU MDR (Regulation 2017/745):** Software qualifying as a medical device under Rule 11 if it provides information used to make decisions with diagnostic or therapeutic purposes.

**Practical consequence:** this is not "deploy a model and monitor accuracy" — it requires a documented risk classification, a model card, change control, and an audit trail *before* production deployment, not as an afterthought.

## 2. EU AI Act risk tier

Under the EU AI Act, AI systems used as a safety component of a medical device (or that are themselves a medical device under MDR) are classified **high-risk** (Annex III / Article 6(1) linkage to MDR). High-risk classification triggers:

- A documented risk management system (not just model validation metrics)
- Data governance requirements on training data quality and representativeness
- Human oversight design (the model must support, not replace, clinical judgement — reflected in `docs/model_card.md`'s "intended use" section)
- Logging/traceability requirements — this is what `governance/audit_trail_log.csv` demonstrates

## 3. GDPR — Article 9 special category data

Patient health data is "special category" personal data under GDPR Article 9, requiring a specific lawful basis beyond the general Article 6 bases (typically explicit consent or a substantial public interest basis for healthcare providers). This changes the data governance approach from a standard Purview classification exercise:

- Health data fields are tagged as a distinct, higher-sensitivity classification tier in `governance/data_classification_catalog.yaml`, not lumped in with general PII
- Access to training data is restricted beyond standard RBAC — clinical data science roles only, not general analytics access

## 4. FDA Predetermined Change Control Plan (PCCP) — relevance to MLOps

FDA's 2023 PCCP guidance allows a SaMD manufacturer to pre-specify what future model updates (retraining on new data, performance-preserving changes) can be made *without* a new regulatory submission, provided the change protocol and acceptance criteria are documented upfront. This is directly relevant to an MLOps pipeline:

- `governance/audit_trail_log.csv` is structured to log every retrain/deployment event against a predefined change type (in-scope-of-PCCP vs. requiring new review) — the kind of discipline a standard MLOps audit log doesn't need to think about
- `pipeline/drift_monitoring.py` monitoring thresholds are framed as the PCCP's "performance monitoring" component — drift beyond a defined threshold is a trigger for review, not just an alert

## 5. What this project does not claim

This project does not claim clinical validation, IRB/ethics approval processes, or actual regulatory submission experience — those are real, substantial undertakings that a portfolio project cannot simulate meaningfully. What it demonstrates is that the regulatory *shape* of the problem — why a healthcare ML pipeline needs different governance than a standard one — is understood and reflected in the pipeline design.
