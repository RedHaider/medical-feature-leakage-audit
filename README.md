# Medical Feature Leakage Audit

Synthetic trustworthy-AI benchmark that audits whether learned medical-image
features encode a protected attribute. It reports disease-prediction AUROC,
protected-attribute probe AUROC, and the effect of training-time residualization.
No patient data are used.

Run: `python -m leakage_audit.experiment --output outputs/demo`
