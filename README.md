# Medical Feature Leakage Audit

Synthetic trustworthy-AI benchmark that audits whether learned medical-image
features encode a protected attribute. It reports disease-prediction AUROC,
protected-attribute probe AUROC, and the effect of training-time residualization.
No patient data are used.

Run: `python -m leakage_audit.experiment --output outputs/demo`

## Heavy audit

The expanded audit simulates mean, variance, and nonlinear protected-attribute
leakage. Across five independent seeds it compares the original representation,
a single-direction projection, and group-conditioned moment matching. Both a
linear probe and a nonlinear gradient-boosting probe are used so a mitigation
cannot appear successful merely by hiding information from a linear auditor.
Disease AUROC, Brier score, equal-opportunity gap, probe AUROC, and 95%
confidence intervals are reported.

```powershell
python -m leakage_audit.heavy_experiment --output outputs/heavy
```

This is a synthetic representation audit, not a clinical fairness validation.
Moment matching also requires the protected group during transformation, an
important deployment limitation. The project does not reproduce or claim to
improve a professor's published method.

### Verified result

Across five seeds, group-conditioned moment matching reduced the strongest
protected-attribute probe AUROC from **0.942** to **0.556** (−0.387), while
disease AUROC changed from **0.80469** to **0.80441** (−0.00028). A key audit
finding is that single-direction projection made the linear probe look nearly
random (0.514) while a nonlinear probe still achieved 0.877. This demonstrates
why a multi-probe audit is necessary. Equal-opportunity gap did not materially
improve, so representation leakage reduction is not presented as proof of
outcome fairness. Confidence intervals are in `results/heavy_metrics.csv`.
