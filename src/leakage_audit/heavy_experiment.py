from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = (27, 41, 73, 101, 137)


def make_audit_dataset(n=3600, seed=27):
    rng = np.random.default_rng(seed)
    protected = rng.binomial(1, .5, n)
    x = rng.normal(size=(n, 12))
    x[:, :3] += protected[:, None] * .85
    x[:, 3:5] *= np.where(protected[:, None] == 1, 1.75, .65)
    x[:, 5] += protected * np.maximum(x[:, 6], 0) * .8
    logit = 1.0*x[:, 7] - .8*x[:, 8] + .55*x[:, 9] + .22*protected - .9
    disease = rng.binomial(1, 1/(1+np.exp(-logit)))
    return x, disease, protected


def single_direction(train, test, protected):
    direction = train[protected == 1].mean(0) - train[protected == 0].mean(0)
    direction /= np.linalg.norm(direction) + 1e-9
    return train - np.outer(train @ direction, direction), test - np.outer(test @ direction, direction)


def moment_match(train, test, train_a, test_a):
    means = np.array([train[train_a == v].mean(0) for v in (0, 1)])
    stds = np.array([train[train_a == v].std(0) + 1e-6 for v in (0, 1)])
    pooled_mean, pooled_std = train.mean(0), train.std(0) + 1e-6
    def transform(x, a):
        return (x - means[a]) / stds[a] * pooled_std + pooled_mean
    return transform(train, train_a), transform(test, test_a)


def _equality_gap(y, probability, protected):
    pred = probability >= .5
    rates = []
    for group in (0, 1):
        positive = (protected == group) & (y == 1)
        rates.append(float(pred[positive].mean()))
    return abs(rates[0] - rates[1])


def _evaluate(train_x, test_x, train_y, test_y, train_a, test_a, seed):
    disease = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200, random_state=seed))
    disease.fit(train_x, train_y); disease_prob = disease.predict_proba(test_x)[:, 1]
    linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200, random_state=seed))
    nonlinear = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15, l2_regularization=1.0,
                                               random_state=seed)
    probe_scores = []
    for probe in (linear, nonlinear):
        probe.fit(train_x, train_a); probe_scores.append(float(roc_auc_score(test_a, probe.predict_proba(test_x)[:, 1])))
    return {"disease_auroc": float(roc_auc_score(test_y, disease_prob)),
            "disease_brier": float(brier_score_loss(test_y, disease_prob)),
            "max_probe_auroc": max(probe_scores), "linear_probe_auroc": probe_scores[0],
            "nonlinear_probe_auroc": probe_scores[1],
            "equal_opportunity_gap": _equality_gap(test_y, disease_prob, test_a)}


def run():
    rows = []
    for seed in SEEDS:
        x, y, a = make_audit_dataset(seed=seed); cut = 2400
        train_x, test_x = x[:cut], x[cut:]; train_y, test_y = y[:cut], y[cut:]
        train_a, test_a = a[:cut], a[cut:]
        variants = {"original": (train_x, test_x),
                    "single_direction": single_direction(train_x, test_x, train_a),
                    "moment_matched": moment_match(train_x, test_x, train_a, test_a)}
        for name, (tr, te) in variants.items():
            rows.append({"seed": seed, "method": name, **_evaluate(tr, te, train_y, test_y, train_a, test_a, seed)})
    summary = {}
    for method in ("original", "single_direction", "moment_matched"):
        summary[method] = {}; subset = [r for r in rows if r["method"] == method]
        for metric in ("disease_auroc", "disease_brier", "max_probe_auroc", "linear_probe_auroc",
                       "nonlinear_probe_auroc", "equal_opportunity_gap"):
            values=np.array([r[metric] for r in subset]); margin=1.96*values.std(ddof=1)/np.sqrt(len(values))
            summary[method][metric]={"mean":float(values.mean()),"ci95_low":float(values.mean()-margin),
                                     "ci95_high":float(values.mean()+margin)}
    base=summary["original"]["max_probe_auroc"]["mean"]; proposed=summary["moment_matched"]["max_probe_auroc"]["mean"]
    report={"protocol":{"seeds":list(SEEDS),"train":2400,"test":1200,
                        "probes":["logistic","histogram gradient boosting"],
                        "important_limitation":"moment matching uses protected group at transformation time"},
            "summary":summary,"primary_comparison":{"metric":"maximum protected-attribute probe AUROC",
            "baseline":base,"proposed":proposed,"absolute_reduction":base-proposed,
            "disease_auroc_delta":summary["moment_matched"]["disease_auroc"]["mean"]-
                                   summary["original"]["disease_auroc"]["mean"]}}
    return rows,report


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/heavy"));a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);rows,report=run()
    (a.output/"heavy_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (a.output/"per_seed_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(report["primary_comparison"],indent=2))


if __name__=="__main__": main()
