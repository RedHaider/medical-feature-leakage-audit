from leakage_audit.experiment import dataset,evaluate
from leakage_audit.heavy_experiment import make_audit_dataset, run
def test_shape():
 x,y,a=dataset(20,2);assert x.shape==(20,10) and len(y)==len(a)==20
def test_metrics():assert all(0<=v<=1 for m in evaluate(3).values() for v in m.values())

def test_heavy_dataset_shape():
 x,y,a=make_audit_dataset(40,4);assert x.shape==(40,12) and len(y)==len(a)==40

def test_heavy_audit_complete():
 rows,report=run();assert len(rows)==15;assert report["protocol"]["test"]==1200
 assert all(0<=r["max_probe_auroc"]<=1 for r in rows)

