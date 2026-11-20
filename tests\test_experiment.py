from leakage_audit.experiment import dataset,evaluate
def test_shape():
 x,y,a=dataset(20,2);assert x.shape==(20,10) and len(y)==len(a)==20
def test_metrics():assert all(0<=v<=1 for m in evaluate(3).values() for v in m.values())

