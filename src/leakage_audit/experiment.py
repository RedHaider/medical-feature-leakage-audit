import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def dataset(n=2400,seed=27):
 r=np.random.default_rng(seed);a=r.binomial(1,.5,n);z=r.normal(size=(n,10));z[:,:3]+=a[:,None]*.9;logit=.9*z[:,3]-.7*z[:,4]+.45*a-1;y=r.binomial(1,1/(1+np.exp(-logit)));return z,y,a
def residualize(train,test,a):
 means=np.array([train[a==v].mean(0) for v in [0,1]]);direction=means[1]-means[0];direction/=np.linalg.norm(direction)+1e-9;return train-train@direction[:,None]*direction, test-test@direction[:,None]*direction
def evaluate(seed=27):
 x,y,a=dataset(seed=seed);cut=1600;out={}
 for name,(tr,te) in {"original":(x[:cut],x[cut:]),"residualized":residualize(x[:cut],x[cut:],a[:cut])}.items():
  disease=LogisticRegression(max_iter=500).fit(tr,y[:cut]).predict_proba(te)[:,1];probe=LogisticRegression(max_iter=500).fit(tr,a[:cut]).predict_proba(te)[:,1];out[name]={"disease_auroc":float(roc_auc_score(y[cut:],disease)),"protected_probe_auroc":float(roc_auc_score(a[cut:],probe))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
