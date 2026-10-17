"""Larger-graph closed-form scalar refits; no large SDP claim."""
import sys,json,time,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.linalg import eigh
from constitutive_box import Graph, scalar_fit, support_order
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for n in [32,64,128]:
 for kind in ['geometric','complete']:
  rng=np.random.default_rng(700+n)
  xy=rng.uniform(size=(n,2)); pairs=np.array(list(itertools.combinations(range(n),2)))
  if kind=='geometric':
   dd=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=-1);np.fill_diagonal(dd,np.inf)
   ee=set((i,i+1) for i in range(n-1))
   for i in range(n):
    for j in np.argsort(dd[i])[:8]:ee.add(tuple(sorted((i,int(j)))))
   pairs=np.array(sorted(ee))
  lo=.6+np.exp(-2*np.linalg.norm(xy[pairs[:,0]]-xy[pairs[:,1]],axis=1))
  g=Graph(pairs,n)
  for seed in [101,202,303]:
   order=support_order(g,lo,np.random.default_rng(seed))
   for fraction in [.5,.75,.9]:
    count=max(n-1,int(np.ceil(len(lo)*fraction))); keep=np.zeros(len(lo),bool);keep[order[:count]]=True
    for kappa in [1,2,4,8]:
     start=time.perf_counter();s,eps=scalar_fit(g,lo,keep,kappa);elapsed=time.perf_counter()-start
     rows.append(dict(n=n,kind=kind,m=len(lo),retained=count,seed=seed,fraction=fraction,
                      kappa=kappa,epsilon=eps,scalar_refit_seconds=elapsed))
  print(kind,n,'done',flush=True)
(ROOT/'results'/'scaling.json').write_text(json.dumps(rows,indent=2))
