"""Validate the delivered raw results without fitting models again."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from constitutive_box import Graph,certify,relative_error,support_lower_bound
root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'results/study.json').read_text())
assert len(rows)==576
maxgap=0
for r in rows:
 spec=json.loads((root/f'results/graph_{r["graph"]}.json').read_text())
 g=Graph(np.array(spec['edges']),spec['n']);lo=np.array(spec['lo']);s=np.array(r['weights']); hi=r['kappa']*lo
 assert np.all(s>=0) and np.all(s[~np.array(r['support'])]==0)
 c=certify(g,lo,hi,s)
 maxgap=max(maxgap,abs(c['epsilon']-r['certified']))
 assert abs(c['epsilon']-r['certified'])<1e-10
 assert abs(c['witnessed_error']-r['certified'])<1e-9
 assert r['learned_test_max']<=r['certified']+1e-9
 assert r['random_box_max']<=r['certified']+1e-9
 assert support_lower_bound(g,lo,hi,np.array(r['support']))<=r['certified']+1e-9
assert len(json.loads((root/'results/scaling.json').read_text()))==216
print(json.dumps(dict(saved_fits_checked=len(rows),failures=0,max_recheck_gap=maxgap),indent=2))
