"""Independent small-graph enumeration and mathematical boundary checks."""
import sys,itertools,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.linalg import eigh
from constitutive_box import Graph, certify, fit, scalar_fit, support_lower_bound

def independent_error(edges,n,c,s):
    # Ground vertex n-1. The remaining coordinates represent the quotient
    # by constants; no incidence matrix, null-space basis, or core error API.
    L=np.zeros((n,n)); H=L.copy()
    for (i,j),w,z in zip(edges,c,s):
        for A,v in [(L,w),(H,w*z)]:
            A[i,i]+=v; A[j,j]+=v; A[i,j]-=v; A[j,i]-=v
    C=np.linalg.cholesky(L[:-1,:-1]); Cinv=np.linalg.inv(C)
    ratios=np.linalg.eigvalsh(Cinv@H[:-1,:-1]@Cinv.T)
    return float(max(abs(ratios-1)))

class Certificates(unittest.TestCase):
    def setUp(self):
        self.edges=np.array(list(itertools.combinations(range(4),2)))
        self.g=Graph(self.edges,4)

    def test_independent_enumeration(self):
        rng=np.random.default_rng(651)
        self.max_gap=0
        for _ in range(60):
            lo=rng.uniform(.5,1.5,6); hi=lo*rng.uniform(1,9,6)
            s=rng.uniform(0,3,6); s[rng.random(6)<.4]=0
            exact=max(independent_error(self.edges,4,lo+(hi-lo)*np.array(z),s)
                      for z in itertools.product([0,1],repeat=6))
            r=certify(self.g,lo,hi,s)
            self.assertAlmostEqual(exact,r['epsilon'],places=9)
            self.assertAlmostEqual(exact,r['witnessed_error'],places=9)

    def test_identity_and_empty(self):
        lo=np.ones(6); hi=10*lo
        self.assertLess(certify(self.g,lo,hi,lo)['epsilon'],1e-10)
        self.assertAlmostEqual(certify(self.g,lo,hi,0*lo)['epsilon'],1,places=10)

    def test_scalar_solution(self):
        lo=np.arange(1,7)/3
        for keep in ([1,1,1,0,0,0],[1,1,1,1,0,1],[1]*6):
            for k in [1,2,8,32]:
                s,eps=scalar_fit(self.g,lo,np.array(keep,bool),k)
                self.assertAlmostEqual(certify(self.g,lo,k*lo,s)['epsilon'],eps,places=9)

    def test_lower_bound_and_fit(self):
        lo=np.ones(6); keep=np.array([1,1,1,0,1,0],bool)
        s,z=fit(self.g,lo,4*lo,keep)
        e=certify(self.g,lo,4*lo,s)['epsilon']
        self.assertLessEqual(support_lower_bound(self.g,lo,4*lo,keep),e+1e-8)
        self.assertLess(abs(e-z['objective']),2e-6)

    def test_disconnected_support(self):
        s=np.array([1,0,0,0,0,0.])
        self.assertGreaterEqual(certify(self.g,np.ones(6),2*np.ones(6),s)['epsilon'],1-1e-10)

    def test_invalid_domains(self):
        with self.assertRaises(ValueError):certify(self.g,np.zeros(6),np.ones(6),np.ones(6))
        with self.assertRaises(ValueError):certify(self.g,np.ones(6),np.ones(6),-np.ones(6))
        with self.assertRaises(ValueError):certify(self.g,np.ones(6),.5*np.ones(6),np.ones(6))

if __name__=='__main__':unittest.main(verbosity=2)
