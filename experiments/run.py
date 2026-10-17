"""Execute the declared finite study. Run from the repository root."""
import sys, json, time, itertools, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.optimize import minimize
from constitutive_box import Graph, fit, certify, relative_error, scalar_fit, support_order, support_lower_bound

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)


def graphs():
    xy=np.array([(i/3,j/3) for i in range(4) for j in range(4)])
    pairs=np.array(list(itertools.combinations(range(16),2)))
    dist=np.linalg.norm(xy[pairs[:,0]]-xy[pairs[:,1]],axis=1)
    grid=pairs[dist<=np.sqrt(2)/3+1e-8]
    rng=np.random.default_rng(20406)
    gx=rng.uniform(size=(16,2))
    dd=np.linalg.norm(gx[:,None,:]-gx[None,:,:],axis=-1)
    np.fill_diagonal(dd,np.inf)
    geo=set()
    for i in range(16):
        for j in np.argsort(dd[i])[:5]: geo.add(tuple(sorted((i,int(j)))))
    geo.update((i,i+1) for i in range(15))
    clustered=np.array([(i,j) for i,j in pairs if (i<8)==(j<8) or (i,j) in [(3,8),(7,12)]])
    for name,edges,coords in [('grid',grid,xy),('geometric',np.array(sorted(geo)),gx),
                               ('clustered',clustered,xy),('complete',pairs,xy)]:
        length=np.linalg.norm(coords[edges[:,0]]-coords[edges[:,1]],axis=1)
        lo=0.6+np.exp(-2*length)
        yield name,Graph(edges,16),coords,lo


def inputs(g,xy,z):
    mid=(xy[g.edges[:,0]]+xy[g.edges[:,1]])/2
    delta=xy[g.edges[:,0]]-xy[g.edges[:,1]]
    E=np.tile(np.c_[mid,delta**2],(len(z),1))
    Z=np.repeat(z,len(g.edges),axis=0)
    return np.c_[2*E[:,:2]-1, E[:,2:], Z]


def train_field(g,xy):
    rng=np.random.default_rng(411)
    trainz=rng.uniform(-1,1,(128,2)); testz=rng.uniform(-1,1,(128,2))
    X=inputs(g,xy,trainz)
    def teacher(X):
        a,b,dx,dy,z1,z2=X.T
        return 1/(1+np.exp(-(1.1*np.sin(2*a+z1)+0.8*np.cos(2*b-z2)+0.6*z1*z2+dx-dy-0.4)))
    y=teacher(X)
    h=16; d=X.shape[1]
    p=np.r_[rng.normal(0,.5,d*h),np.zeros(h),rng.normal(0,.25,h),0.]
    def unpack(p): return p[:d*h].reshape(d,h),p[d*h:d*h+h],p[d*h+h:-1],p[-1]
    def predict(p,X):
        W,b,v,k=unpack(p); H=np.tanh(X@W+b)
        return 1/(1+np.exp(-np.clip(H@v+k,-40,40)))
    def objective(p):
        W,b,v,k=unpack(p); H=np.tanh(X@W+b)
        out=1/(1+np.exp(-np.clip(H@v+k,-40,40)))
        r=out-y; dz=2*r*out*(1-out)/len(y)
        back=(dz[:,None]*v)*(1-H*H)
        grad=np.r_[(X.T@back).ravel(),back.sum(0),H.T@dz,dz.sum()]
        return np.mean(r*r)+1e-7*np.sum(p*p),grad+2e-7*p
    start=time.perf_counter()
    result=minimize(objective,p,jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-11,'gtol':1e-7})
    testX=inputs(g,xy,testz)
    info=dict(train_rmse=float(np.sqrt(np.mean((predict(result.x,X)-y)**2))),
              test_rmse=float(np.sqrt(np.mean((predict(result.x,testX)-teacher(testX))**2))),
              iterations=int(result.nit), success=bool(result.success), message=str(result.message),
              seconds=time.perf_counter()-start)
    np.savez(OUT/f'learned_{g.n}_{len(g.edges)}.npz',parameters=result.x,train_z=trainz,test_z=testz)
    return lambda z: predict(result.x,inputs(g,xy,z)).reshape(len(z),len(g.edges)),info


def exhaustive():
    rng=np.random.default_rng(1720406)
    g=Graph(np.array(list(itertools.combinations(range(4),2))),4)
    records=[]; maxgap=0
    for i in range(120):
        lo=rng.uniform(.5,1.5,6); hi=lo*rng.uniform(1,8,6)
        s=rng.uniform(0,3,6); s[rng.random(6)<.35]=0
        corners=np.array(list(itertools.product([0,1],repeat=6)))
        exact=max(relative_error(g,lo+(hi-lo)*z,s) for z in corners)
        cert=certify(g,lo,hi,s)
        gap=abs(exact-cert['epsilon']); maxgap=max(gap,maxgap)
        assert gap<1e-9 and abs(cert['witnessed_error']-exact)<1e-9
        records.append(dict(id=i,lo=lo.tolist(),hi=hi.tolist(),s=s.tolist(),enumerated=exact,certificate=cert['epsilon']))
    (OUT/'exhaustive.json').write_text(json.dumps(records,indent=2))
    return dict(instances=120,corners_per_instance=64,corner_evaluations=7680,max_gap=maxgap)


def dynamics(g,c,s,seed):
    rng=np.random.default_rng(seed)
    M=rng.uniform(.7,1.3,g.n)
    mass=np.diag(M); L=g.lap(c); H=g.lap(c*s)
    # Common coefficient sequence: external changing material, not two state-dependent laws.
    u=rng.uniform(0,1,g.n); v=u.copy(); u0=u.copy()
    dt=.04; E0=.5*np.sum(M*u*u)
    maxerr=0.; bound=0.; worst_mass=0.; minval=1.; max_energy_increment=0.
    prevE=.5*np.sum(M*v*v)
    eps=relative_error(g,c,s)
    traj=[]
    for k in range(100):
        multiplier=1+.25*np.sin(k*.31)
        Lk=multiplier*L; Hk=multiplier*H
        un=np.linalg.solve(mass+dt*Lk,M*u)
        vn=np.linalg.solve(mass+dt*Hk,M*v)
        u,v=un,vn
        err=float(np.sqrt(np.sum(M*(u-v)**2)))
        maxerr=max(maxerr,err)
        worst_mass=max(worst_mass,abs(float(M@(v-u0))))
        minval=min(minval,float(v.min()))
        energy=.5*np.sum(M*v*v)
        max_energy_increment=max(max_energy_increment,float(energy-prevE)); prevE=energy
        traj.append([dt*(k+1),err,energy])
    # General switched-generator energy estimate proved in the paper.
    bound=eps*np.sqrt(E0/max(1-eps,1e-15)) if eps<1 else None
    if bound is not None: assert maxerr <= bound+1e-9
    return dict(max_error=maxerr,energy_bound=bound,mass_drift=worst_mass,
                minimum_state=minval,max_energy_increment=max_energy_increment,trajectory=traj)


def main():
    start=time.perf_counter(); summary={'exhaustive':exhaustive()}; rows=[]; fields={}; cases=[]
    for name,g,xy,lo in graphs():
        field,info=train_field(g,xy); fields[name]=info
        (OUT/f'graph_{name}.json').write_text(json.dumps(dict(n=g.n,edges=g.edges.tolist(),xy=xy.tolist(),lo=lo.tolist())))
        print(name,'learned',info,flush=True)
        for seed in [101,202,303,404]:
            rng=np.random.default_rng(seed)
            order=support_order(g,lo,rng)
            fitted=field(rng.uniform(-1,1,(8,2)))
            tested=field(rng.uniform(-1,1,(128,2)))
            random_box=rng.uniform(size=(128,len(lo)))
            for fraction in [.5,.75,.9]:
                count=max(g.n-1,int(np.ceil(fraction*len(lo))))
                support=np.zeros(len(lo),bool); support[order[:count]]=True
                for kappa in [1.,2.,4.,8.]:
                    hi=kappa*lo
                    scenarios=lo*(1+(kappa-1)*fitted)
                    testcs=lo*(1+(kappa-1)*tested)
                    randcs=lo*(1+(kappa-1)*random_box)
                    methods={}
                    for method,cs in [('nominal',[(lo+hi)/2]),('scenario',scenarios),('box',None)]:
                        s,solve=fit(g,lo,hi,support,cs)
                        cert=certify(g,lo,hi,s)
                        row=dict(graph=name,seed=seed,fraction=fraction,n=g.n,m=len(lo),retained=count,
                                 kappa=kappa,method=method,weights=s.tolist(),support=support.tolist(),
                                 certified=cert['epsilon'],witnessed=cert['witnessed_error'],
                                 nominal=relative_error(g,(lo+hi)/2,s),
                                 learned_test_max=max(relative_error(g,c,s) for c in testcs),
                                 random_box_max=max(relative_error(g,c,s) for c in randcs),
                                 lower_bound=support_lower_bound(g,lo,hi,support),**solve)
                        if method=='box': assert abs(solve['objective']-cert['epsilon'])<2e-5
                        rows.append(row); methods[method]=(s,cert)
                    if seed==101 and fraction==.75 and kappa==4:
                        # Each model is tested at its own attained worst spectral setting.
                        case=dict(graph=name)
                        for method,(s,cert) in methods.items():
                            c=np.array(cert['witness_c'])
                            case[method]={'certificate':cert,'dynamics':dynamics(g,c,s,501)}
                        cases.append(case)
                    if seed==101 and fraction==.75:
                        sc,closed=scalar_fit(g,lo,support,kappa)
                        assert abs(certify(g,lo,hi,sc)['epsilon']-closed)<1e-9
            print(name,seed,'done',flush=True)
        (OUT/'study.json').write_text(json.dumps(rows,indent=2))
    (OUT/'dynamics.json').write_text(json.dumps(cases,indent=2))
    summary.update(fields=fields,rows=len(rows),cells=len(rows)//3,
                   seconds=time.perf_counter()-start,python=sys.version,numpy=np.__version__,
                   scipy=scipy.__version__,platform=platform.platform())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
