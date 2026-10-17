"""Box-uniform certification of fixed-multiplier graph diffusion compression.

All numerical certificates use float64 eigensolvers, not interval arithmetic.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, null_space


@dataclass
class Graph:
    edges: np.ndarray
    n: int
    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=int)
        if self.n < 2 or self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("Expected n>=2 and m by 2 edges")
        if np.any(self.edges < 0) or np.any(self.edges >= self.n):
            raise ValueError("Invalid vertex")
        if np.any(self.edges[:, 0] == self.edges[:, 1]):
            raise ValueError("Self loops are not supported")
        self.B = np.zeros((len(self.edges), self.n))
        self.B[np.arange(len(self.edges)), self.edges[:, 0]] = 1
        self.B[np.arange(len(self.edges)), self.edges[:, 1]] = -1
        self.Q = null_space(np.ones((1, self.n)))
        self.C = self.B @ self.Q

    def lap(self, c, reduced=False):
        X = self.C if reduced else self.B
        return X.T @ (np.asarray(c)[:, None] * X)

    def validate(self, lo, hi, s=None):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if lo.shape != (len(self.edges),) or hi.shape != lo.shape:
            raise ValueError("Incorrect coefficient shape")
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError("Nonfinite interval")
        if np.any(lo <= 0) or np.any(hi < lo):
            raise ValueError("Require 0 < lo <= hi")
        if eigh(self.lap(lo, True), eigvals_only=True)[0] <= 1e-11:
            raise ValueError("Base graph must be connected and well scaled")
        if s is not None and (np.shape(s) != lo.shape or
                              not np.all(np.isfinite(s)) or np.any(np.asarray(s) < 0)):
            raise ValueError("Multipliers must be finite and nonnegative")
        return lo, hi


def relative_error(g, c, s, witness=False):
    vals, vec = eigh(g.lap(c * s, True), g.lap(c, True))
    j = 0 if abs(vals[0]-1) >= abs(vals[-1]-1) else len(vals)-1
    err = float(abs(vals[j]-1))
    return (err, g.Q @ vec[:, j], float(vals[j])) if witness else err


def envelopes(g, lo, hi, s, eps):
    """Loewner-maximal upper/lower violation matrices at a trial tolerance."""
    du, dl = s - 1 - eps, 1 - eps - s
    cu = np.where(du > 0, hi, lo)
    cl = np.where(dl > 0, hi, lo)
    U = g.lap(du * cu, True)
    V = g.lap(dl * cl, True)
    return U, V, cu, cl


def certify(g, lo, hi, s, iterations=48):
    lo, hi = g.validate(lo, hi, s)
    s = np.asarray(s, float)
    left, right = 0., max(1., float(np.max(s)-1)) + 1e-12
    for _ in range(iterations):
        mid = (left+right)/2
        U, V, _, _ = envelopes(g, lo, hi, s, mid)
        if max(eigh(U, eigvals_only=True)[-1], eigh(V, eigvals_only=True)[-1]) <= 0:
            right = mid
        else:
            left = mid
    U, V, cu, cl = envelopes(g, lo, hi, s, right)
    choices = [(relative_error(g, c, s, True), c) for c in (cu, cl)]
    (err, x, ratio), c = max(choices, key=lambda a: a[0][0])
    return dict(epsilon=right, witnessed_error=err, ratio=ratio,
                witness_state=x.tolist(), witness_c=c.tolist(),
                upper_margin=float(-eigh(U, eigvals_only=True)[-1]),
                lower_margin=float(-eigh(V, eigvals_only=True)[-1]))


def fit(g, lo, hi, support, scenarios=None):
    """Exact convex formulation; scenarios=None uses the two-envelope SDP.

    Support is fixed, so this does not solve combinatorial support selection.
    Returned weights are independently recertified by certify().
    """
    import cvxpy as cp
    lo, hi = g.validate(lo, hi)
    m, d = len(lo), g.n-1
    s = cp.Variable(m, nonneg=True)
    eps = cp.Variable(nonneg=True)
    support = np.asarray(support, bool)
    constraints = [s[~support] == 0] if not np.all(support) else []
    Aflat = np.einsum('ei,ej->eij', g.C, g.C).reshape(m, d*d).T
    def matrix(v):
        return cp.reshape(Aflat @ v, (d, d), order='C')
    if scenarios is None:
        p, q = cp.Variable(m, nonneg=True), cp.Variable(m, nonneg=True)
        constraints += [p >= s-1-eps, q >= 1-eps-s]
        constraints += [matrix(cp.multiply(lo, s-1-eps)+cp.multiply(hi-lo,p)) << 0,
                        matrix(cp.multiply(lo, 1-eps-s)+cp.multiply(hi-lo,q)) << 0]
    else:
        for c in np.asarray(scenarios):
            constraints += [matrix(cp.multiply(c, s-1-eps)) << 0,
                            matrix(cp.multiply(c, 1-eps-s)) << 0]
    prob = cp.Problem(cp.Minimize(eps), constraints)
    prob.solve(solver='CLARABEL', tol_gap_abs=1e-8, tol_feas=1e-8,
               tol_gap_rel=1e-8, max_iter=150)
    if prob.status not in ('optimal', 'optimal_inaccurate') or s.value is None:
        raise RuntimeError(f"SDP failed: {prob.status}")
    weights = np.maximum(s.value, 0)
    weights[~support] = 0
    return weights, dict(status=prob.status, objective=float(eps.value),
                         solve_seconds=float(prob.solver_stats.solve_time))


def scalar_fit(g, lo, support, kappa):
    """Closed-form minimax scalar weight on a fixed support, hi=kappa*lo."""
    L = g.lap(lo, True)
    T = g.lap(lo*np.asarray(support), True)
    t = np.clip(eigh(T,L,eigvals_only=True),0,1)
    a = t[0]/(kappa-(kappa-1)*t[0])
    b = kappa*t[-1]/(1+(kappa-1)*t[-1])
    if a+b < 1e-14:
        return np.zeros_like(lo), 1.
    alpha = 2/(a+b)
    return alpha*np.asarray(support), float((b-a)/(a+b))


def support_lower_bound(g, lo, hi, support):
    """Any multipliers on this support satisfy epsilon >= delta/(2+delta)."""
    D = g.lap((hi-lo)*(~np.asarray(support, bool)), True)
    delta = max(0., float(eigh(D,g.lap(lo,True),eigvals_only=True)[-1]))
    return delta/(2+delta)


def support_order(g, lo, rng):
    """Maximum-weight spanning tree plus leverage-weighted random ordering.

    An experimental support generator, not the SS sampling algorithm.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    adjacency = np.zeros((g.n,g.n))
    for (i,j),w in zip(g.edges,lo):
        adjacency[i,j]=adjacency[j,i]=-w
    tree = minimum_spanning_tree(adjacency).toarray()
    chosen = [e for e,(i,j) in enumerate(g.edges) if tree[i,j] or tree[j,i]]
    inv = np.linalg.inv(g.lap(lo,True))
    leverage = lo*np.einsum('ei,ij,ej->e',g.C,inv,g.C)
    keys = -np.log(rng.uniform(size=len(lo)))/np.maximum(leverage,1e-15)
    rest = [e for e in np.argsort(keys) if e not in chosen]
    return np.array(chosen+rest)
