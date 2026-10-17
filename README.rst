Constitutive-Box Certificates
============================

Uniform spectral accuracy for compressed learned diffusion networks.

Author: Shivam Gupta, Independent Researcher.

The paper studies a fixed nonnegative multiplier vector reused over a box
of positive edge conductances. Two tolerance-dependent extremal settings
exactly characterize the uniform relative quadratic-form error in real
arithmetic. The implementation evaluates this characterization in float64.

Paper
-----

The manuscript is in ``paper/Constitutive_Box_Certificates_Shivam_Gupta.pdf``.
It includes five vector figures, proofs, numerical limitations, and an
appendix mapping every result to code. This is a research manuscript,
not an accepted publication.

Reproduce
---------

Use Python 3.9 or a compatible later version. From this directory::

    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiments/run.py
    OPENBLAS_NUM_THREADS=1 python experiments/scaling.py
    python -m unittest discover -s tests -v
    python experiments/figures.py
    python experiments/validate_saved.py
    cd paper
    tectonic main.tex

Tectonic is needed only to build the PDF. The original primary experiment
took about 102 seconds locally; numerical timing varies by platform.
Reproduction regenerates result files and timing fields.

Minimal use
-----------

Add ``src/`` to PYTHONPATH, then::

    import numpy as np
    from constitutive_box import Graph, scalar_fit, certify
    g = Graph(np.array([[0, 1], [1, 2], [0, 2]]), 3)
    lo = np.ones(3)
    support = np.array([True, True, False])
    s, error = scalar_fit(g, lo, support, 4)
    cert = certify(g, lo, 4*lo, s)
    assert abs(error - 0.8) < 1e-10

The certificate contains the error, an attained coefficient setting and
state, the ratio at that witness, and eigenvalue margins. Fixed-support
per-edge minimax fitting is available as ``fit``. Specifying ``scenarios``
uses ordinary per-scenario spectral constraints instead of the box SDP.

Evidence
--------

* 576 fits: four graph families, four seeds, three budgets, four contrasts,
  and three fitting objectives.
* 7,680 enumerated corner evaluations on 120 K4 designs.
* 3,840 further corner evaluations with independent grounded assembly.
* 216 larger scalar refits up to 128 vertices and 8,128 edges.
* Twelve switched diffusion rollouts with mass, positivity, energy and
  state-error checks.

The raw results retain inaccurate solver statuses. Robust fitting improves
the specified worst-case objective but often worsens accuracy on ordinary
learned test settings. These are synthetic mechanism experiments, not an
industrial performance benchmark. The dense per-edge SDP is only evaluated
on the small graphs; larger experiments use the closed-form scalar refit.

Assumptions
-----------

* Fixed connected undirected graph; scalar, strictly positive conductance
  intervals; independent box uncertainty.
* Shared conductance vector in full and compressed operators; fixed
  nonnegative multipliers; no changing support during a certificate.
* A supplied envelope is not a learned confidence region.
* Dynamical error bounds concern a common externally specified coefficient
  sequence and a positive diagonal capacity matrix.
* Float64 checks are not interval-arithmetic or formal numerical proofs.

The two-envelope argument specializes established Loewner monotonicity.
The proposed contribution is the explicit compression contract and its
combined audit, refit, support bound, scalar solution, and empirical study.
See the paper and ``literature_review.csv`` for the contribution boundary.

License and citation
--------------------

Code is MIT licensed. Cite the manuscript using ``CITATION.cff``. The
manufactured-data generators and raw results are included without external
dataset downloads. ``SHA256SUMS.txt`` identifies the delivered source and
result snapshot.
