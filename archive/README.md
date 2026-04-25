# Archive

This directory holds files from earlier (falsified or historical) lines
of work on this problem. They are kept as evidence and reference, not as
active development. The current working design lives at the top level
(`../design-journal.md` §9, `../surface_preserving_ops.py`, and the two test
files).

## Contents

**Design history**
- `ellipsoid-approach.md` — the original guess-and-flip + analytic-center
  fitness design. Empirically falsified; banner at top describes which
  claims didn't hold.

**Empirical audit of the falsified design**
- `findings.md` — the audit document that catalogues the falsification
  with concrete N = 6, 8, 10 counterexamples and discusses the discrete
  vs. continuous fitness comparison.
- `experiment_hillclimb.py` — core library used by the audit (instance
  generator, both fitness functions, hill-climb, local-optimum counter).
- `experiment_v4.py` — sweep over (N, m) that produces the table in
  `findings.md` §3.
- `counterexample.py` — produces the concrete counterexample printed in
  `findings.md` §3b.

**Single-purpose evidence file**
- `analyze_v3.py` — small N=3 verification that the standard ellipsoid-
  method MVCE update does NOT preserve x* on surface. Referenced from
  `../design-journal.md` §9.5 and the `ellipsoid-approach.md` banner. The
  point of this file is to demonstrate why we ended up needing a
  different construction (the pencil-of-ellipsoids approach in
  `../surface_preserving_ops.py`).

## Why these are kept

- The `findings.md` audit is the empirical justification for abandoning
  the original design; without the supporting code the claims would be
  hard to reproduce or check.
- `analyze_v3.py` is the concrete falsification of the "x* on surface"
  invariant under the standard MVCE update — also referenced from the
  current docs as the motivation for the pencil construction.
- `ellipsoid-approach.md` is the historical record of how we got here.

## Internal imports

These scripts import from each other:
- `counterexample.py` imports from `experiment_v4` and
  `experiment_hillclimb`.
- `experiment_v4.py` imports from `experiment_hillclimb`.
- `experiment_hillclimb.py` is the leaf.

All imports resolve within this directory.
