# Interview Preparation — Likely Questions, Attacks and Honest Answers

Not part of the submission — preparation notes for the interview.

## A. Conceptual questions

**Q: Why is one particle one ball and not the joint state of all n balls?**
A joint state has 4n dimensions; the number of particles needed grows
exponentially with dimension (curse of dimensionality), and the likelihood of
a joint particle would require solving the measurement-to-ball assignment
inside every particle. With one ball per particle, the cloud as a whole
represents the *mixture* of all per-ball posteriors, modes emerge naturally,
and similar vs. clearly different launches need no special handling.
Trade-off to admit: we lose correlations between balls (irrelevant here —
balls don't interact) and ball *identity* (see crossing question below).

**Q: Is this still a proper Bayesian particle filter? You added heuristics.**
The core is standard SIR: predict with the transition model, weight with the
measurement likelihood, systematic resampling. Three pragmatic regularizations
deviate from pure Bayes, each addressing a known failure mode of SIR:
1. *Measurement balancing* — prevents mode starvation: with a shared particle
   pool, the mode with more particles wins ever more at each resampling (a
   random walk that eventually kills a mode). Justified because each
   measurement provably stems from exactly one ball.
2. *Injection at unexplained measurements* — re-acquisition, analogous to
   Augmented MCL in robot localization (random particle injection when
   observations contradict the particle set).
3. *Likelihood floor (1e-12)* — robustness against weight underflow.
I can switch each off live and show the resulting failure.

**Q: Why max (nearest measurement) in the likelihood and not the sum
(mixture likelihood)?**
Both are valid; max is a tight approximation of the sum when modes are
separated (the nearest term dominates) and identical when they overlap up to a
constant factor. Max also gives the `argmin` assignment for free, which the
balancing step needs. Easy live experiment: replace `min` by a proper sum —
behavior is nearly identical.

**Q: Why k-means? The density is multimodal — why not GMM-EM, mean-shift,
DBSCAN?**
The number of active modes is known to the filter (from the measurement
count), so a fixed-k method is appropriate and k-means is the cheapest robust
one. It is weighted (centroid = weighted mean) and warm-started with the
previous estimates → stable identities and ~2-3 iterations to converge.
Mean-shift/DBSCAN would be the choice if k were unknown; GMM-EM adds little
since we only need the modes' means.

**Q: Where does the velocity estimate come from? You never measure velocity.**
Through the transition model: particles whose velocity is wrong drift away
from subsequent position measurements and die at resampling. Velocity is
observable from ≥2 positions (plus gravity); convergence takes ~1 s at the
default rates. Demo: pause right after launch — velocity arrows are still
random; resume and watch them align.

**Q: Why resample only when ESS < N/2? Why systematic resampling?**
Resampling every step destroys diversity (sample impoverishment) without
adding information. ESS = 1/Σw² measures weight degeneracy; N/2 is the usual
threshold. Systematic resampling has lower variance than multinomial and is
O(N).

**Q: The flight is deterministic — why process noise at all?**
Without it, resampling duplicates particles exactly and the cloud collapses to
a few trajectories that can never be corrected (impoverishment). Process noise
is the regularizer that keeps the support of the posterior covered; it also
absorbs model mismatch (e.g. no drag in the model).

**Q: Does the filter ever use ground truth?**
No. Ground truth is used only for drawing the white balls and the error
metric. The filter sees: the noisy measurement set, the time step, gravity,
the ground at y=0, the launch-area size and the sensor noise σ.

**Q: Is knowing σ cheating?**
It's the standard assumption (sensor datasheet). With +/- I change the true
and the assumed σ together; mismatch (assumed ≠ true) can be discussed: too
small assumed σ → overconfident, ESS collapses, jittery; too large → sluggish
but stable. Easy to demonstrate by editing one line.

## B. "Let me try to break it" — attacks and what happens

**Two balls crossing each other.**
Estimates can swap identities (colors) at the crossing point — *by design*:
the task states balls are indistinguishable, so any labeling is only
positional. The position estimates themselves stay correct. Maintaining
identity through crossings would need velocity-consistent association
(JPDA/MHT) — out of scope, but I can explain how it would attach here
(associate clusters over time via the predicted positions instead of nearest
centroid).

**Two balls launched from the same spot in the same direction.**
One merged mode; k-means splits it into two nearby estimates — both close to
both true balls. As the trajectories separate (slightly different speeds),
measurement balancing guarantees both modes keep particle mass and the
estimates separate cleanly. This is the "very similar launch parameters" case
the task demands — same code path as distinct launches.

**Crank observation noise to the maximum (σ = 15 m).**
Modes blur and may merge; estimates average over a wide cloud; error grows
roughly with σ but the filter stays stable (floor + ESS-gated resampling).
Note the re-acquisition threshold scales with σ (3σ = 45 m), so injection
effectively switches off — correct, because at that noise level nothing is
"unexplained".

**σ very small (0.5 m).**
Likelihood becomes very peaked → ESS collapses → resampling every step; the
process noise prevents impoverishment. Tracking becomes extremely accurate.

**Long sensor dropout (hold D).**
Pure prediction: clouds fly ballistically and spread (process noise), the
estimates keep moving on plausible parabolas — exactly the required behavior.
Honest limitation: our uncertainty growth comes from process noise only, so
after a *very* long dropout the true ball may be > 3σ from the cloud; the
first measurement then triggers re-acquisition (a fresh, broad-velocity mode)
rather than a smooth update. That is the designed recovery path, not a crash.

**Dropout while a ball lands.**
The filter cannot know the ball landed; its cloud continues, hits the ground,
dies (ground constraint) and the mass moves to surviving modes. The tracked
count corrects itself with the next measurement. If *all* balls land during a
dropout, tracked count goes to 0 and no estimates are drawn.

**Many balls (n = 10) or few particles (N = 500).**
Particles per mode ≈ N/n; below a few hundred per mode, modes get noisy and
may need re-acquisition more often. Cost is O(N·m) per update — still real-time
for N = 4000, m = 10. Honest scaling answer: N should grow ~linearly with n
(thanks to balancing; without it, worse).

**Very sparse observations (interval 1–2 s).**
Between measurements the cloud spreads; velocity convergence is slower; with
only 2-3 measurements per flight the velocity stays uncertain (visible as
fanned-out velocity arrows). Works, but error grows — fundamental information
limit, not an implementation bug.

**Ball launched outside the assumed 50×50 launch area.**
(Professor edits `Ball.launch` or questions the prior.) The initial particle
set misses the ball, but its first measurement is "unexplained" → injection
acquires it within 1-2 observations. The launch-area prior is a soft
assumption, not a hard constraint.

**A measurement below ground (noise can push y < 0).**
Allowed — observations are noisy; particles themselves stay above ground
(constraint at state level, not measurement level). No crash.

**What about false measurements (clutter) or per-ball detection failures?**
Not required by the task and not modeled. Honest answers: clutter would fool
the unexplained-measurement injection (it would spawn a ghost mode that dies
once clutter disappears — mass is bounded, ~N/6n per event) and `k = len(Z)`
would briefly over/under-count; the fix is a median filter over recent
measurement counts and a minimum-persistence rule before accepting a new mode
(both are small changes; I can point to the exact lines).

**Initial tracked count is n before the first measurement — isn't that
cheating?**
It's a prior, and an arbitrary one: the first observation (within 0.15 s)
overwrites it with the measurement count. Could equally start at 0.

**Numerical attacks: all weights zero? Empty clusters? n = 1?**
All covered: weight-sum guard re-initializes the filter (degenerate case),
empty/negligible-mass clusters are reseeded at high-weight particles, n = 1
runs the identical code path (k-means with k = 1 = weighted mean).

## C. Parameter cheat sheet (for live tuning)

- `q_pos = 0.3`, `q_vel = 1.5` (process noise per √s): raise → more robust to
  model mismatch / long dropouts, but blurrier estimates; lower → sharper but
  brittle.
- ESS threshold `0.5·N`: lower → less resampling, more diversity, slower
  reaction.
- Re-acquisition threshold `3σ`: lower → faster acquisition, more ghost risk;
  higher → the opposite.
- Injection sizes: `N/(6n)` per unexplained measurement, ~1 % after resampling.
- Typical results (defaults, σ = 3 m): mean position error ≈ 3–8 m incl.
  acquisition transients; scales roughly with σ.
