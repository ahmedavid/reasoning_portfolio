# Portfolio Exam 2 — Sensor Fusion: Particle Filter (Task P2.1)

Particle filter that estimates the positions and velocity vectors of `n ≥ 1`
simultaneously flying balls from noisy, **unlabeled** position observations.
Interactive pygame visualization + a plot-based Jupyter notebook.

## Files

| File | Purpose |
|------|---------|
| `simulation.py` | ground truth: ball physics + the noisy, unreliable sensor |
| `filter.py` | the particle filter (no visualization dependencies) |
| `viz_pygame.py` | interactive pygame visualization (uses the two modules) |
| `experiment.py` | headless episode runner for notebooks/analysis |
| `notebook.ipynb` | static analysis with matplotlib plots |
| `main.py` | original single-file version (identical behavior, kept for reference) |
| `CONCEPT.md` | conceptual explanation of the approach |

## Requirements / Run

```
pip install numpy pygame matplotlib
python viz_pygame.py            # interactive pygame visualization
jupyter notebook notebook.ipynb # plot-based analysis (no pygame needed)
python main.py                  # original single-file version, same options
```

Runs without modification. Useful options (see `python viz_pygame.py --help`):

```
python viz_pygame.py --balls 5 --particles 8000 --obs-noise 5 --obs-interval 0.3
python viz_pygame.py --headless 1800 --seed 7   # no window, prints error stats
```

| Option            | Meaning                                              | Default |
|-------------------|------------------------------------------------------|---------|
| `--balls`         | number of balls n                                    | 3       |
| `--particles`     | number of particles                                  | 4000    |
| `--obs-noise`     | observation noise std dev sigma [m]                  | 3.0     |
| `--obs-interval`  | mean time between observations [s], jittered ±30 %   | 0.15    |
| `--dropout-prob`  | probability per observation that a dropout starts    | 0.02    |
| `--dropout-len`   | duration of a random sensor dropout [s]              | 1.5     |
| `--launch-area`   | side length of the unknown launch area [m]           | 50      |
| `--relaunch`      | balls relaunch after landing (otherwise they stay down and tracking of them stops) | off |
| `--seed`          | RNG seed for reproducibility                         | random  |

## Keys (change parameters live)

| Key        | Action                                        |
|------------|-----------------------------------------------|
| `SPACE`    | pause / resume                                |
| `R`        | reset simulation + filter                     |
| `D`        | toggle a manual, complete sensor dropout      |
| `P`        | show / hide the particle cloud                |
| `+` / `-`  | increase / decrease observation noise sigma   |
| `,` / `.`  | decrease / increase observation interval      |
| `UP`/`DOWN`| more / fewer balls (resets)                   |
| `ESC`      | quit                                          |

Visualization: white dots = true balls (with trails), yellow crosses = noisy
observations (fading), dimmed colored pixels = particles (colored by cluster),
colored circles with line = estimated positions with estimated velocity
vectors. The HUD shows all parameters, the sensor state (incl. DROPOUT) and
the current mean estimation error.

## Design decisions (interview talking points)

**State definition.** Each particle is ONE ball hypothesis `(x, y, vx, vy)`,
not the joint `4n`-dimensional state of all balls. The posterior over all n
balls is represented by the multimodal particle cloud as a whole. This keeps
the state space small (no curse of dimensionality), needs no data association
inside the state, and treats similar and clearly different launch parameters
with one common approach — similar balls simply produce overlapping modes.

**Initialization.** The launch parameters are unknown: particles are drawn
uniformly from the 50×50 m launch area with a broad velocity prior
(|v| ≤ 35 m/s, upward).

**Transition model.** Ballistic motion (gravity, no drag) integrated over the
*actually elapsed* time, plus Gaussian process noise scaled with sqrt(dt).
Because prediction runs every simulation step independently of measurements,
the filter keeps estimating during variable observation intervals and complete
sensor dropouts (pure prediction). Additionally, the known ground at y = 0 is
used as a constraint: particles below ground are impossible ("the ball this
particle followed must have landed"), so that mode dies and its particles are
re-distributed to the surviving modes. Tracking of a landed ball therefore
stops cleanly. With `--relaunch`, landed particles are instead recycled into
the launch prior, providing exploration for the next launch.

**Number of active modes.** The filter never sees the ground truth; it infers
how many balls are currently flying from the measurement count (balls are only
observed while airborne) and keeps that number during dropouts. The estimator
returns exactly that many position/velocity estimates — zero once everything
has landed.

**Evaluation model.** Observations are an unordered set of noisy positions —
balls are indistinguishable. A particle is weighted with the Gaussian
likelihood of its *nearest* measurement (max over per-measurement
likelihoods), i.e. an implicit nearest-neighbour association on particle
level. A small likelihood floor gives robustness. Two additions make the
multi-ball case stable:

1. *Measurement balancing*: each measurement stems from exactly one ball, so
   the particle mass assigned (nearest-neighbour) to each measurement is
   normalized to an equal share. Without this, modes slowly starve because the
   mode with more particles wins ever more particles at each resampling.
2. *Re-acquisition*: a measurement whose nearest particle is farther than
   3 sigma is "unexplained" (newly launched ball, or mode lost during a long
   dropout) and triggers injection of fresh hypotheses around it.

**Resampling.** Systematic (low variance) resampling, triggered only when the
effective sample size drops below N/2 (avoids unnecessary particle depletion).
A small fraction (~1 %) of fresh particles is injected afterwards to keep
diversity.

**Estimating n positions from the multimodal density.** Weighted k-means with
k = n on the particle positions, seeded with the previous estimates so cluster
identities stay stable over time (empty clusters are reseeded at high-weight
particles). Cluster centroids are the position estimates; the weighted mean
velocity of each cluster is the velocity estimate. (Alternatives would be
mean-shift or DBSCAN if n were unknown; k-means is appropriate here because n
is given.)

**Evaluation.** For display only (the filter never uses ground truth), the
mean distance between estimates and true airborne balls (greedy matching) is
shown. Typical values with default parameters: ~3–7 m mean error at sigma = 3 m
observation noise, including re-acquisition transients after balls land and
relaunch; error grows during dropouts and shrinks again afterwards.

**Sensor report semantics.** The sensor distinguishes "I am alive but detect
nothing" (an empty report — meaning nothing is airborne, tracked count drops
to 0) from "no report at all" (dropout — the filter keeps its current belief
and predicts onward).
