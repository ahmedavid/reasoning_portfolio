"""
The particle filter -- completely independent of any visualization.

See README.md / CONCEPT.md for the design rationale. In short:
- one particle = one ball hypothesis (x, y, vx, vy); the multimodal cloud as a
  whole represents all n balls
- transition model: ballistic motion over the truly elapsed time + process
  noise (works through variable intervals and complete sensor dropouts)
- evaluation model: Gaussian likelihood of the nearest measurement (balls are
  indistinguishable), with measurement balancing and re-acquisition injection
- estimates: weighted k-means over the particle positions, k = number of
  currently active modes (inferred from the measurement count)
"""

import math

import numpy as np

from simulation import G, VEL_PRIOR


class ParticleFilter:
    def __init__(self, n_particles, n_balls, launch_area, obs_noise,
                 rng_seed=None, relaunch=False):
        self.N = n_particles
        self.n_balls = n_balls
        self.relaunch = relaunch
        self.launch_area = launch_area
        self.sigma = obs_noise           # sensor model std dev (assumed known)
        self.rng = np.random.default_rng(rng_seed)
        # process noise stds (per sqrt(second))
        self.q_pos = 0.3
        self.q_vel = 1.5
        self.estimates = None            # (k_active, 4) after first update
        self.init_particles()

    def init_particles(self):
        """Uniform prior: launch somewhere in the launch area, broad velocity prior."""
        r = self.rng
        N = self.N
        self.P = np.empty((N, 4))
        self.P[:, 0] = r.uniform(0, self.launch_area, N)            # x
        self.P[:, 1] = r.uniform(0, self.launch_area, N)            # y
        self.P[:, 2] = r.uniform(*VEL_PRIOR["vx"], N)               # vx
        self.P[:, 3] = r.uniform(*VEL_PRIOR["vy"], N)               # vy
        self.w = np.full(N, 1.0 / N)
        self.estimates = None
        # number of currently tracked modes. The filter cannot see the ground
        # truth; it uses the measurement count (balls are only observed while
        # airborne), so tracking of a ball stops once it has landed.
        self.k_active = self.n_balls

    # -- transition model ----------------------------------------------------
    def predict(self, dt):
        """Ballistic motion over the true elapsed time dt + process noise.
        Used every simulation step, so the filter keeps estimating during
        sensor dropouts (pure prediction)."""
        self.P[:, 3] -= G * dt
        self.P[:, 0] += self.P[:, 2] * dt
        self.P[:, 1] += self.P[:, 3] * dt
        s = math.sqrt(dt)
        self.P[:, 0:2] += self.rng.normal(0, self.q_pos * s, (self.N, 2))
        self.P[:, 2:4] += self.rng.normal(0, self.q_vel * s, (self.N, 2))
        # Ground constraint: the ground at y=0 is known. A hypothesis below
        # ground is impossible -> the ball it followed must have landed and
        # its mode dies. This removes "dead" modes from the particle set.
        below = self.P[:, 1] < -2.0
        nb = int(below.sum())
        if nb:
            r = self.rng
            if self.relaunch:
                # balls relaunch -> a new launch is the most plausible
                # continuation: recycle into the launch prior (exploration)
                self.P[below, 0] = r.uniform(0, self.launch_area, nb)
                self.P[below, 1] = r.uniform(0, self.launch_area, nb)
                self.P[below, 2] = r.uniform(*VEL_PRIOR["vx"], nb)
                self.P[below, 3] = r.uniform(*VEL_PRIOR["vy"], nb)
                self.w[below] = 1.0 / self.N
                self.w /= self.w.sum()
            else:
                # tracking stops: re-distribute the particles to the
                # surviving modes (particles still meaningfully airborne)
                above = np.where(self.P[:, 1] > 0.5)[0]
                if len(above) == 0:           # everything landed -> stop
                    self.k_active = 0
                    self.P[below, 1] = 0.0
                    self.P[below, 2:4] = 0.0
                else:
                    wa = self.w[above] / self.w[above].sum()
                    src = r.choice(above, nb, p=wa)
                    self.P[below] = self.P[src] + r.normal(0, 0.3, (nb, 4))
                    self.w[below] = self.w[src]
                    self.w /= self.w.sum()

    # -- evaluation model ----------------------------------------------------
    def update(self, observations):
        """Weight particles with the likelihood of the NEAREST measurement
        (max over Gaussian likelihoods). No explicit data association is
        needed because measurements are an unordered set."""
        Z = np.asarray(observations)                      # (m, 2)
        # balls are only observed while airborne -> the measurement count
        # tells the filter how many modes are currently active. An empty
        # report ("sensor alive, zero detections") means nothing is flying.
        self.k_active = len(Z)
        if len(Z) == 0:
            return
        d2 = ((self.P[:, None, 0:2] - Z[None, :, :]) ** 2).sum(axis=2)  # (N, m)

        # Re-acquisition: a measurement whose nearest particle is farther than
        # 3 sigma is "unexplained" (e.g. a newly launched ball after the old
        # mode died, or a mode lost during a long dropout). Spawn fresh
        # hypotheses around it, replacing the lowest-weight particles.
        unexplained = np.where(d2.min(axis=0) > (3 * self.sigma) ** 2)[0]
        if len(unexplained) > 0:
            k_each = max(8, self.N // (6 * max(1, self.n_balls)))
            order = np.argsort(self.w)                    # lowest weights first
            pos = 0
            r = self.rng
            for j in unexplained:
                idx = order[pos:pos + k_each]
                pos += k_each
                self.P[idx, 0:2] = Z[j] + r.normal(0, self.sigma, (len(idx), 2))
                self.P[idx, 2] = r.uniform(*VEL_PRIOR["vx"], len(idx))
                self.P[idx, 3] = r.uniform(-35, 35, len(idx))   # may already fall
                self.w[idx] = 1.0 / self.N
            self.w /= self.w.sum()
            d2 = ((self.P[:, None, 0:2] - Z[None, :, :]) ** 2).sum(axis=2)

        min_d2 = d2.min(axis=1)
        assign = d2.argmin(axis=1)                        # nearest measurement
        lik = np.exp(-0.5 * min_d2 / self.sigma ** 2) + 1e-12  # floor: robustness
        # ground constraint: the ground at y=0 is known, a flying ball cannot
        # be below it -> hypotheses below ground are (nearly) impossible.
        lik[self.P[:, 1] < -1.0] = 1e-12
        self.w *= lik

        # Measurement balancing: every measurement stems from exactly one ball,
        # so each ball should keep (roughly) the same share of posterior mass.
        # Without this, modes slowly starve because the mode with more
        # particles wins ever more particles at each resampling step.
        mtot = len(Z)
        for j in range(mtot):
            cohort = assign == j
            mass = self.w[cohort].sum()
            if mass > 1e-9:
                self.w[cohort] *= (1.0 / mtot) / mass
        wsum = self.w.sum()
        if wsum <= 0 or not np.isfinite(wsum):            # degenerate -> restart
            self.init_particles()
            return
        self.w /= wsum

        # resample only when the effective sample size gets low
        ess = 1.0 / np.sum(self.w ** 2)
        if ess < 0.5 * self.N:
            self.systematic_resample()
            self.inject(observations)

    def systematic_resample(self):
        """Low variance (systematic) resampling."""
        positions = (self.rng.random() + np.arange(self.N)) / self.N
        idx = np.searchsorted(np.cumsum(self.w), positions)
        idx = np.clip(idx, 0, self.N - 1)
        self.P = self.P[idx]
        self.w = np.full(self.N, 1.0 / self.N)

    def inject(self, observations):
        """Replace a small random fraction of particles with fresh hypotheses
        around current measurements (and a few uniform in the launch area).
        Keeps diversity so modes are not lost permanently."""
        r = self.rng
        k = max(4, self.N // 100)                          # ~1 %
        idx = r.choice(self.N, k, replace=False)
        half = k // 2
        Z = np.asarray(observations)
        zi = r.integers(0, len(Z), half)
        self.P[idx[:half], 0:2] = Z[zi] + r.normal(0, 2 * self.sigma, (half, 2))
        self.P[idx[:half], 2] = r.uniform(*VEL_PRIOR["vx"], half)
        self.P[idx[:half], 3] = r.uniform(-20, 30, half)
        rest = idx[half:]
        self.P[rest, 0] = r.uniform(0, self.launch_area, len(rest))
        self.P[rest, 1] = r.uniform(0, self.launch_area, len(rest))
        self.P[rest, 2] = r.uniform(*VEL_PRIOR["vx"], len(rest))
        self.P[rest, 3] = r.uniform(*VEL_PRIOR["vy"], len(rest))

    # -- multimodal state extraction ------------------------------------------
    def estimate(self):
        """Weighted k-means (k = active modes) on particle positions to resolve
        the multimodal density into k ball estimates. Seeded with the previous
        estimates so cluster identities are stable over time.
        Returns (k, 4): x, y, vx, vy and the particle->cluster labels."""
        k = self.k_active
        if k <= 0:                       # nothing airborne -> no estimates
            self.estimates = np.zeros((0, 4))
            return self.estimates, np.zeros(self.N, dtype=int)
        X = self.P[:, 0:2]
        w = self.w
        if self.estimates is None or len(self.estimates) != k:
            # seeding from high-weight particles
            seed_idx = self.rng.choice(self.N, k, replace=False, p=w)
            C = X[seed_idx].copy()
        else:
            C = self.estimates[:, 0:2].copy()

        labels = np.zeros(self.N, dtype=int)
        for _ in range(8):                                  # few iterations suffice
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            labels = d2.argmin(axis=1)
            for j in range(k):
                m = labels == j
                wm = w[m]
                if wm.sum() > 1e-3:                     # significant mass
                    C[j] = np.average(X[m], axis=0, weights=wm)
                else:                  # (nearly) empty cluster -> reseed at a
                    C[j] = X[self.rng.choice(self.N, p=w)]  # high-weight particle

        est = np.zeros((k, 4))
        est[:, 0:2] = C
        for j in range(k):                                  # per-cluster velocity
            m = labels == j
            wm = w[m]
            if wm.sum() > 1e-12:
                est[j, 2:4] = np.average(self.P[m][:, 2:4], axis=0, weights=wm)
        self.estimates = est
        return est, labels


# ----------------------------------------------------------------------------
# Error metric (uses ground truth, for display/evaluation only)
# ----------------------------------------------------------------------------
def mean_match_error(estimates, balls):
    """Greedy assignment of estimates to airborne true balls; mean distance."""
    truth = [(b.x, b.y) for b in balls if b.airborne]
    if not truth or estimates is None:
        return None
    est = [tuple(e[0:2]) for e in estimates]
    errs = []
    used = set()
    for tx, ty in truth:
        best, bi = None, None
        for i, (ex, ey) in enumerate(est):
            if i in used:
                continue
            d = math.hypot(ex - tx, ey - ty)
            if best is None or d < best:
                best, bi = d, i
        if bi is not None:
            used.add(bi)
            errs.append(best)
    return sum(errs) / len(errs) if errs else None
