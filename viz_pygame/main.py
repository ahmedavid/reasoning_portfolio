"""
Portfolio Exam 2 -- Sensor Fusion: Particle Filter
Reasoning and Decision Making under Uncertainty, Summer 2026, THWS

Estimates positions and velocity vectors of n simultaneously flying balls
from noisy, unlabeled position observations using a particle filter.

Design overview
---------------
State / particles:
    Each particle is ONE ball hypothesis (x, y, vx, vy). The full posterior
    over all n balls is represented by the (multimodal) particle cloud as a
    whole, NOT by a joint 4n-dimensional state. This keeps the state space
    small, requires no data association in the state, and handles both the
    case of very similar and clearly different launch parameters with one
    common approach: similar balls simply share (overlapping) modes.

Transition model:
    Ballistic motion (gravity, no drag) integrated over the true elapsed
    time since the last filter step + Gaussian process noise (scaled with
    sqrt(dt)). Because prediction uses the real elapsed time, variable
    observation intervals and complete sensor dropouts are handled
    naturally: during dropout the filter only predicts.

Evaluation (measurement) model:
    Observations are an unordered set of noisy (x, y) positions -- the balls
    are indistinguishable. A particle is weighted with the likelihood of the
    NEAREST measurement (max over per-measurement Gaussian likelihoods),
    i.e. an implicit nearest-neighbour data association on particle level.
    A small floor probability makes the filter robust against outliers and
    missing detections. Measurements that no particle explains (a freshly
    launched ball, or a mode lost during a long dropout) trigger targeted
    injection of new hypotheses around that measurement.

Resampling:
    Systematic (low variance) resampling, triggered only when the effective
    sample size drops below half the particle count. After resampling a
    small fraction of particles is re-injected around current measurements
    (and uniformly in the launch area) so lost modes can be re-acquired.

Estimating n positions from the multimodal density:
    Weighted k-means (k = n) on the particle positions, seeded with the
    previous estimates for temporally stable cluster identities. Cluster
    centroids = position estimates, per-cluster weighted mean velocity =
    velocity estimates.

Run:  python main.py [options]      (see --help; keys listed in README)
"""

import argparse
import math
import os
import sys
from collections import deque

import numpy as np

# ----------------------------------------------------------------------------
# Command line parameters
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Particle filter ball tracking")
    p.add_argument("--balls", type=int, default=3, help="number of balls n >= 1")
    p.add_argument("--particles", type=int, default=4000, help="number of particles")
    p.add_argument("--obs-noise", type=float, default=3.0,
                   help="std dev of observation noise [m]")
    p.add_argument("--obs-interval", type=float, default=0.15,
                   help="mean time between observations [s] (jittered +-30%%)")
    p.add_argument("--dropout-prob", type=float, default=0.02,
                   help="probability per observation that a sensor dropout starts")
    p.add_argument("--dropout-len", type=float, default=1.5,
                   help="duration of a random sensor dropout [s]")
    p.add_argument("--launch-area", type=float, default=50.0,
                   help="side length of the unknown launch area [m]")
    p.add_argument("--relaunch", action="store_true",
                   help="balls relaunch after landing (default: they stay down "
                        "and tracking of them stops)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--headless", type=int, default=0, metavar="FRAMES",
                   help="run FRAMES frames without a window and print errors (testing)")
    return p.parse_args()


G = 9.81          # gravity [m/s^2]
SIM_DT = 1 / 60   # simulation time step [s]

# velocity prior for unknown launches (speed <= ~35 m/s, launched upwards)
VEL_PRIOR = dict(vx=(-35.0, 35.0), vy=(0.0, 35.0))


# ----------------------------------------------------------------------------
# Ground truth simulation
# ----------------------------------------------------------------------------
class Ball:
    """One real ball with unknown (to the filter) launch parameters."""

    def __init__(self, rng, launch_area, relaunch=False):
        self.rng = rng
        self.launch_area = launch_area
        self.relaunch = relaunch
        self.trail = deque(maxlen=400)
        self.launch()

    def launch(self):
        r = self.rng
        self.x = r.uniform(0.0, self.launch_area)
        self.y = r.uniform(0.0, self.launch_area)
        speed = r.uniform(15.0, 32.0)
        angle = r.uniform(math.radians(25), math.radians(155))  # roughly upward
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        self.airborne = True
        self.relaunch_timer = 0.0
        self.trail.clear()

    def step(self, dt):
        if self.airborne:
            self.vy -= G * dt
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.trail.append((self.x, self.y))
            if self.y <= 0.0 and self.vy < 0.0:   # hit the ground
                self.y = 0.0
                self.airborne = False
                self.relaunch_timer = 1.5         # seconds until new launch
        elif self.relaunch:                   # default: landed balls stay down
            self.relaunch_timer -= dt
            if self.relaunch_timer <= 0.0:
                self.launch()


class Sensor:
    """Produces unlabeled noisy position observations at variable intervals,
    with random (and manually togglable) complete dropouts."""

    def __init__(self, rng, args):
        self.rng = rng
        self.noise = args.obs_noise
        self.interval = args.obs_interval
        self.dropout_prob = args.dropout_prob
        self.dropout_len = args.dropout_len
        self.next_obs_t = 0.0
        self.dropout_until = -1.0
        self.forced_dropout = False   # toggled with key D

    def in_dropout(self, t):
        return self.forced_dropout or t < self.dropout_until

    def observe(self, t, balls):
        """Returns a list of noisy (x, y) tuples or None if no measurement now."""
        if t < self.next_obs_t:
            return None
        # schedule next observation: variable time span (jitter +-30 %)
        self.next_obs_t = t + self.interval * self.rng.uniform(0.7, 1.3)
        if self.in_dropout(t):
            return None
        # maybe start a new random dropout period
        if self.rng.random() < self.dropout_prob:
            self.dropout_until = t + self.dropout_len
            return None
        obs = [(b.x + self.rng.gauss(0, self.noise),
                b.y + self.rng.gauss(0, self.noise))
               for b in balls if b.airborne]
        self.rng.shuffle(obs)        # balls are indistinguishable
        return obs                   # may be []: "alive, but zero detections"


# ----------------------------------------------------------------------------
# Particle filter
# ----------------------------------------------------------------------------
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
        self.estimates = None            # (n_balls, 4) after first update
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
        # 4 sigma is "unexplained" (e.g. a newly launched ball after the old
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
        """Weighted k-means (k = n_balls) on particle positions to resolve the
        multimodal density into n ball estimates. Seeded with the previous
        estimates so cluster identities are stable over time.
        Returns (n_balls, 4): x, y, vx, vy and the particle->cluster labels."""
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


# ----------------------------------------------------------------------------
# Visualization (pygame)
# ----------------------------------------------------------------------------
WIN_W, WIN_H = 1280, 720
WORLD = dict(xmin=-70.0, xmax=170.0, ymin=-6.0, ymax=110.0)

CLUSTER_COLORS = [(80, 180, 255), (255, 140, 80), (140, 255, 120),
                  (255, 110, 200), (255, 230, 90), (170, 130, 255),
                  (120, 230, 230), (240, 240, 240)]


def world_to_screen(x, y):
    sx = (x - WORLD["xmin"]) / (WORLD["xmax"] - WORLD["xmin"]) * WIN_W
    sy = WIN_H - (y - WORLD["ymin"]) / (WORLD["ymax"] - WORLD["ymin"]) * WIN_H
    return int(sx), int(sy)


class App:
    def __init__(self, args):
        self.args = args
        self.headless = args.headless > 0
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        import pygame  # imported here so --help works without pygame installed
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Particle Filter - Ball Tracking (THWS P2.1)")
        self.font = pygame.font.SysFont("consolas,monospace", 15)
        self.clock = pygame.time.Clock()

        import random
        self.rng = random.Random(args.seed)
        self.reset()

        self.paused = False
        self.show_particles = True
        self.err_log = []

    def reset(self):
        a = self.args
        self.t = 0.0
        self.balls = [Ball(self.rng, a.launch_area, a.relaunch)
                      for _ in range(a.balls)]
        self.sensor = Sensor(self.rng, a)
        self.pf = ParticleFilter(a.particles, a.balls, a.launch_area,
                                 a.obs_noise, a.seed, a.relaunch)
        self.last_obs = None
        self.obs_history = deque(maxlen=60)
        self.labels = np.zeros(a.particles, dtype=int)
        self.estimates = None

    # -- one simulation + filter step ----------------------------------------
    def step(self):
        for b in self.balls:
            b.step(SIM_DT)
        self.t += SIM_DT

        # transition model runs every step -> estimation also during dropout
        self.pf.predict(SIM_DT)

        obs = self.sensor.observe(self.t, self.balls)
        if obs is not None:           # a report arrived (may be empty)
            self.pf.update(obs)
            if obs:
                self.last_obs = obs
                self.obs_history.append((self.t, obs))

        self.estimates, self.labels = self.pf.estimate()

        err = mean_match_error(self.estimates, self.balls)
        if err is not None:
            self.err_log.append(err)

    # -- input handling --------------------------------------------------------
    def handle_key(self, key):
        pg = self.pg
        a = self.args
        if key == pg.K_SPACE:
            self.paused = not self.paused
        elif key == pg.K_r:
            self.reset()
        elif key == pg.K_d:
            self.sensor.forced_dropout = not self.sensor.forced_dropout
        elif key == pg.K_p:
            self.show_particles = not self.show_particles
        elif key in (pg.K_PLUS, pg.K_KP_PLUS, pg.K_EQUALS):
            a.obs_noise = min(15.0, a.obs_noise + 0.5)
            self.sensor.noise = self.pf.sigma = a.obs_noise
        elif key in (pg.K_MINUS, pg.K_KP_MINUS):
            a.obs_noise = max(0.5, a.obs_noise - 0.5)
            self.sensor.noise = self.pf.sigma = a.obs_noise
        elif key == pg.K_PERIOD:
            a.obs_interval = min(2.0, a.obs_interval + 0.05)
            self.sensor.interval = a.obs_interval
        elif key == pg.K_COMMA:
            a.obs_interval = max(0.05, a.obs_interval - 0.05)
            self.sensor.interval = a.obs_interval
        elif key == pg.K_UP:
            a.balls += 1
            self.reset()
        elif key == pg.K_DOWN and a.balls > 1:
            a.balls -= 1
            self.reset()

    # -- drawing ----------------------------------------------------------------
    def draw(self):
        pg = self.pg
        scr = self.screen
        scr.fill((12, 14, 22))

        # ground
        gx0 = world_to_screen(WORLD["xmin"], 0)
        gx1 = world_to_screen(WORLD["xmax"], 0)
        pg.draw.line(scr, (70, 70, 80), gx0, gx1, 2)
        # launch area
        la = self.args.launch_area
        p0 = world_to_screen(0, la)
        p1 = world_to_screen(la, 0)
        pg.draw.rect(scr, (40, 48, 60),
                     pg.Rect(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1]), 1)

        # particles, colored by k-means cluster
        if self.show_particles:
            for i in range(self.pf.N):
                c = CLUSTER_COLORS[self.labels[i] % len(CLUSTER_COLORS)]
                dim = (c[0] // 3, c[1] // 3, c[2] // 3)
                sx, sy = world_to_screen(self.pf.P[i, 0], self.pf.P[i, 1])
                if 0 <= sx < WIN_W and 0 <= sy < WIN_H:
                    scr.set_at((sx, sy), dim)

        # recent observations (yellow crosses, fading)
        for ot, obs in self.obs_history:
            age = self.t - ot
            if age > 1.5:
                continue
            f = max(0.15, 1 - age / 1.5)
            col = (int(250 * f), int(220 * f), int(60 * f))
            for zx, zy in obs:
                sx, sy = world_to_screen(zx, zy)
                pg.draw.line(scr, col, (sx - 4, sy - 4), (sx + 4, sy + 4), 1)
                pg.draw.line(scr, col, (sx - 4, sy + 4), (sx + 4, sy - 4), 1)

        # true balls + trails (white)
        for b in self.balls:
            pts = [world_to_screen(x, y) for x, y in b.trail]
            if len(pts) > 1:
                pg.draw.lines(scr, (90, 90, 100), False, pts, 1)
            if b.airborne:
                pg.draw.circle(scr, (255, 255, 255), world_to_screen(b.x, b.y), 5)

        # estimates: colored circles + velocity arrows
        if self.estimates is not None:
            for j, e in enumerate(self.estimates):
                c = CLUSTER_COLORS[j % len(CLUSTER_COLORS)]
                sx, sy = world_to_screen(e[0], e[1])
                pg.draw.circle(scr, c, (sx, sy), 9, 2)
                # velocity arrow (length ~ 0.8 s of flight)
                ex, ey = world_to_screen(e[0] + e[2] * 0.8, e[1] + e[3] * 0.8)
                pg.draw.line(scr, c, (sx, sy), (ex, ey), 2)

        # HUD
        a = self.args
        drop = "DROPOUT (sensor failed)" if self.sensor.in_dropout(self.t) else "ok"
        err = self.err_log[-1] if self.err_log else float("nan")
        lines = [
            f"t={self.t:6.2f}s  balls(n)={a.balls}  particles={a.particles}  "
            f"obs-noise sigma={a.obs_noise:.1f}m  obs-interval~{a.obs_interval:.2f}s  "
            f"sensor: {drop}  tracked balls: {self.pf.k_active}",
            f"mean position error: {err:5.2f} m   (white=truth, yellow x=observation, "
            f"colored=particles/estimates+velocity)",
            "keys: SPACE pause | R reset | D dropout | P particles | +/- noise | "
            ",/. interval | UP/DOWN #balls | ESC quit",
        ]
        for i, txt in enumerate(lines):
            scr.blit(self.font.render(txt, True, (200, 200, 210)), (10, 8 + 18 * i))

        pg.display.flip()

    # -- main loop ----------------------------------------------------------------
    def run(self):
        pg = self.pg
        frames = 0
        running = True
        while running:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_ESCAPE:
                        running = False
                    else:
                        self.handle_key(ev.key)
            if not self.paused:
                self.step()
            if not self.headless:
                self.draw()
                self.clock.tick(60)
            frames += 1
            if self.headless:
                if frames % 300 == 0:
                    recent = self.err_log[-200:]
                    m = sum(recent) / len(recent) if recent else float("nan")
                    print(f"frame {frames:5d}  t={self.t:6.2f}s  "
                          f"mean err (recent) = {m:.2f} m")
                if frames >= self.args.headless:
                    running = False
        pg.quit()
        if self.err_log:
            n = len(self.err_log)
            late = self.err_log[n // 4:]                  # skip initial convergence
            print(f"overall mean position error after convergence: "
                  f"{sum(late)/len(late):.2f} m  ({n} samples)")


if __name__ == "__main__":
    sys.exit(App(parse_args()).run())
