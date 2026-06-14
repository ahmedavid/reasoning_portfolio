"""
Ground-truth world: ball physics and the noisy, unreliable sensor.

No filtering logic in here -- this module only produces the "reality" the
particle filter (filter.py) has to estimate.
"""

import math
from collections import deque

G = 9.81          # gravity [m/s^2]
SIM_DT = 1 / 60   # simulation time step [s]

# velocity prior for unknown launches (speed <= ~35 m/s, launched upwards)
VEL_PRIOR = dict(vx=(-35.0, 35.0), vy=(0.0, 35.0))


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
    with random (and manually togglable) complete dropouts.

    `params` is any object with attributes obs_noise, obs_interval,
    dropout_prob, dropout_len (e.g. argparse namespace or SimpleNamespace).
    """

    def __init__(self, rng, params):
        self.rng = rng
        self.noise = params.obs_noise
        self.interval = params.obs_interval
        self.dropout_prob = params.dropout_prob
        self.dropout_len = params.dropout_len
        self.next_obs_t = 0.0
        self.dropout_until = -1.0
        self.forced_dropout = False   # toggled with key D in the pygame UI

    def in_dropout(self, t):
        return self.forced_dropout or t < self.dropout_until

    def observe(self, t, balls):
        """Returns a list of noisy (x, y) tuples (empty if the sensor works
        but nothing is airborne) or None if no report arrives now (not
        scheduled, or complete sensor dropout)."""
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
