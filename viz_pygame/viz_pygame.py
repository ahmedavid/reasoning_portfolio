"""
Interactive pygame visualization of the particle filter.

All filtering logic lives in filter.py, the simulated world in simulation.py
-- this module only renders and forwards key presses.

Run:  python viz_pygame.py [options]      (see --help; keys in the HUD/README)
"""

import argparse
import os
import sys
from collections import deque

import numpy as np

from simulation import SIM_DT, Ball, Sensor
from filter import ParticleFilter, mean_match_error


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
