# How the Solution Works — Conceptual Overview

## The problem in one sentence

Several balls fly through the air at the same time; a bad sensor occasionally
reports their positions — noisy, unlabeled, sometimes not at all — and from
this we must continuously estimate where every ball is and how fast it is
moving, without ever knowing where or how the balls were launched.

## The big idea: belief as a swarm of guesses

We cannot compute the probability distribution over ball states analytically
(it is non-Gaussian and has several peaks — one per ball), so we *represent it
by samples*. The filter maintains thousands of **particles**, where each
particle is one complete guess about one ball: "a ball is at position (x, y)
and flies with velocity (vx, vy)". Where many particles agree, the filter
believes a ball is; where particles are spread out, the filter is uncertain.

The crucial design decision: a particle describes **one** ball, not all n
balls at once. The full belief about all n balls is simply the whole cloud,
which forms n clumps (modes) on its own. This is what makes the approach
scale: tracking five balls does not need a 20-dimensional state, just one
shared swarm of 4-dimensional guesses. It also means the case "two balls
launched almost identically" and the case "two balls launched completely
differently" are not different cases at all — the clumps just happen to
overlap or not.

## Starting from ignorance

At the start nothing is known except that balls launch somewhere inside a
50×50 m area with a plausible throwing speed. So the initial particles are
scattered uniformly over that area with random upward velocities — a fog of
hypotheses. The first few measurements rapidly burn this fog away: only the
guesses compatible with what the sensor saw survive.

## The heartbeat: predict — weigh — resample

The filter repeats three steps forever:

**1. Predict (physics).** Every particle flies like a real ball would:
gravity pulls it down, position follows velocity, over exactly the time that
really passed. A pinch of random noise is added so the guesses stay diverse.
This step runs *always*, measurement or not — which is precisely why the
filter keeps producing estimates while the sensor is dead: the swarm simply
keeps flying along believable parabolas, spreading out slowly to reflect
growing uncertainty.

**2. Weigh (compare with the sensor).** When a measurement set arrives, each
particle asks: "how well does the *nearest* measurement match me?" and
receives a weight accordingly. Using the nearest measurement is how we deal
with indistinguishable balls — no particle needs to know *which* ball it
follows, only that *some* ball supports it. Two corrections are applied on
top, both following from things we genuinely know:

- a ball cannot be below the ground, so guesses underground are impossible;
- each measurement comes from exactly one ball, so the particle mass is
  rebalanced to give every measurement an equal share. Without this, the
  clump with the most particles would slowly steal particles from the others
  until one ball is forgotten (mode starvation).

**3. Resample (survival of the fittest).** Once the weights become too
unequal, particles are redrawn proportionally to weight: bad guesses die,
good guesses multiply. Done only when needed, so diversity is preserved.

## Self-healing: noticing what the swarm cannot explain

If a measurement appears far away from *every* particle, the filter concludes
that something is flying that it has no hypothesis for (a ball it lost, or —
with relaunching enabled — a freshly thrown one). It reacts by spawning a
batch of new particles around that measurement with wide-open velocity
guesses. One or two observations later, physics has filtered out the wrong
velocities and the new clump tracks the ball. The filter therefore recovers
from its own failures instead of diverging.

## The life and death of a ball, as the filter sees it

A ball is born for the filter when its measurements first carve a clump out
of the fog. It lives as a clump that flies a parabola alongside the real
ball. When the real ball lands, its measurements stop; the clump flies on,
hits the known ground, and dies — its particles are handed over to the balls
still in the air. The filter counts how many balls are alive purely from how
many measurements arrive at once (balls are only visible while flying), so
the number of reported estimates rises and falls without ever peeking at the
truth.

## From a cloud to n concrete answers

The swarm *is* the answer, but the task demands n positions and velocities.
Since the density is multimodal, taking one global average would be nonsense
(it would land between the balls). Instead the particles are clustered
(weighted k-means with k = number of active balls, warm-started from the
previous answer so clusters keep their identity over time). Each cluster
center is one ball's estimated position; the weighted average velocity of the
cluster's particles is its estimated velocity vector.

## Why this is robust, in two sentences

Every piece of prior knowledge — gravity, the ground, the launch area, the
sensor's noise level, "one measurement per ball" — is encoded exactly once,
either in the transition model, the evaluation model, or the clustering.
Everything else (where the balls are, how many are flying, how fast they
move, whether the sensor currently works) is *inferred*, and every failure
mode has a recovery path: impossible guesses are recycled, starving clumps
are refed, unexplained evidence spawns new hypotheses.
