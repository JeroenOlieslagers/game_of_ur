# Stage 1: how well can a simple heuristic play?

All numbers are Finkel unless stated. Everything is measured against the exact
solution, so there is no Monte Carlo noise in any value.

## The metric

For a position where `m*` is optimal and `m` is the model's choice,

```
regret = V(s o m*) - V(s o m)      in win-probability points
```

Mean regret is the headline; the 95th percentile and the move-agreement rate
(fraction of positions where the optimal move is chosen) are reported alongside,
because a model can have low mean regret and occasional catastrophic errors.

Regret is exact per position, so it needs no sampling: `regret-all` computes it
over **every** decision in the map -- 526,203,867 of them for Finkel -- in about
19 minutes. Where a sample is used, it is on-policy (states drawn from
optimal-vs-optimal play), because that is the distribution in which strength is
actually decided.

Regret is not the same as win rate. It is a per-move quantity under a fixed state
distribution, so it cannot see error compounding, nor the fact that a weaker
agent visits different positions, nor that some positions are pivotal. Win rate
against the optimal agent remains the ground truth; regret is the cheap exact
proxy.

## Baselines

| | regret | p95 | agreement |
| --- | --- | --- | --- |
| uniform random | 2.1795 | 10.53 | 36.5% |
| first legal move | 1.2409 | 5.59 | 44.6% |

**"First legal move" is not a neutral baseline.** The engine lists the scoring
move first, then entry from hand, then pieces by increasing path position, so
"take the first legal move" is really *score, else enter, else move your least
advanced piece* -- a real strategy, and why it beats random by a factor of 1.8.
Uniform random is the honest zero point.

## The features

36 in three tiers, all from the mover's perspective.

**State features (14)** describe the position after the move:
`advancement_self/opp`, `scored_self/opp`, `hand_self/opp`, `safe_self/opp`,
`exposure_self`, `threat_self`, `centre_self/opp`, `frontmost_self/opp`.

Every one is paired self/opponent. That is required, not cosmetic: the two
players' perspectives are related by a reflection about 50, and only an
antisymmetric weight vector is consistent under it.

**Move features (12)** describe the transition, so unlike state features every
one varies between the candidate moves of a position: `advance`, `captures`,
`scores`, `enters`, `lands_rosette`, `lands_centre`, `leaves_centre`,
`dest_safe`, `src_was_exposed`, `delta_exposure`, `delta_threat`, `keeps_turn`.

**Magnitudes (4)**, added after error analysis showed capture decisions carried
the most regret while every capture feature was binary: `capture_value`,
`rescue_value`, `delta_exposure_value`, `delta_threat_value`.

**Structure (6)**, which no aggregate can express -- two positions with identical
totals differ in *which* piece is capturable: `captures_frontmost`,
`capture_gap_to_front`, `moves_frontmost`, `becomes_safe_forever`,
`contact_possible`, `threat_count`.

## Results

60,000 on-policy positions, 178,416 candidate moves.

| Model | params | regret | p95 | agreement |
| --- | --- | --- | --- | --- |
| state features, **value fit** | 15 | 0.3083 | 1.67 | 71.8% |
| state features, ordering fit | 15 | 0.2307 | 1.33 | 74.5% |
| move features (original 12) | 13 | 0.2842 | 1.41 | 73.7% |
| move features (all 22) | 23 | 0.2363 | 1.37 | 74.8% |
| state + move (original) | 27 | 0.2304 | 1.35 | 74.6% |
| state + move (all) | 37 | 0.2135 | 1.26 | 75.4% |
| state + move (original) + pairwise | 352 | 0.1330 | 0.84 | 80.2% |
| **state + move (all) + pairwise** | 667 | **0.1136** | **0.73** | **81.7%** |
| decision list, 6 hand predicates | 6 rules | 0.398 | | |
| decision list, from 933 generated rules | 7 rules | **0.319** (held out) | | |

Interaction order, on the 14 most influential features: order 1 → 0.2928,
order 2 → 0.2250, order 3 → 0.2112.

### Fitting the value is the wrong objective

The first two rows have identical features and parameter counts. The value fit
has the **higher R^2** (0.9812 against 0.9718) and plays **25% worse**
(0.3083 against 0.2307).

The mechanism: what decides a move is the *ordering* of scores among one
position's successors, so anything constant across those siblings is irrelevant
however much variance it explains. `scored_self` carries 24% of explained
variance but is nearly identical for every candidate move -- a given move rarely
scores. Least squares spends its capacity getting the level right.

The fix is a within-position transform: subtract each position's mean from the
design and the target, which annihilates exactly the constant components. It is
still one least-squares call.

### Variance explained ranks features almost backwards

Exact Shapley decomposition of R^2 over all 2^14 subsets, computed from one Gram
matrix accumulated over all 137,870,097 non-terminal states. Values sum to
R^2 = 0.948202 and are all non-negative, as nesting requires.

| feature | share of explained variance |
| --- | --- |
| scored_self | 23.9% |
| scored_opp | 23.7% |
| hand_self | 17.4% |
| hand_opp | 16.4% |
| ... | |
| threat_self | 0.9% |
| exposure_self | 0.6% |

Score and pieces-in-hand carry 81% of the variance and hardly vary within a
position; `threat` and `exposure` are the bottom two for variance yet are among
the few features that discriminate between moves. Fitting the ordering roughly
inverts this ranking: `exposure_self` goes from -0.3 to -5.0 units of
advancement, `threat_self` from +2.6 to +5.9, while `scored_self` is pulled down
from +22.9 to +19.8.

### Interpretable weights

From the full-population value fit, in units of a square of advancement:

- a scored piece: **+22 squares** -- more than walking a piece the whole 14-square
  path, because it cannot be undone
- the centre rosette: **+12 squares**, quantifying the folk wisdom about that
  square
- an available capture: +7; exposure to capture: -1.4

### A readable strategy

Greedy selection over 933 rules proposed in a small DSL, assembled into a
priority list, selection on half the positions and reported on the other half:

1. `capture_value + 3 * lands_rosette - 5 * delta_exposure`
2. `(delta_threat > 0.2 or captures) and exposure_self < 0.35`
3. `not leaves_centre and delta_exposure <= 0`
4. `4 * enters + 3 * lands_rosette - 5 * exposure_self`
5. `(lands_centre or delta_exposure_value < -4)`
6. `delta_threat > 0.35`
7. `delta_exposure > 0 and delta_threat > 0.2`

Held-out regret 0.319 pp, against 0.398 for a list built from hand-written
predicates. Rules are parsed by a recursive-descent parser rather than evaluated,
so generated text is safe to run; of 992 proposals, 0 were rejected and 59 never
discriminated.

### Search depth

Depth applies only to *position* evaluators. Move features and decision lists
score a transition, so at depth greater than one -- where the leaves are
positions -- they have nothing to evaluate.

Depths 1/2/3: `advancement` 0.852/0.540/0.508, `composite` 0.570/0.375/0.349,
ordering-fitted 0.218/0.213/0.146. Lookahead does substitute for evaluator
quality (`advancement` at depth 2 beats `composite` at depth 1), but the
ordering-fitted evaluator at depth 1 beats every hand-tuned heuristic at depth 3.

## Model complexity: a frontier, not a criterion

Information criteria do not fit here. BIC penalises a likelihood, and our loss is
a decision loss; penalising the squared-error fit would regularise the objective
already shown to be wrong. There is also little in-sample optimism at these
parameter counts, and no "true model" to recover.

Instead the frontier is traced directly: for each number of terms, the lowest
regret greedy forward selection reaches, selecting **on regret**. Each candidate
fit is a submatrix solve of one precomputed Gram, so a step over hundreds of
candidates costs no extra data passes.

Optimism does enter through the *selection*: choosing the best of hundreds of
candidates repeatedly is itself a fit. Positions are split, selection runs on
one half, the curve is measured on the other.

Use the full model for feature importance and the pruned model as the reportable
heuristic. Do not read importance off the pruned model: among correlated
features greedy selection picks one of a redundant pair arbitrarily, so "not
selected" does not mean "unimportant".

## On dimensionality reduction

PCA does not help here, for a reason worth stating rather than assuming.

It finds directions of maximum **variance**, and this whole analysis shows
variance is the wrong currency: the high-variance directions are exactly the
aggregate quantities that do not vary between the candidate moves of a position.
PCA would rediscover `scored` and `hand`.

More basically, principal-components regression is a *constrained* linear model,
so on the same features it can never beat unconstrained least squares. It earns
its place when p > n or the design is ill-conditioned, and here there are 178,416
rows against 667 parameters.

A supervised reduction targeting the response (PLS, or reduced-rank regression on
the ordering objective) would be the apt version, and could be worth it purely as
interpretation -- "move choice lives in a 3-dimensional subspace" is a claim
worth testing. But for performance, greedy selection on regret already *is* the
right dimensionality reduction: it reduces dimensions in the currency we care
about.

## All three rule sets

Mean regret, 60,000 on-policy positions each. The findings replicate.

| Model | params | Blitz | Finkel | Masters |
| --- | --- | --- | --- | --- |
| state features, value fit | 15 | 0.2730 | 0.3083 | 0.2541 |
| state features, ordering fit | 15 | 0.2300 | 0.2307 | 0.1669 |
| move features (original 12) | 13 | 0.2610 | 0.2842 | 0.2769 |
| move features (all 22) | 23 | 0.2674 | 0.2363 | 0.2604 |
| state + move (original) | 27 | 0.2211 | 0.2304 | 0.1580 |
| state + move (all) | 37 | 0.2173 | 0.2135 | 0.1410 |
| state + move (original) + pairwise | 352 | 0.1418 | 0.1330 | 0.1203 |
| **state + move (all) + pairwise** | 667 | **0.1052** | **0.1136** | **0.0922** |

The value fit loses to the ordering fit in every rule set despite having the
higher R^2 in every one; on Masters it is 34% worse. Interactions roughly halve
regret everywhere.

Masters is the easiest to approximate and Finkel the hardest, which fits the
structure: Masters' twelve war tiles make captures routine, while Finkel's safe
rosettes create sharper tactical distinctions that a smooth linear score cannot
represent.

## Regret is not win rate, and the gap is large

The additive Finkel model gives up 0.213 percentage points of regret per move.
Played against the optimal agent with sides alternating, it wins **28.76%**
(20,000 games, standard error 0.35, ceiling 50%).

So a fifth of a point per move compounds into a **21-point** win-rate deficit --
roughly a hundredfold amplification, because a game holds many decisions and the
errors do not cancel. Anyone reading "0.2 pp regret" as "nearly optimal" would be
badly wrong.

This is the clearest argument for reporting win rate and not only regret. Larger
runs across all models and rule sets are in progress; with several (regret, win
rate) pairs the relationship itself becomes measurable rather than assumed, and
it is likely sub-linear at the strong end, since a model that rarely errs also
rarely compounds errors.

## What is not yet done

- **Win rate against the optimal agent** for these models. Regret is a proxy; win
  rate is the ground truth. It needs each model implemented inside the engine,
  since playing a game means choosing moves online.
- **Shapley over regret** rather than over R^2. The machinery exists; only the
  R^2 decomposition has been run at scale.
- **Blitz and Masters.** Everything above is Finkel. Masters has 12 war tiles
  against Finkel's 8 and blitz grants an extra roll for a capture, so the feature
  ranking should genuinely differ.
- **Depths beyond 3**, running.
- **Blocking and per-piece geometry** have no feature. The N-tuple network reads
  board configurations directly and did add something on top of the scalars,
  which suggests this gap is real.
