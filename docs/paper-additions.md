# Candidate additions to the paper

Results produced after the current draft, with a note on what the draft already
says, so nothing is duplicated or contradicted.

## 1. The simulation differences are exactly binomial noise

**Already in the draft:** a TOST equivalence test over 500 tournaments of 10
million games each, giving a mean difference of 0.00055 with 95% CI
[-0.0006, 0.0017] against an equivalence margin of +/-0.002, and
`r^2 = 0.9999996`. That establishes the mean difference is equivalent to zero.

**Not in the draft, and stronger:** the *distribution* of differences matches
binomial sampling noise exactly, so the residual disagreement is fully accounted
for by finite sample size, with no systematic component left over.

At 2,000 states x 1,000,000 games per state (SE = 0.0500 percentage points):

| Rule set | mean abs. error | predicted `sigma*sqrt(2/pi)` | within 3 SE | normal theory |
| --- | --- | --- | --- | --- |
| Blitz | 0.0402 pp | 0.0399 pp | 99.8% | 99.73% |
| Finkel | 0.0371 pp | 0.0399 pp | 99.8% | 99.73% |
| Masters | 0.0398 pp | 0.0399 pp | 99.8% | 99.73% |

The claim to make: not merely that theory and simulation agree on average, but
that every deviation is explained by sampling noise alone.

Note the design differs from the draft's: 2,000 states x 1e6 games rather than
500 x 1e7. More points with slightly wider error bars, which suits the scatter
panel; the draft's design is tighter per point. Either can be quoted, but the
numbers above belong with the 2,000-state design.

## 2. Solve times, and that refining is not cheaper

The solver now reports honest from-scratch times: only the published maps' state
*keys* are used, with every value starting at a flat 50%.

| Rule set | From scratch | Refining published values |
| --- | --- | --- |
| Blitz | 31.5 s | 26.8 s |
| Finkel | 200.5 s | 223.2 s |
| Masters | 884.1 s | 880.9 s |

16 cores of an Intel Xeon Platinum 8592+ node. Both initialisations reach the
same fixed point, agreeing on the start-position value to all ten printed digits.

The point worth making: starting four orders of magnitude further from the
solution costs a *constant* number of extra sweeps (137 against 103 on Masters'
last layer), not a multiple of the work, because convergence is geometric.

## 3. Where the winning actually is (roadmap stage 0)

An agent that plays uniformly at random until the position reaches score layer
*k*, and optimally from then on, against a fully optimal opponent.

Finkel pilot, 2,000 games per stage:

| Stage | Layer | Win % |
| --- | --- | --- |
| 0 | (6,6) | 0.2 |
| 20 | (1,1) | 20.6 |
| 25 | (0,2) | 25.0 |
| 26 | (0,1) | 33.3 |
| 27 | (0,0), fully solved | 47.5 -> 50 |

Layer `(0,0)` alone is worth 14.2 points, and the last two layers together
account for 22 of the 47.5. Playing randomly only until the first piece is
scored costs 16.7 points against an optimal opponent.

**So most of the winning is decided in the early game, not the endgame** — the
opposite of the naive reading. Note explicitly that the layer order is chosen for
correctness (dependency order), so this is a statement about the game, not an RL
learning curve.

## 4. Twelve features explain ~90% of the win probability

Least squares of the exact value on twelve hand-built features
(`docs/analysis-roadmap.md` lists them), fitted on on-policy states:

| Rule set | R^2 |
| --- | --- |
| Blitz | 0.873 |
| Finkel | 0.900 |
| Masters | 0.902 |

The weights are directly interpretable in win-probability points. On Finkel,
expressed in squares of advancement:

- a scored piece: **+22 squares** — more than walking a piece the whole
  14-square path, because it cannot be undone
- the central rosette: **+12 squares**, quantifying the folk wisdom about that
  square
- an available capture: +7; being exposed to capture: -1.4

## 5. Fitting values is the wrong objective for choosing moves

The finding most worth reporting, because it is counterintuitive: the fitted
model, despite explaining ~90% of the variance, is a **worse** move chooser than
plain advancement.

Mean move regret in win-probability points, 200,000 on-policy states:

| Rule set | best heuristic | `advancement` | least-squares `fitted` |
| --- | --- | --- | --- |
| Blitz | composite **0.278** | 0.612 | 1.285 |
| Finkel | advancement **0.546** | 0.546 | 0.634 |
| Masters | composite **0.301** | 0.344 | 0.832 |

The explanation: the terms that dominate the regression (`scored_self`,
`scored_opp`, about +/-10 pp) are nearly *constant across the candidate moves
within a position* — a given move rarely scores — so they carry most of the
variance between positions while contributing nothing to ordering within one.
Advancement is the mirror image: a weak absolute predictor but a strong
discriminator between siblings.

The correct objective for move choice is the *difference* in value between
sibling moves. See `docs/analysis-roadmap.md`.

## 6. Ranking of heuristics differs by rule set

Blitz gains most from the composite heuristic (0.278 against advancement's 0.612,
a factor of 2.2), while on Finkel advancement alone is already the best of the
hand-built set. This is consistent with the rules: blitz grants an extra roll for
a capture, so threat and centre control compound in a way they do not in Finkel.

Supports the point that heuristic weights must be fitted per rule set.

## Still to do

- Shapley decomposition of regret reduction over the twelve features, to rank
  them by contribution to *play strength* rather than to variance explained.
- Search depth: regret against ply depth for each heuristic, expected to show a
  weak evaluator with deeper lookahead beating a strong one at 1 ply.
- Error analysis on the worst-regret states, to motivate further features.
