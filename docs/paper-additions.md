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

## 5a. Correction: an offset bug inflated every Python-side regret number

The ordering fit moves a constant to the target side: it estimates
`V - 100 * passed`, where `passed` marks a move that hands the turn over. That
constant has to be added back before sibling moves are compared, because
`passed` **differs between the candidate moves of one position** -- a move onto a
rosette keeps the turn while others pass it. Omitting it penalised every
turn-passing move by 100 points, so the policy grabbed rosettes.

The symptom was visible and I misread it: the chooser took a turn-keeping move
33.8% of the time against the optimal move's 24.4%. I wrote that up as "the
model over-values rosettes", treating my own bug as a finding about the model,
and used it to motivate new features.

| Model | as first reported | corrected |
| --- | --- | --- |
| additive, 30 features | 0.2642 pp | **0.2225 pp** |
| with pairwise interactions | 0.1944 pp | **0.1230 pp** |
| turn-keeping picks | 0.338 | 0.252 (optimal 0.244) |

Every regret number produced by the Python harness was inflated; the numbers
from the Rust `regret` and `regret-all` commands were always correct, since they
reflect through `100 - value` exactly as the table does. Fitted weights were also
correct -- only the evaluation was wrong.

## 5. The fitted linear model is the strongest heuristic

Mean move regret in win-probability points, 200,000 on-policy states per rule
set, corrected code:

| Heuristic | Blitz | Finkel | Masters |
| --- | --- | --- | --- |
| random | 1.685 | 2.177 | 1.117 |
| advancement | 1.224 | 0.905 | 0.650 |
| score_race | 0.993 | 0.901 | 0.588 |
| safety | 0.595 | 0.891 | 0.493 |
| centre | 0.534 | 0.584 | 0.429 |
| exposure | **0.502** | 0.640 | **0.428** |
| composite | 0.521 | 0.626 | 0.465 |
| fitted (least squares) | 0.533 | **0.526** | 0.525 |

Move agreement (fraction of positions where the optimal move is chosen) tells a
different story: the fitted model is **highest everywhere** — 68.7% on blitz,
62.3% on Finkel, 60.8% on Masters — while `exposure` leads on mean regret for
blitz and Masters.

So the fitted model picks the best move most often, but when it errs it errs by
more. That is expected: least squares on values optimises neither agreement nor
regret directly. It is the cleanest remaining evidence for the theoretical point
that squared value error is not the loss that governs move choice, and motivates
the within-position centred fit.

A good heuristic gives up roughly half a win-probability point per move against
perfect play, against 1.1-2.2 for random.

Caveat on the ladder: `lead` currently reports identically to `advancement` on
all three rule sets. The definitions do differ (`lead` adds a pieces-in-hand
term) but they can only disagree when entering a piece ties exactly against a
two-square move, so it is a redundant rung and should be replaced with a more
distinct feature combination.

**Retraction.** An earlier version of this note claimed the opposite — that the
fitted model was a *worse* move chooser than plain advancement, and drew a
conclusion about value-fitting being the wrong objective. That was an
implementation bug, not a finding:

- A score is the *mover's* win percentage, and converting it to a fixed
  perspective is a reflection about 50 (`v -> 100 - v`), not a negation. The code
  negated it.
- The fit dropped the intercept, on the reasoning that a constant shifts all
  candidate moves equally. That is false here: a move onto a rosette keeps the
  turn while other moves pass it, so the reflection applies to some successors
  and not others, and the constant does not cancel. The fitted model had the
  largest intercept, so it was hurt most.

Both are fixed, and the feature set is now fully paired self/opponent so that
weight vectors can be antisymmetric, which is what the reflection requires.

The *theoretical* point still stands and is worth a sentence: squared value error
is not the loss that determines move choice, since what matters is the ordering
of sibling moves. It simply does not bite here — least squares already produces
the best evaluator tested. Whether a within-position centred fit (which
annihilates everything constant across siblings) improves on it is open.

## 6. Ranking of heuristics differs by rule set

Blitz benefits far more from centre control and capture threat than Finkel does,
consistent with the rules: blitz grants an extra roll for a capture, so
aggression compounds in a way it does not in Finkel.

Numbers to be regenerated for all three rule sets with the corrected code.

Supports the point that heuristic weights must be fitted per rule set.

## Still to do

- Shapley decomposition of regret reduction over the twelve features, to rank
  them by contribution to *play strength* rather than to variance explained.
- Search depth: regret against ply depth for each heuristic, expected to show a
  weak evaluator with deeper lookahead beating a strong one at 1 ply.
- Error analysis on the worst-regret states, to motivate further features.
