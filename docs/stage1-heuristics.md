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

### Shapley over regret: what actually decides moves

The decomposition above is over R^2. Redone in the currency that matters --
mean regret removed as a feature enters -- over all 36 features. 2^36 subsets is
out of reach, so this samples 120 random permutations (`scripts/shapley_regret.py`);
values sum exactly to the total reduction by construction.

Top eight per rule set, as a share of the total regret reduction:

| rank | Blitz (0.616 -> 0.170) | Finkel (0.910 -> 0.220) | Masters (0.551 -> 0.133) |
| --- | --- | --- | --- |
| 1 | delta_exposure 14.8% | capture_value 17.2% | capture_value 16.8% |
| 2 | delta_exposure_value 13.7% | captures 13.0% | delta_exposure_value 12.8% |
| 3 | threat_self 10.8% | leaves_centre 10.4% | captures 12.1% |
| 4 | safe_opp 9.2% | centre_opp 9.4% | delta_exposure 9.3% |
| 5 | becomes_safe_forever 7.8% | delta_exposure_value 6.1% | threat_self 8.0% |
| 6 | capture_value 6.7% | hand_self 5.7% | captures_frontmost 6.0% |
| 7 | src_was_exposed 6.4% | enters 4.9% | hand_self 4.7% |
| 8 | captures 4.4% | capture_gap_to_front 4.6% | threat_count 4.1% |

Three things stand out.

**The ranking inverts the R^2 one exactly as predicted.** `scored_self` and
`scored_opp`, 48% of explained variance between them, are worth 1.0% and -2.2%
of regret on blitz. `exposure` and `threat`, last for variance, are at the top
for play. This is the same point as the value-versus-ordering fit, now measured
directly on the features rather than inferred from two model fits.

**Every rule set puts capture and exposure first, but weighted differently.**
Finkel's top two are the capture features (30% combined) while its exposure
features are mid-table; blitz and Masters lead with exposure. The reason is
structural: in Finkel a piece is either in the 8-tile war zone or permanently
safe, so the live question is *whether to take the capture available now*. On the
Masters path, where 12 tiles are contested and safety is rarely permanent, the
question is the standing risk you carry, which is what `delta_exposure` measures.

**The magnitude and structure features earn their place.** They were added after
error analysis, and four of them --`delta_exposure_value`, `capture_value`,
`becomes_safe_forever`, `captures_frontmost` -- appear in the top eight of at
least one rule set. `capture_value` alone is 17% of Finkel's total. A binary
`captures` flag really does throw away most of what a capture decision is about.

Small negative values (down to -2.2%) are expected: Shapley values under a
non-monotone objective may be negative, and greedy regret is not monotone in the
feature set because adding a feature can change the argmax adversely at some
positions. Their size bounds the Monte Carlo error at roughly ±0.01 pp.

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

Played against the optimal agent it wins **21.65%** (1,000,000 games, standard
error 0.05). This required porting the rule language into the engine, since
playing a game means choosing moves online. The two implementations -- the
vectorised Python one used for selection, and a recursive-descent parser in Rust
used for play -- agree exactly on the same 60,000 positions: identical counts for
all seven rules and mean regret matching to six decimal places (0.318780). Two
hand-written parsers for one grammar is a place where a divergence yields a
plausible wrong number rather than an error, so the agreement is checked rather
than assumed (`check-rules` against `scripts/check_rules.py`).

### Search depth

Depth applies only to *position* evaluators. Move features and decision lists
score a transition, so at depth greater than one -- where the leaves are
positions -- they have nothing to evaluate.

Expectimax over the four dice outcomes, leaves scored by a 15-weight state
evaluator. The evaluator here is ordering-fitted over a stride-20 sample of
*every* decision in the map (8.0M blitz, 26.3M finkel, 98.5M masters), not the
60,000 on-policy positions used elsewhere, so its depth-1 number differs slightly
from the model table above.

| | d1 | d2 | d3 | d4 | d5 |
| --- | --- | --- | --- | --- | --- |
| blitz `advancement` | 1.2097 | 0.4699 | 0.3687 | 0.3239 | 0.2917 |
| blitz `composite` | 0.5350 | 0.2234 | 0.1834 | 0.1303 | 0.1004 |
| blitz fitted | 0.3031 | 0.1790 | 0.1276 | 0.0958 | **0.0768** |
| finkel `advancement` | 0.8981 | 0.5246 | 0.5019 | 0.4411 | 0.4002 |
| finkel `composite` | 0.6021 | 0.4035 | 0.3595 | 0.2953 | 0.2730 |
| finkel fitted | 0.2825 | 0.2504 | 0.1987 | 0.1478 | **0.1311** |
| masters `advancement` | 0.6508 | 0.2739 | 0.2439 | 0.2120 | 0.1882 |
| masters `composite` | 0.4832 | 0.2535 | 0.2473 | 0.2235 | 0.2020 |
| masters fitted | 0.2006 | 0.1930 | 0.1566 | 0.1132 | **0.0902** |

Depth is monotone in regret everywhere, with no plateau by depth 5 -- the last
step still buys 15-20%. It is not monotone in *agreement*: on Finkel,
`advancement` and `composite` both agree with the optimal move slightly less
often at depth 3 than at depth 2 while having lower regret. Deeper search trades
a few cheap ties for the expensive decisions, which is the trade one wants.

**Search substitutes for capacity, at a rule-set-dependent exchange rate.**
Against the best depth-1 model in the table above (667 parameters), the
15-parameter evaluator at depth 5 is better on blitz (0.0768 against 0.1052),
level on Masters (0.0902 against 0.0922) and worse on Finkel (0.1311 against
0.1136). Finkel resists search for the same reason it resists a linear score: its
safe rosettes create discontinuities that neither smoothing nor averaging over
dice will find.

The exchange rate is not free. The depth-5 sweep takes 23-31 s against under
0.05 s at depth 1-2 -- roughly 1000x the compute per decision, spent to save one
memory lookup per weight. Where a table of 667 weights fits, it is the better
buy; the case for depth is when parameters, not time, are the scarce resource.

Lookahead also substitutes for evaluator quality: `advancement` at depth 2 beats
`composite` at depth 1 on every rule set.

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

Held-out regret against number of terms, from 667 candidates (36 main effects and
630 pairwise products):

| terms | Blitz | Finkel | Masters |
| --- | --- | --- | --- |
| 1 | 0.3855 | 0.5174 | 0.3207 |
| 2 | 0.2963 | 0.4147 | 0.2434 |
| 4 | 0.2471 | 0.3567 | 0.1716 |
| 6 | 0.2223 | 0.3016 | **0.1547** |
| 8 | 0.2002 | 0.2487 | 0.1602 |
| 10 | **0.1843** | 0.2281 | 0.1596 |
| 12 | 0.1789 | **0.2028** | 0.1582 |
| 18 | 0.1700 | 0.1908 | 0.1572 |

The knees are at roughly 10 terms (blitz), 12 (Finkel) and 6 (Masters). Masters
is done after six: terms 7 onward are flat to within noise, and by term 15 greedy
selection is adding terms with no measurable gain at all. At the knee each rule
set is within about 10% of its 37-parameter additive model using a quarter of the
terms, and Finkel's 12-term model (0.2028) actually beats its additive one
(0.2197).

**Greedy selection almost never picks a main effect.** Across 54 selection steps
over the three rule sets, only three main effects were chosen (`captures` and
`enters` on Finkel, `captures` on blitz); everything else is a product. The
selected products are overwhelmingly state x move -- `advancement_opp x
capture_value`, `hand_opp x leaves_centre`, `threat_self x advance` -- which is
the interaction reading it should be: a move feature's worth is conditional on
the position. "Capture when you are behind" is one term, and it is not
expressible as a sum of `advancement_opp` and `capture_value`.

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

Masters has the lowest regret and Finkel the highest, which fits the structure:
Masters' twelve war tiles make captures routine, while Finkel's safe rosettes
create sharper tactical distinctions that a smooth linear score cannot represent.

Do **not** read this column as "Masters is easiest to play". Regret is per
decision and Masters games hold roughly twice as many decisions as blitz ones, so
the ordering reverses at the game level -- see the win-rate section below, where
the Masters models lose more heavily than the blitz ones despite lower regret.

## Regret is not win rate, and regret is not comparable across rule sets

Each policy compiled into the engine and played against the optimal agent with
sides alternating, 1,000,000 games each (standard error 0.05, ceiling 50%).

| Rule set | Model | regret | win % | deficit | deficit / regret |
| --- | --- | --- | --- | --- | --- |
| Blitz | additive (37) | 0.1699 | 39.31 | 10.69 | 63 |
| Blitz | pairwise (667) | 0.1081 | **43.37** | 6.63 | 61 |
| Finkel | additive | 0.2197 | 28.53 | 21.47 | 98 |
| Finkel | pairwise | 0.1151 | **39.46** | 10.54 | 92 |
| Masters | additive | 0.1334 | 33.50 | 16.50 | 124 |
| Masters | pairwise | 0.0930 | **37.37** | 12.63 | 136 |

**Amplification is enormous.** A fifth of a win-probability point given up per
move compounds into a 21-point win-rate deficit on Finkel. Anyone reading
"0.2 pp mean regret" as "nearly optimal" would be badly wrong: a game holds many
decisions and the errors do not cancel.

**The amplification factor is roughly constant within a rule set and very
different between them** -- about 62 on blitz, 95 on Finkel, 130 on Masters.
Within a rule set the deficit is close to linear in regret over the range
measured, which is the useful practical fact: halving regret roughly halves the
deficit, so the cheap exact metric is a sound optimisation target.

That constancy holds across *model families*, not just across sizes of one
family, which is the stronger claim and the one that licenses using regret as an
objective. On Finkel:

| model | regret | deficit | ratio |
| --- | --- | --- | --- |
| 7-rule decision list | 0.3188 | 28.35 | 88.9 |
| additive linear, 37 params | 0.2197 | 21.47 | 97.7 |
| pairwise linear, 667 params | 0.1151 | 10.54 | 91.6 |

A priority list of boolean rules, a linear score and a linear score with
interactions are about as structurally different as three policies can be, and
over a threefold range of regret they sit on one line to within 5%. The
prediction for the decision list from the linear models' factor was ~20%; it
scored 21.65%.

Between rule sets it is not. Regret is a *per-decision* quantity, so converting
it to a game-level deficit multiplies by the number of decisions per game, and
Masters games hold about twice as many as blitz ones. This has a direct
consequence for how the model table should be read: Masters shows the lowest
regret of the three rule sets at every capacity, yet its models lose more badly
than the blitz ones. **Masters is the easiest to approximate per move and the
hardest to play well.** A cross-rule-set regret comparison is meaningless without
the game-length factor, and the earlier reading of Masters as "easiest" was a
per-move statement being mistaken for a strength statement.

**Interactions buy far more game strength than regret suggests.** Going additive
to pairwise cuts Finkel regret by 48% but converts a 21.5-point deficit into a
10.5-point one -- the model goes from losing three games in four to nearly even.
On blitz the pairwise model reaches 43.4% against a perfect opponent, which for
667 weights against a 41-million-entry table is the headline number of this
stage.

The relationship shows no sign of the sub-linearity guessed at earlier; over
0.09-0.22 pp of regret it is linear within each rule set. That may still bend
closer to optimal, but nothing here measures it.

## What is not yet done

- **Blocking and per-piece geometry** have no feature. Two positions with the
  same aggregate can differ in whether the opponent's route is obstructed, and
  nothing in the 36 sees it. The N-tuple network reads board configurations
  directly and did add something on top of the scalars, which suggests the gap is
  real -- and it is a natural thing for stage 2 to recover on its own.
- **Depth beyond 7.** Depths 1-7 are running; regret was still falling 15-20%
  per ply at depth 5 with no sign of saturation, and depth 6 confirms it
  continues (Finkel `advancement` 0.4047 -> 0.3324). Each ply costs about 13.5x
  the last, so depth 8 needs the search rewritten with move ordering and pruning
  rather than a bigger allocation.
- **A hybrid** of move features at the root over a position evaluator at the
  leaves. Currently move-based policies are root-only and position evaluators get
  the depth axis; the combination is untested and is the obvious best-of-both.

## Reproducing

```
royalur_analysis dump-move-features --ruleset finkel --positions 60000 > mv.csv
scripts/shapley_regret.py mv.csv --permutations 120     # feature importance
scripts/prune_model.py mv.csv --terms 18                # complexity frontier
scripts/export_policies.py mv.csv policies/finkel       # weights for the engine
royalur_analysis winrate --ruleset finkel --weights policies/finkel/policy_pairwise.txt --games 1000000
royalur_analysis depth-regret --ruleset finkel --max-depth 5
```

Cluster wrappers for all of these are in `cluster/`. Every number in this
document is reproducible from a solved map plus these commands; none of them
takes more than a few minutes except the depth-5 sweep and the 1M-game runs.
