# Stage 2: how many bits is a solved game worth?

Stage 1 asked which *features* carry the signal at fixed small capacity. Stage 2
turns capacity into the free variable and asks for a curve.

## The framing: this is one plot, and stage 1 is its left end

A solved rule set is a table. Blitz is 41,254,034 entries at 12 bytes; Finkel
137,892,016; Masters 500,981,472 -- 495 MB, 1.66 GB and 6.01 GB. Any policy that
plays the game without that table is a lossy compression of it, and the honest
way to compare policies is by the two numbers every compressor has: **rate**
(bits of model) and **distortion** (how much win probability it gives up).

Seen that way stage 1 already produced points on this curve, and extreme ones.
The 667-parameter blitz model is 2.7 KB against 495 MB -- a compression ratio of
185,000x, or 5.2 x 10^-4 bits per state -- and it wins 43.4% against a perfect
opponent. Stage 2 does not start a new comparison; it extends the same axes to
the right.

| model | bytes | bits/state (blitz) | ratio vs blitz table |
| --- | --- | --- | --- |
| linear, 15 params | 60 | 1.2e-5 | 8,250,000x |
| linear + pairwise, 667 | 2.7 KB | 5.2e-4 | 185,000x |
| net, 100k params | 400 KB | 0.078 | 1,240x |
| net, 1M params | 4 MB | 0.78 | 124x |
| net, 10M params | 40 MB | 7.8 | 12x |
| the table | 495 MB | 96 | 1x |

The deliverable is one figure per rule set: log rate on x, regret on y, one curve
per model family, anchored on the left by stage 1's linear models and on the
right by the table itself at zero distortion.

## What makes this more than a benchmark

**There is a computable reference curve.** Every learned model can be compared
against simply *storing the table at fewer bits*. Quantise each value to b bits,
take the argmax over siblings from the quantised values, measure the regret
exactly. That is a real rate-distortion curve for the same object, obtained with
no model at all, and it costs one pass over the map per b. Any learned model that
does not beat it is not doing anything.

Two more reference lines fall out of the same idea:

- **The policy table.** Perfect play does not need values, only the argmax at
  each decision state. Tabulating that costs about 2 bits per decision state,
  which is the true floor for perfect play by tabulation -- far below 96 bits and
  the fairest "you must beat this" line.
- **Lloyd-Max on the exact value distribution.** Because the whole population is
  in hand, the optimal b-level scalar quantiser is computable exactly rather than
  estimated. This is the memoryless coding bound; everything below it is
  structure a model can exploit.

**The labels are exact and cover the entire domain.** 500M noiseless examples, no
sampling error, no distribution shift in principle. The usual excuse for a
disappointing curve -- not enough data -- is unavailable.

**The distortion has an absolute meaning.** Not "held-out loss" but win
probability given up against a known-optimal opponent, and via stage 1's measured
amplification factors (62 blitz, 95 Finkel, 130 Masters) it converts to a
game-level win-rate deficit. The plots get a second y-axis in deficit units, with
the conversion validated by direct simulation at two or three points per family.

## The four questions

Only the first is engineering. The rest are the reason to do this.

**Q1. What is the smallest model that plays within 1 point of optimal?** A
quotable per-rule-set number, and the headline.

**Q2. Does the value/ordering divergence survive capacity?** Stage 1's central
result is that fitting value is the wrong objective: the value fit had higher R^2
and played 25-34% worse. But that must be a small-capacity phenomenon -- as value
error goes to zero, regret must follow. So the gap closes somewhere, and *where*
is a real measurement. If it persists to 1M parameters, "train on the ordering"
is a general lesson rather than a quirk of linear models.

**Q3. Do the model families trace the same curve?** If MLPs, gradient boosting
and N-tuple networks land on one envelope, that envelope is a property of the
game and the claim "Finkel needs about N bits to play at 49%" is about Ur. If
they separate persistently, the differences are inductive bias and the claim is
about the models. Either answer is worth reporting, and this is the deepest
question of the stage.

**Q4. Where does the residual error live?** Exactly answerable here. Do
compressed models err on near-tie positions, where being wrong is nearly free, or
on pivotal ones? Regret already weights by consequence, but the joint
distribution of (sibling value gap, error rate) says whether a small-model policy
is *safe-but-dull* or *occasionally catastrophic* -- and stage 1's p95 column
hints these differ.

## Model families

| family | capacity knob | range | why it is here |
| --- | --- | --- | --- |
| linear + interactions | terms | 15 - 667 | stage 1; anchors the left end |
| N-tuple network | tuples x bank size | 10^3 - 10^7 | reads board geometry directly; the one family that can see blocking |
| gradient boosting | trees x depth | 10^3 - 10^6 | strong on tabular features; a fair test of whether engineered features suffice |
| MLP | width x depth | 10^4 - 10^7 | the general function approximator |

The N-tuple network is the interesting middle: it is a lookup table with tied
entries, so it interpolates continuously between stage 1's parametric models and
the raw table, and it is the one family that sees per-tile configuration rather
than aggregates. Stage 1 flagged blocking and per-piece geometry as the features
nothing captures -- this is where that gap should show up as a measurable gain.

## Input representation

Per the roadmap, and this is where the result is won or lost:

- per-tile occupancy over 20 usable tiles x {empty, self, other} = 60
- pieces in hand for both players, one-hot 0-7 = 16
- 76 inputs; scores are derivable as `pieces - hand - on_board`

Two structural wins available cheaply:

- The stored encoding is already **self-other symmetric** (always "self to
  move"), so the symmetry is free rather than learned.
- **Index tiles along each player's own path**, not by board index, so "position
  5 of my path" means the same thing for both colours. Choosing the coordinate
  frame is worth more than a layer.
- Optionally concatenate stage 1's 14 state features. A hybrid usually wins at
  small parameter counts, which is exactly the regime of interest -- and it makes
  the left end of the curve continuous with stage 1 rather than a separate story.

## Protocol

**Training distribution.** Compression and play want different ones, and saying
so is part of the result. Uniform over stored states is correct for compressing
the table; it over-weights positions no game reaches. On-policy is correct for
play strength. Train both at three capacities; expect the difference to matter at
small capacity and vanish at large, for the same reason as Q2.

**Holdout.** Hold out states, but be honest about what it measures. With 100k
parameters against 137M states memorisation is impossible and the train/test gap
will be nil; the holdout is insurance, not the point. It becomes load-bearing at
the right-hand end, where 10M parameters against 41M blitz states can genuinely
start to memorise.

**Metrics.** Mean and max absolute error in win-probability points (the
compression view), policy regret and move agreement (the play view), and win rate
against the optimal agent for the best model of each family. Report all four;
stage 1 showed they disagree.

## The reflection trap

Any new evaluator must be checked against all three of these before its numbers
mean anything. Each caused a real bug in stage 1.

1. Converting a value from the mover's perspective to a fixed one is a
   **reflection about 50** (`v -> 100 - v`), never a negation. A network trained
   as "self to move" returns the *successor's* mover value, so for a move that
   passes the turn the mover's value is `100 - net(successor)`.
2. The constant matters. If a fit estimates `V - 100 * passed`, that 100 must be
   added back before sibling moves are compared, or every turn-passing move is
   penalised by a full 100 points.
3. Features must be paired self/opponent so the weight vector can be
   antisymmetric under the reflection.

The check: score a position and its colour-swapped mirror; the two must sum to
100. Add this as an assertion in the evaluation harness, not as a manual step.

## Infrastructure to build

1. **`dump-tensors`** (Rust) -- stream shards of packed (board occupancy, hands,
   value) to binary. 8 bytes per example packed: 5 for 20 tiles at 2 bits, 1 for
   both hands, 2 for an f16 value. Masters is then 4 GB, which mmaps comfortably.
2. **`dump-successors`** (Rust) -- for an evaluation set of positions, the
   successor encodings and their exact values. This makes regret evaluation
   *model-agnostic*: Python scores the successors, takes the argmax, and computes
   regret without the model ever entering Rust. Essential for XGBoost, which
   would be painful to reimplement.
3. **`quantise-regret`** (Rust) -- the reference curve. One pass per bit depth.
4. **Forward pass in Rust** for the single best model only, to get a 1M-game win
   rate through the existing `winrate` path. An MLP is about 30 lines of matmul;
   the N-tuple is a lookup. Not needed for the sweep.

Note the division: (2) makes the whole capacity sweep cheap and framework-neutral,
and (4) is paid once at the end. Building (4) first would be the mistake.

## Compute

Torch has H200 (8/node), H100 (4/node), L40S and RTX6000, plus `anaconda3/2025.06`.

The models are small and the data is the cost. A 340k-parameter MLP at batch
65,536 processes the full Masters map in roughly two minutes per epoch on one
H200, so a capacity sweep is minutes to hours, not days -- and single-GPU jobs
schedule far more easily than multi-GPU ones. The binding constraint is streaming
4 GB per epoch from scratch, so shard once and mmap.

Long jobs go on the cluster. Local runs get killed.

## Order of work

1. `dump-tensors` and `dump-successors`, verified with the reflection check.
2. `quantise-regret` -- the reference curve, before any model. It is a day's work
   and it sets the bar everything else is judged against.
3. MLP sweep on Finkel, uniform target, value objective. Establishes the pipeline.
4. Add the ordering objective and the on-policy distribution -- Q2.
5. N-tuple and gradient boosting sweeps -- Q3.
6. All three rule sets, error analysis (Q4), win rate for the best of each family.

Stages 2 and 3 remain a separate project from the current paper, whose claim is
that these rule sets are solved exactly. Folding a compression study into it would
unbalance it.
