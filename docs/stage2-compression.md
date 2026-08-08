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

## Architecture

The value network is the reference, not the answer. Three things vary.

**Input representation.** The baseline is per-tile one-hot over 20 slots plus
both hands, 76 inputs. Two known weaknesses: it discards the *ordinal* meaning of
a position, so "how far along my path" has to be relearned as a weighted sum of
independent tile symbols; and it has no relational structure, while capture,
blocking and exposure are all relations *between* pieces. Path indexing fixes the
first for free -- "position 5 of my path" means the same thing for both colours.

**Head.** A value head must represent V accurately enough to resolve the *sibling
gap*, which stage 1 found is often below 1e-3 win-probability points. A **policy
head** -- logits over the mover's own path indices, given the position and the
roll, with illegal sources masked -- only has to get the *sign* of that gap
right, and it chooses a move in one forward pass rather than one per successor.
This is the direct architectural consequence of stage 1's value-versus-ordering
result, and the prediction is that it dominates on regret per parameter.

Keep the two claims separate when writing up: a policy head is not a value
function, so it cannot sit at the leaves of a depth search and cannot be compared
against the quantisation reference on equal terms. Value models anchor the
rate-distortion curve; the policy head answers the different question of what the
cheapest thing that *plays* well is.

**Attention.** A set encoder over path positions -- one token per path position
per player, self-attention between them, positional embeddings carrying
distance-to-home -- supplies exactly the relational bias the dense stack lacks,
at ~34 tokens and negligible cost. This is the sharp version of Q3: if a
relational bias moves the envelope, the envelope was about the model; if it does
not, it was about the game.

### What does not apply

**A VAE.** There is no generative task and no distribution to sample: state ->
value is a deterministic function for which exact labels exist everywhere. The KL
term buys a smooth stochastic latent with nothing to use it. A plain autoencoder
over the table would technically be compression, but the function-approximation
framing strictly dominates it, because it exploits input structure the
autoencoder would have to rediscover.

**A sequence model over game history.** The game is Markov and fully observed, so
history is redundant by construction.

**Learning the dynamics.** Predicting the next state burns capacity on a function
that is already known exactly and costs nanoseconds to compute.

## Protocol

**Training distribution.** Compression and play want different ones, and saying
so is part of the result. Uniform over stored states is correct for compressing
the table; it over-weights positions no game reaches. On-policy is correct for
play strength. Train both at three capacities; expect the difference to matter at
small capacity and vanish at large, for the same reason as Q2.

**No holdout.** The task is to reproduce *this* table in fewer bits, and every
state in it is part of the object being compressed. Withholding a slice would
mean compressing 95% of the object and reporting error on the other 5%, which
measures interpolation to unseen positions -- a real question, but a different
one, and not the one a rate-distortion curve answers. Models are fitted on the
whole table.

The exception is a **memorisation control** at the top of the sweep. "Compression"
is only a meaningful word while the parameter count is far below the state count;
at 10M parameters against 41M blitz states that stops being obviously true, and a
holdout is the way to check. Run it there, as a diagnostic on a specific claim,
not as a blanket protocol. At the sizes measured so far it is moot -- 563k
parameters against 41M states is 73 states per parameter, and holdout error
tracked fitted error.

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

## Results so far (blitz)

Preliminary. The convergence control has not finished, so every number is "what
this recipe reached", not "what this capacity can do" -- see the caveats at the
end, which are load-bearing rather than decorative.

### The reference curve

Quantising the table itself, scored by policy regret on 60,000 on-policy
decisions. The policy floor -- the cost of tabulating only the argmax -- is
**1.433 bits per decision state**.

| bits/state | uniform | Lloyd-Max |
| --- | --- | --- |
| 1 | 1.2141 | 0.6402 |
| 4 | 0.5042 | 0.1741 |
| 5 | 0.2173 | 0.1131 |
| 6 | 0.0842 | 0.0633 |
| 8 | 0.0076 | 0.0237 |
| 12 | 0.0000 | 0.0018 |

Lloyd-Max wins below about six bits and **loses above seven**. It minimises
squared value error, so it spends code points where the density is and accepts a
ten-point maximum error; uniform keeps every sibling gap resolvable. Another
instance of value error and move ordering coming apart, now in a setting with no
model in it at all.

### The learned curve

MLP, value objective, 60k steps on a GPU.

| params | regret | equivalent table rate | ratio vs table |
| --- | --- | --- | --- |
| 3,425 | 0.0841 | ~6 bits/state | 2,260x |
| 25,985 | 0.0282 | ~7 bits/state | 347x |
| 563,201 | 0.0049 | ~8.5 bits/state | 20x |

**The advantage collapses by two orders of magnitude across the curve.** At low
rate a network exploits structure the table does not encode, and that structure
is worth enormous amounts; at high rate there is little left to find and it
competes with raw storage on storage's terms. Where the learned and quantisation
curves meet is the honest measure of how much of the game is *compressible*
rather than merely tabulated.

Stage 1's linear models still hold the extreme low end: 667 parameters reach
0.1052, against 0.0841 for a 3,425-parameter network. Better in absolute terms,
worse per parameter by about five times -- engineered features remain the right
choice at a budget of a few hundred weights.

### Value head against policy head

Both at 60k steps, the policy head trained on all 158,968,482 decisions.

| params | value | params | policy |
| --- | --- | --- | --- |
| 3,425 | **0.0841** | 4,209 | 0.1158 |
| 25,985 | **0.0282** | 29,073 | 0.0450 |
| 563,201 | **0.0049** | 575,505 | 0.0111 |
| 2,174,977 | 0.0090 (unstable) | 2,199,569 | 0.0058 |
| — | — | 8,593,425 | **0.0029** |

**The value head wins at every matched capacity and the gap widens with size**
(27% to 56%). The reason is that scoring successors is not merely a cost of one
forward pass per candidate -- it is a free ply of search, using dynamics the
model does not have to learn. A policy head sees only the position and the roll
and must internalise those dynamics. The advantage compounds as the evaluator
becomes accurate enough to exploit it.

A second effect runs the other way: **the policy head trains stably where the
value head does not.** Its curve is monotone to 8.6M parameters while the value
head regressed at 2.2M and timed out above that. Cross-entropy over a discrete
class appears better conditioned at scale than BCE against a continuous target,
so at the top of the sweep the comparison is confounded by trainability rather
than by information.

### Engineered features do not wash out

Stage 1's 14 state features concatenated to the input, same recipe both arms:

| params | without | with | change |
| --- | --- | --- | --- |
| ~9k | 0.0790 | 0.0609 | -23% |
| ~85k | 0.0279 | 0.0241 | -14% |
| ~565k | 0.0134 | 0.0112 | -16% |

The benefit persists at half a million parameters, which is the interesting
outcome. The likely reason is `exposure` and `threat`: they are
`sum_r P(r) * 1[capture available with roll r]`, a lookahead over the dice
distribution rather than a function of current occupancy, so recovering them
means learning to simulate a roll. They were also top of the stage-1
Shapley-over-regret ranking.

### Model families do not trace one envelope

Gradient boosting on the same inputs and the same evaluation set, capacity swept
as trees x depth, with parameters counted as tree nodes:

| nodes | regret | MLP at comparable size |
| --- | --- | --- |
| 1,550 | 0.3142 | — |
| 25,398 | 0.2608 | 0.0282 (25,985 params) |
| 303,086 | 0.1786 | — |
| 3,626,490 | 0.0730 | 0.0058 (2.2M params) |

**Boosted trees are 9-12x worse per parameter than a dense net**, and that
understates it: a tree node stores a feature index, a threshold and two child
pointers, so it costs more than one float while being counted as one.

Boosting is not minibatch-incremental, so this trained on 4M of 41M states,
which raises the obvious objection that the trees are sample-limited rather than
capacity-limited. Rerunning at 20M rows moves nothing: -2.4%, +4.8%, -5.3%,
-8.6% at the four sizes, one of them in the wrong direction. The gap is real.

The mechanism is a mismatch between the target and axis-aligned splits. The
input is one-hot occupancy, and the quantities that matter -- total advancement,
number of safe pieces -- are weighted sums over many indicators. A dense layer
computes one in a single operation; a tree approximates it with a deep cascade
of splits. This is a statement about representation, not about boosting being a
weak learner.

**Consequence for how stage 2 should be stated.** A claim like "this rule set
needs N bits" survives only as an upper bound achieved by the best known family,
not as a property of Ur. The rate-distortion curve measured here describes dense
networks; the reference quantisation curve describes the table. Whether some
third family sits below both is open, and the transformer sweep is the next test
of it.

### How much of this is noise

Two runs differing only in seed:

| params | run A | run B | spread |
| --- | --- | --- | --- |
| 3,425 | 0.1413 | 0.1087 | 26% |
| 25,985 | 0.0437 | 0.0461 | 5% |
| 563,201 | 0.0129 | 0.0134 | 4% |

Small models are far more sensitive to initialisation. Any difference under
about 25% at a few thousand parameters, or under 5% at half a million, is not a
result. This retired one comparison that had already been drawn.

### Caveats

- **Not converged.** The same 3,425-parameter model scored 0.1413 at 12k steps
  and 0.0841 at 60k. Every point is an upper bound on distortion, and the
  looseness may vary with capacity, which distorts the *shape* rather than
  merely shifting the curve.
- **The value head above 563k parameters is unreliable** -- 2.2M came out worse
  than 563k, which means undertraining rather than a capacity limit.
- **The ordering objective is not comparable.** It is trained on a sampled set
  rather than the full space, so its rate axis measures a different object. Its
  regret falls to zero while its value error *rises* to 21.8 points, which is the
  signature of memorising a ranking.
- **The pointer head result is uninformative.** It was built without the
  per-move features that were its entire justification, so it adds parameters
  without information and loses accordingly.

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
