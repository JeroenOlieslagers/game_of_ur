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

## Results

All curves below use one recipe: 240,000 steps, batch 16,384, learning rate
2e-3, two hidden layers, fitted on the whole table with no holdout. Getting to a
single recipe took several passes and the story of that is at the end, because
it changed conclusions rather than just numbers.

### The reference curve

Quantising the table itself, scored by policy regret on 60,000 on-policy
decisions. The **policy floor** -- tabulating only the argmax, no values -- is
**1.433 bits per decision state**.

| bits/state | uniform | Lloyd-Max |
| --- | --- | --- |
| 1 | 1.2141 | 0.6402 |
| 4 | 0.5042 | 0.1741 |
| 6 | 0.0842 | 0.0633 |
| 8 | 0.0076 | 0.0237 |
| 12 | 0.0000 | 0.0018 |

Lloyd-Max wins below about six bits and **loses above seven**. It minimises
squared value error, so it spends code points where the density is and accepts a
ten-point maximum error; uniform keeps every sibling gap resolvable. Value error
and move ordering coming apart, in a setting with no model in it at all.

### The learned curves

Mean regret in win-probability points, on-policy.

| params | Blitz | Finkel | Masters |
| --- | --- | --- | --- |
| ~3.5k | 0.0799 | 0.0783 | 0.0649 |
| ~9k | 0.0470 | 0.0399 | 0.0336 |
| ~26k | 0.0235 | 0.0215 | 0.0161 |
| ~86k | 0.0110 | 0.0100 | 0.0092 |
| ~302k | 0.0066 | 0.0047 | 0.0049 |
| ~1.13M | 0.0033 | 0.0024 | 0.0033 |
| ~4.35M | 0.0032 | **0.0013** | **0.0020** |

Blitz saturates at ~0.0032 -- four times the parameters buys nothing. Finkel and
Masters are still improving at 4.35M.

### The two compressors obey different scaling laws

Against the reference curve, on blitz:

| params | regret | model | equivalent table rate | ratio |
| --- | --- | --- | --- | --- |
| 3,425 | 0.0799 | 0.11 Mbit | 6.04 bits/state | 2275x |
| 25,985 | 0.0235 | 0.83 Mbit | 7.06 | 350x |
| 300,545 | 0.0066 | 9.6 Mbit | 8.11 | 35x |
| 4,347,905 | 0.0032 | 139 Mbit | 8.67 | 3x |

The headline is usually stated as "the network's advantage collapses", which is
true but hides the mechanism. **The equivalent table rate moves only 6.0 to 8.7
bits per state across the entire sweep, while the model rate spans 1265x.**

Quantisation improves roughly *exponentially* in bits -- going from 6 to 9 bits
cuts regret by 40x for 1.5x the storage. The network improves as a *power law*
in parameters. Two different scaling laws, crossing around 8-9 bits per state.
Below the crossing the network is worth thousands of times its size; above it,
storing the table is simply the better compressor and no amount of capacity
changes that.

Stage 1's linear models still hold the far left end: 667 parameters reach 0.1052
where a 3,425-parameter network reaches 0.0799 -- better in absolute terms, worse
per parameter by about five times. Engineered features remain the right choice at
a budget of a few hundred weights.

### The curve, in the units that matter

Regret is the cheap exact proxy; win rate against the exact solution is the
quantity anyone actually cares about. Every model on the curve was compiled into
the engine and played 100,000 games with sides alternating, so the
rate-distortion curve below is *measured* in win-probability deficit rather than
inferred from regret. Standard error is 0.158 throughout.

| params | Blitz | Finkel | Masters |
| --- | --- | --- | --- |
| ~3.5k | 45.15 | 42.98 | 41.38 |
| ~9k | 47.25 | 46.75 | 45.94 |
| ~26k | 48.67 | 48.09 | 47.85 |
| ~86k | 49.43 | 48.94 | 48.64 |
| ~302k | 49.75 | 49.56 | 49.25 |
| ~1.13M | **49.88** | **49.78** | **49.83** |

A 1.13-million-parameter network -- 4.5 MB -- plays every rule set within 0.22
points of a perfect opponent, against tables of 495 MB, 1.66 GB and 6.01 GB. At
this sample size all three top models are within 1.4 standard errors of 50%,
which is to say **indistinguishable from optimal play**. Even 86k parameters
(339 KB) is inside half a point on blitz.

The engine's own regret measurement on each exported model matches the training
script's to three digits, which is what makes these measurements of the model
rather than of a broken export: the Rust encoder is a second implementation of
the Python one, and a mismatch would have been silent.

### The amplification factor is a property of the rule set

Stage 1 found that per-move regret converts to game-level win-rate deficit by a
factor that looked constant within a rule set -- but that rested on two or three
points from one family. It now rests on eighteen, spanning a 30x range of regret
and three orders of magnitude of model size.

| | stage 1 (rule lists, linear models) | stage 2 (networks, 3.5k - 1.1M params) |
| --- | --- | --- |
| Blitz | 62 - 63 | **58** |
| Finkel | 89 - 98 | **98** |
| Masters | 122 - 140 | **138** |

Averaged over points where the deficit exceeds 0.5 (about three standard
errors); below that the deficit is comparable to the noise and the ratio becomes
unmeasurable rather than small. The agreement across completely disjoint model
families is close to exact.

So the conversion holds across **model families**, across **capacity**, and
across a **30x range of regret**. That is what licenses optimising regret: it is
not merely correlated with strength, it is proportional to it, with a constant
that belongs to the rule set. And the constant tracks game length -- Masters
games hold roughly twice the decisions of blitz ones -- which is why regret is
not comparable between rule sets even though it is exact within one.

**One anomaly worth recording.** Blitz's two largest models have essentially
identical regret (0.002947 and 0.002943) but win 49.88% and 49.59% -- a 0.29
point gap against a combined standard error of 0.22. At 1.3 standard errors this
is suggestive rather than established, but if real it means regret stops being
sufficient at the strong end: two models can give up the same average win
probability per move while differing in whether those losses fall on pivotal or
near-tie decisions. Settling it needs about a million games per point, roughly
ten hours each at this network size.

### Model families do not trace one envelope

Same inputs, same evaluation set, blitz.

| family | penalty per parameter | compute |
| --- | --- | --- |
| MLP | -- (sets the envelope) | 1x |
| Transformer | 1.4-2.7x worse | 25-47x |
| Gradient boosting | 9-12x worse | ~2x |

**Boosted trees**: 25,398 nodes -> 0.2608 against an MLP's 25,985 params ->
0.0282. Trees train on 4M of 41M states because boosting is not
minibatch-incremental, so the obvious objection is that they are sample-limited;
rerunning at 20M rows moves nothing (-2.4%, +4.8%, -5.3%, -8.6% at the four
sizes, one in the wrong direction). The mismatch is representational: the
quantities that matter are weighted sums over one-hot indicators, which a dense
layer computes in one operation and a tree approximates with a deep cascade of
axis-aligned splits.

**Attention**: 69,761 params -> 0.0281 against the MLP's 25,985 -> 0.0282. It
needs 2.7x the parameters to match, at 47x the compute. The relational bias is
real but there is nothing here to spend it on: a position is ~34 tokens with no
long-range structure, and what decides moves are aggregates a dense layer
already represents.

**Consequence for the claim.** "This rule set needs N bits" is an upper bound
achieved by the best family tried, not a property of Ur. But since the simplest
family wins, the bound is unlikely to be an artefact of a favoured architecture.

### Engineered features do not wash out

Stage 1's state features concatenated to the input, same recipe both arms:
-23% at ~9k params, -14% at ~85k, -16% at ~565k. The benefit persists at half a
million parameters, which is the interesting part. The likely reason is
`exposure` and `threat`: they are `sum_r P(r) * 1[capture available with roll r]`,
a lookahead over the dice distribution rather than a function of current
occupancy, so recovering them means learning to simulate a roll.

### How much of this is noise

Two runs differing only in seed: 26% apart at 3,425 params, 5% at 25,985, 4% at
563,201. Small models are far more sensitive to initialisation. Any difference
under about 25% at a few thousand parameters, or 5% at half a million, is not a
result. This retired one comparison that had already been drawn.

## What went wrong, and what it cost

Four things were measured wrongly before they were measured rightly. They are
recorded because three of them produced *plausible numbers* rather than errors.

**The learning rate was tuned once and generalised.** A coordinate sweep at
256x2 on blitz chose 5e-3. At 1024x2 it gave 0.0099 and 0.0158 on two seeds
where 2e-3 gives 0.0033; Finkel's largest size came out 9x worse. The tuning job
existed *because* single-configuration results are untrustworthy, and its own
output was a single configuration at one size on one rule set. What caught it was
the anomaly being erratic rather than uniform -- two non-monotone points on one
rule set. A smoothly wrong curve would have been reported.

**Undertraining distorted the shape, not just the level.** 60k steps against
240k costs 13% at 3.4k parameters and 24% at 26k. Larger models sit further from
their asymptote, so the curve bends as well as shifts, and every ratio in the
table above had to be recomputed rather than rescaled.

**The policy head's training set contained its evaluation set.** Both were
sampled on-policy from the opening, so a separately seeded 60k evaluation set was
100% contained in the 400k training sample; a 575k-parameter model scored zero
regret by memorising it. Fixed by enumerating the whole 159M-decision space.
Alongside it, a `u16` legal mask silently aliased the last path position onto
entry-from-hand on the two 16-tile rule sets, because Rust masks an over-wide
shift in release rather than trapping.

**Q2 is unresolved.** The ordering objective failed four times -- memorisation,
softmax saturation on a 0-100 scale, sigmoid saturation of centred residuals, and
a fourth that came back worse still. Each fix was made confidently and twice was
wrong. The value and policy objectives both worked on first correct
implementation, which is what a working objective looks like here. Rather than
report a null result indistinguishable from a fifth bug, **the question is left
open**: whether stage 1's value-versus-ordering finding survives capacity is not
answered by this work.

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
