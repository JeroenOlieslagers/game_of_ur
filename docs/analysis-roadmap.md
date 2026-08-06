# Analysis roadmap

Planned analyses that build on the solved maps, in the order they should be
done. Each stage reuses infrastructure from the one before it.

The through-line: because these rule sets are *exactly solved*, every agent can
be scored against ground truth in absolute win-probability points. That is
unusual — normally "how good is this agent" has no exact answer — and it is what
makes each of these analyses worth doing.

## Board facts the analyses depend on

Verified against the path definitions in `rust/src/main.rs`.

- Board indices are `(y - 1) * 3 + (x - 1)`; 20 usable tiles. Indices 12, 14, 15
  and 17 are off-board.
- **Rosettes: 0, 2, 10, 18, 20.** Each player owns two private rosettes; index 10
  (the centre) is contested. Rosettes grant an extra roll in every rule set.
- **Finkel (Bell path, 14 tiles):** the shared war zone is the 8 middle-lane
  tiles `{1,4,7,10,13,16,19,22}`. Each player's 6 private tiles are safe, and
  rosettes additionally cannot be captured. The last path square is the player's
  own exit rosette.
- **Masters (16 tiles):** the war zone is **12** tiles — `{1,4,7,10,13,16,19}`
  plus `{18,20,21,22,23}`. The path detours through the opponent's back row, so
  both exit rosettes and the whole top row are contested. Masters is structurally
  far more violent than Finkel.
- **Blitz** uses the Masters path with 5 pieces, and captures grant an extra
  roll, which rewards aggression much more strongly.

Consequence: heuristic weights must be fitted **per rule set**, never shared.

## Stage 0 — value of optimal play by game stage

The cheapest analysis, and the one to do first: it needs no training and no
change to the solver, only the final maps and the existing simulation harness.

Score layers are solved in descending score order, so the *late* game is solved
first. An agent using a partially solved map therefore plays optimally from the
moment it scores a piece, and each completed layer moves the "optimal from here
on" boundary earlier in the game.

Rather than instrument the solver, define directly from the **final** map:

> agent *k* = plays uniformly at random while in a stage before *k*, and
> optimally from stage *k* onward.

Evaluate each agent *k* against the fully optimal agent, alternating light and
dark so the converged agent sits at exactly 50%.

This avoids a subtle trap. Initialising a partial map to a flat 50% would make
`choose_optimal_move` break ties by move order — a deterministic, positionally
biased policy, *not* random play. The curve would then measure "always take the
first legal move", which is not what anyone would think it measured.

- **x axis: cumulative state expansions** (sweeps x layer size) needed to solve
  up to stage *k*, on a log scale, with layer boundaries marked. Layer *index* is
  a poor axis: blitz layers range from 277 to 14,993,636 states, so an index axis
  compresses almost all the work into the final tick. Expansions also make Jacobi
  and Gauss-Seidel directly comparable on one plot.
- **y axis: win percentage against the optimal agent**, floor given by a fully
  random agent, ceiling exactly 50%.
- Add a second series where the unsolved stages use a weak heuristic rather than
  random play — random is a very low floor and overstates what the solved map
  buys.

The increments between consecutive *k* measure **the marginal value of playing
well in stage *k***, which is a statement about the game, not about the
algorithm. Say so explicitly in any write-up: the layer order is chosen for
correctness (dependency order), not to maximise improvement per step, so the
shape must not be read as an RL learning curve.

Start with blitz.

## Stage 1 — heuristic strength comparison

### Metric

With the full LUT available, strength can be measured **exactly, without
simulation**. For a sampled state `s`, where `m*` is optimal and `m_h` is the
heuristic's choice:

```
regret(s) = V(s o m*) - V(s o m_h)      in win-probability points
```

Report mean regret, 95th percentile, max, and move-agreement rate (fraction with
zero regret). Millions of states can be scored in seconds with no Monte Carlo
noise.

**Sample from two distributions and report both**, because they answer different
questions and can disagree substantially:

- **uniform over stored states** — quality as a function approximator. Note this
  over-weights positions that essentially never occur in play.
- **on-policy** — states drawn from simulated optimal-vs-optimal games — quality
  where it matters. This is the one that predicts game strength.

Mean per-move regret does not linearly predict win rate: errors compound and some
positions are pivotal. Keep a small game-level win-rate check as an anchor.

### Feature catalogue

*Progress and material*

1. `advancement` — sum of own pieces' path indices, plus `score x path_len`.
2. `scored` — pieces safely home. Should carry a much larger weight than raw
   advancement, being irreversible.
3. `pieces_in_hand` — flexibility against being behind; sign is not obvious,
   good candidate for fitting.
4. `lead` — own advancement minus opponent's. Racing games are about
   differentials, so this usually beats absolute advancement.

*Safety*

5. `safe_fraction` — own pieces on uncapturable tiles. Far more informative in
   Finkel than in Masters.
6. `capture_exposure` — `sum_r P(r) * 1[opponent can legally capture with roll r]`.
   Exact and cheap: for each opponent piece at their path index `j`, the
   destination is `opp_path[j + r]`; check collisions against own pieces,
   respecting the safe-rosette rule. Likely the strongest non-obvious feature.
7. `weighted_exposure` — as above, weighted by the advancement that would be
   lost. A piece one square from home is worth far more than a freshly entered one.

*Tempo*

8. `on_rosette` — occupying any rosette; an extra roll is close to free tempo.
9. `centre_control` — a piece on index 10 specifically: extra roll, blocks the
   opponent's only route, and in Finkel it is immune to capture.
10. `expected_extra_rolls` — `sum_r P(r) * 1[best reply lands on a rosette]`.

*Structure*

11. `blockade` — occupying the opponent's entry tile or exit approach.
12. `spacing` — own pieces sitting 1-4 squares ahead of an enemy piece, i.e.
    within capture range.
13. `frontmost_distance` — how close the lead piece is to home; decides pure races.

*Rule-set specific*

14. Blitz: `capture_tempo` — captures grant an extra roll, so aggression
    compounds; blitz weights should push much harder on offence.
15. Masters: `back_row_risk` — pieces on `{20,21,22,23}` are exposed where Finkel
    would keep them safe.

### The ladder

`random` -> `advancement` -> `lead` -> `+scored weight` -> `+safety` ->
`+centre` -> `+exposure` -> fitted linear model.

**Add search depth as a second axis.** Every heuristic above is 1-ply. A weak
evaluator with 2-ply expectimax over the next roll often beats a strong evaluator
at 1-ply, and dice keep the branching modest. A regret grid over
(heuristic x depth) is far more informative than a single ladder.

### Fitted linear model

Least squares of `V(s)` on the feature vector, then measure regret. The fitted
weights are interpretable — "a scored piece is worth N squares of advancement",
"the centre rosette is worth M" — which is publishable game analysis rather than
engineering. Fit on **on-policy** states; fitting on uniform samples tunes the
weights to positions nobody reaches. Consider fitting to move ordering (rank or
regret) rather than value, since ordering is what determines play.

## Stage 2 — neural network compression of the lookup table

Frame this as **rate-distortion**: parameters against error. The headline is a
curve, not a single number.

| Rule set | Table | 100k-param net (fp32) | Ratio |
| --- | --- | --- | --- |
| Blitz | 41,254,034 values, ~82 MB at 2 bytes | 400 KB | ~200x |
| Masters | 500,981,472 values, ~1 GB | 1 MB (1M params) | ~1000x |

What makes this a good problem rather than a demo: **exact labels over the entire
domain**. 500M noiseless examples, no distribution shift in principle. Still hold
out states, to distinguish generalisation from memorisation.

### Input representation

This is where the result is won or lost. Start with:

- per-tile occupancy, one-hot over the 20 usable tiles x {empty, self, other} = 60
- pieces in hand for both players, one-hot 0-7 = 16
- 76 inputs total; scores are derivable as `pieces - hand - on_board`

Two structural wins available cheaply:

- The stored encoding is already **self-other symmetric** (always "self to
  move"), so the network models a single perspective and the symmetry is free
  rather than learned.
- **Index tiles along each player's own path** instead of by board index, so
  "position 5 of my path" means the same thing for both colours. This is the
  equivalent of choosing a good coordinate frame.
- Optionally concatenate the engineered features from stage 1. A hybrid usually
  wins at small parameter counts, which is exactly the regime of interest.

Output a sigmoid trained with BCE against the LUT probability as a soft target;
better calibrated than MSE.

### Metrics

Mean and max absolute error in win-probability points, **and** policy regret from
stage 1. A network can have tiny MSE while misordering near-equal moves, and
ordering is where play strength lives. Sweep 3-4 sizes for the curve.

## Stage 3 — supervised against reinforcement learning

Same network, three training regimes, each scored against the exact optimum.

| Regime | Expectation |
| --- | --- |
| Supervised on the LUT | Best by a wide margin: exact dense labels |
| RL against an optimal opponent | Worst, and possibly no learning at all |
| Self-play | In between, and the most compute-hungry |

Two problems to plan for.

**RL against a perfect opponent is a known-bad exploration setting.** Nearly
every game is a loss, so the reward signal is almost constant and gradients
vanish.

The fix is to **reuse the score-layer hierarchy as a curriculum**: start training
on high-score layers, where episodes are short and outcomes are dense and
decisive, then walk backwards towards `(0,0)`. This is the same dependency
ordering the solver uses, repurposed as a difficulty schedule, and it turns one
sparse-reward problem into a sequence of tractable ones. The `epsilon` machinery
already built for the random-play sweep also gives a ready-made opponent
curriculum (an epsilon-corrupted optimal player).

**Self-play needs chance-node handling.** Dice mean vanilla AlphaZero-style MCTS
does not transfer directly; expectimax or explicit chance nodes are required.
That is real implementation work, not a configuration change.

The novel contribution is not any single number: it is that **the game is solved,
so the gap to optimal is measurable exactly** for every regime, in absolute
win-probability points.

## Shared infrastructure

Each stage reuses the previous one:

- Stage 0 needs an agent-versus-agent harness where one side is optimal and the
  other is arbitrary. Stage 1 needs the same harness with a heuristic in place of
  the partial map, so building stage 0 builds most of stage 1.
- Stage 1's regret metric is the evaluation for stages 2 and 3.
- `royalur_analysis simulate` already shards Monte Carlo games across array tasks
  with exact aggregation (binomial counts sum), at roughly 65,000 games/s per
  8-core task on Masters and ~3x that on the smaller maps.

## Scope

Stages 0 and 1 reuse existing infrastructure and belong with the current paper as
appendix material. Stages 2 and 3 are a separate project — folding them into a
paper whose claim is "we solved these rule sets exactly" would unbalance it.
