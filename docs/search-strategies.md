# Search strategies (side investigation)

Deliberately kept out of the stage 1 and stage 2 results. Those are about how
well a model plays; this is about how cheaply an answer can be computed. Nothing
here changes a reported regret -- that is the point, and it is asserted rather
than assumed.

## Why not MCTS

Two structural facts rule it out, and they are worth stating because "try MCTS"
is the reflex answer to any game-search question.

**The branching factor is tiny.** After a roll there are about 2.8 legal moves
on Finkel (mean `log2` of legal moves is 1.433 bits). MCTS earns its keep when
full-width enumeration is hopeless -- Go's ~250. At b ~ 3, full-width expectimax
already visits nearly everything MCTS would, without the bookkeeping.

**The dice distribution is known exactly.** Expectimax enumerates the four or
five roll outcomes and computes the exact expectation; MCTS samples them,
introducing variance where exactness was free. At equal node counts that is
strictly worse.

The precedent supports this: backgammon has the same shape -- dice, small
post-roll branching -- and its strongest engines use shallow full-width
expectimax with a strong evaluator. AlphaZero-style MCTS never displaced that.

AlphaZero-style self-play belongs in **stage 3**, where the question is a
comparison of training regimes against a known optimum, not a way to get a
stronger agent. Nothing learned can beat a solved table.

## What was built instead

Two exact techniques, neither of which approximates anything:

- **A transposition table.** Ur transposes heavily: moving piece A then B
  reaches the same position as B then A. Direct-mapped, always-replace, 2^22
  entries.
- **Star1 pruning** (Ballard 1983), the alpha-beta analogue for chance nodes. A
  chance node's value is `sum_r p_r v_r` with each `v_r` in [0, 100], so once
  some outcomes are known the rest are bounded and a child need only be searched
  within the window that could still move the expectation past alpha or beta.

## Results

Finkel, 2,000 on-policy positions, 15-weight fitted evaluator at the leaves.

| depth | plain | +table | +table+star1 | speedup | table hits |
| --- | --- | --- | --- | --- | --- |
| 4 | 7.3 s | 5.9 s | 5.1 s | 1.2x | 20% |
| 5 | 96.2 s | 40.0 s | 38.1 s | 2.4x | 54% |
| 6 | 1296.7 s | 235.7 s | 234.1 s | **5.5x** | 72% |

Regret is identical across all three at every depth (0.1308, 0.1189, 0.0970), so
this is pure compute saving.

**The table is the whole win; star1 is nearly redundant once it is present.**
Star1 adds 15% at depth 4, 5% at depth 5 and 0.7% at depth 6. The two compete
rather than compose: the more of the tree the table answers outright, the less
is left for pruning to cut. A measurement stopping at depth 4 would have
concluded the opposite about which technique mattered.

**The speedup compounds**, because transposition density rises with depth -- more
of the tree consists of positions reachable by several move orderings. At depth 6
nearly three quarters of probes hit.

Consequence: depth 8 becomes reachable. Depth 7 should cost roughly 400 s rather
than 4,305 s, putting depth 8 near 1.5 hours instead of about 16. "Where does
search saturate" changes from impractical to runnable, which matters because
regret was still falling 12-25% per ply at depth 7.

## Two bugs, and why the assertion existed

The benchmark requires all three configurations to choose the **same move at
every position**, aborting otherwise. It caught two bugs within seconds, both of
which would otherwise have surfaced as a faster search with quietly wrong moves.

**Star1 tested cutoffs against clamped bounds.** The true bound a child must
break can lie outside [0, 100], because it is divided by a probability as small
as 1/16. The *window handed to the child* must be a valid value range; the
*cutoff test* must use the unclamped bound. Conflating them made a single
winning roll look like a fail-high and returned 100 for a node whose value is an
expectation, not a maximum.

**The transposition table accepted entries from deeper searches.** The standard
rule is `entry.depth >= requested`, valid when a deeper search is a better
estimate of one quantity -- true when leaves are terminal positions. Here the
leaves are a *heuristic*, so depth-5 and depth-3 values are different functions,
and reusing the deeper one silently changes what a fixed-depth search returns.

Both are the same shape of error: two quantities sharing units and meaning
different things. So were the reflection bugs in stage 1 -- a value in the
mover's frame and a value in light's frame are both win percentages.

## Design note: the transposition key

The key is *absolute* -- light is always 1, dark always 2, plus a side-to-move
bit, 55 bits total -- rather than reusing `pack_position`. That canonical packing
maps mirrored positions to one key, so a table holding light-relative values
would need a reflection about 50 on every store and probe. Spending 55 bits to
never transform a stored value is a good trade in a codebase where that
transformation has caused several bugs. The cost is losing symmetry sharing in
the table; the benefit is a correctness argument short enough to check by eye.
