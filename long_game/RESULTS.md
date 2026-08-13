# Longest-game Finkel: what was computed, and what it means

## The result

Under a policy that both players cooperate to maximise game duration, a Finkel
game lasts

> **412,543 +/- 252 legal moves** (1,000,000 self-play games)

against the few dozen moves a normal game takes. The distribution is wide:

| quantile | actions |
| --- | --- |
| minimum | 10,317 |
| 5th | 123,192 |
| 25th | 232,679 |
| median | 354,234 |
| 75th | 529,148 |
| 95th | 899,502 |
| 99th | 1,263,511 |
| maximum | 3,334,254 |

Mean dice rolls 440,121. `light_win_fraction` 0.4998, confirming the cooperative
objective is symmetric between the players.

## What this number is, and is not

It **is** a measured lower bound on the optimum, and a tight statistical one:
any policy's value lower-bounds the optimal value, so `V* >= 412,543` to within
the Monte Carlo standard error of 252.

It is **not** a certified optimum. The lookup table the policy came from has not
converged, and the honest position is that the exact value of the starting state
remains unknown -- only bounded below.

## Why the exact value is out of reach with this solver

The objective is **undiscounted**: each legal move earns +1 and nothing is
discounted, so the Bellman operator is not a contraction. Convergence is
governed instead by the probability of leaving a score layer, and cooperating
players specifically avoid scoring, so that probability is tiny. The measured
per-sweep contraction is

```
rho = 0.99998        (delta 4.86 at sweep 2,500 -> 0.107 at sweep 205,500)
```

Two consequences follow, and the second is what invalidated an earlier result.

**Value iteration is impractically slow.** Reaching a residual of 1e-8 on a
single mid-size layer needs about 800,000 sweeps, roughly seven hours, and the
map has 49 layers of which the largest is 30 million states.

**A small Bellman residual does not imply a small error.** The bound is
`|V - V*| <= residual / (1 - rho)`, and here `1/(1 - rho) ~ 50,000`. Measured
directly: at residual 3.88 the actual error was 290,442, so error is about
75,000 x residual -- the theoretical factor, near enough.

That second point produced a wrong answer that was reported before the
simulation caught it. A solve certified at "3.1e-5 relative residual" gave a
starting value of 121,993, while the simulation of its own policy achieved
412,543 -- a 1,150-sigma discrepancy. Both are valid lower bounds; the value
iterate was simply far from converged, and the residual test could not see it.

**Tightening the tolerance does not rescue it.** To bound the error at 1% of
412,543 needs a residual near 0.053. Layer (2,3) was falling 0.2% per block at
that stage, which is about 2,000 more blocks, or **12 days for one layer**.

## The useful asymmetry: good policy, bad values

The policy is far better than its values. Policy iteration is known to find
near-optimal policies long before the value estimates converge, and that is
exactly what happened: values that understate the optimum by a factor of 3.4
nonetheless induce a policy achieving 412,543 actions.

So the deliverable is a *policy* result rather than a *value* result, and the
simulation measures the policy directly. Whether that policy is optimal is
open.

## What would be needed for the exact value

1. **A real linear solve per layer.** Freeze the maximising action and solve
   `(I - P_pi) V = r_pi` with a Krylov method rather than Gauss-Seidel sweeps.
   This targets the slow recurrent modes directly, which sweeping cannot.
2. **An upper bound.** Any V with `T(V) <= V` satisfies `V >= V*`, so
   constructing one would bracket the answer instead of only bounding it below.
   Combined with the lower bound above, that would give a genuine interval.

## Files

| file | contents |
| --- | --- |
| `finkel_longest_tol1e-4.rgu` | duration values, all 137,892,016 states, NOT converged |
| `finkel_longest.policy` | direct state-by-roll action table, 689 MB |
| `finkel_longest.checkpoint` | per-layer values and residuals |
| `summary.json` | starting value, Monte Carlo mean, quantiles, timings |
| `length_histogram.csv` | `(actions, count, probability, cdf)` |
| `length_distribution.png` / `.svg` | binned mass and log survival curve |

The `.rgu` is retained deliberately: it is a valid lower bound and the correct
starting point for any refinement, since value iteration from below is monotone
and none of its progress is wasted.

## Reproducing

```bash
# Solve (does not converge to the exact optimum; see above)
royalur_analysis train-long-game models/finkel.rgu out/finkel_longest.rgu 1e-4 2000000
# Independent residual check and starting value
royalur_analysis verify-long-game out/finkel_longest.rgu 20000
# Self-play under the derived policy
royalur_analysis simulate-long-game out/finkel_longest.rgu out 1000000 <seed>
python3 long_game/plot_lengths.py out/length_histogram.csv out/summary.json out/length_distribution.png
```

A 10,000,000-game run was requested originally. At ~4x10^5 actions per game that
is 1.2x10^11 moves, roughly eleven hours, which exceeds the four-hour partition
limit and the simulation has no checkpointing -- so it would restart forever.
1,000,000 games gives a standard error of 0.06% of the mean, which is far finer
than the solve error, so the larger run would not change any conclusion here.
