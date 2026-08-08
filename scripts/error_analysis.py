#!/usr/bin/env python3
"""Where does a compressed model's residual error live? -- Q4 of the stage-2 plan.

Regret already weights an error by its consequence, but it does not say whether
a model is *safe-but-dull* or *occasionally catastrophic*. Two models with equal
mean regret differ in kind if one loses a little everywhere and the other throws
away a won position now and then.

Three breakdowns, all exact because the true values are known:

  * by **sibling gap** -- how much separated the best move from the second best.
    Errors where the gap is tiny are nearly free; errors on wide gaps are the
    ones that lose games.
  * by **game stage**, proxied by total pieces scored, which says whether the
    model is weak in the opening, midgame or endgame.
  * by **position value** -- is the model worse when winning or when losing?

Usage:
    error_analysis.py <model.pt> <ruleset> <tensors.bin> <eval_successors.csv>
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import nn_sweep as S  # noqa: E402
from ur_tensors import Successors  # noqa: E402


def bucket_report(name, keys, edges, gap, chose_wrong):
    print(f"\n  by {name}:")
    print(f"    {'range':>18} {'decisions':>10} {'wrong %':>9} {'mean regret':>12}")
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (keys >= low) & (keys < high)
        if mask.sum() == 0:
            continue
        print(f"    {f'[{low:g}, {high:g})':>18} {int(mask.sum()):>10} "
              f"{100 * chose_wrong[mask].mean():>8.1f}% {gap[mask].mean():>12.4f}")


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    model_path, ruleset, _tensors, eval_path = sys.argv[1:5]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluation = Successors(eval_path, ruleset)
    unpack = S.Unpacker(ruleset, device)
    state = torch.load(model_path, map_location=device)
    width = state["stack.0.weight"].shape[0]
    depth = sum(1 for k in state if k.endswith(".weight")) - 1
    model = S.build("mlp", width, depth, unpack, 1, 0).to(device)
    model.load_state_dict(state)

    score = S.score_successors(model, unpack, evaluation.packed, device)
    report = evaluation.regret(score)
    print(f"{model_path}: regret {report['regret']:.4f}, agreement {report['agreement']:.4f}")

    # Per-decision outcome, recomputed here so the breakdowns line up with it.
    reflected = np.where(evaluation.passed > 0, 100.0 - score, score)
    reflected = np.where(evaluation.terminal, 100.0, reflected)
    offsets, counts = evaluation.offsets, evaluation.counts
    group_max = np.repeat(np.maximum.reduceat(reflected, offsets), counts)
    hits = np.flatnonzero(reflected >= group_max)
    who = np.searchsorted(offsets, hits, side="right") - 1
    _, first = np.unique(who, return_index=True)
    chosen = evaluation.value[hits[first]]
    gap = evaluation.best - chosen
    chose_wrong = gap > 1e-9

    # How far apart the best and second-best moves were: the stakes of the call.
    sorted_values = [np.sort(evaluation.value[o:o + c])[::-1] for o, c in zip(offsets, counts)]
    margin = np.array([v[0] - v[1] for v in sorted_values])
    scored = ((evaluation.parent >> np.uint64(54)) & np.uint64(15)).astype(np.int64) + \
             ((evaluation.parent >> np.uint64(58)) & np.uint64(15)).astype(np.int64)

    print(f"\n  overall: {100 * chose_wrong.mean():.1f}% of decisions wrong, "
          f"mean regret {gap.mean():.4f}, p99 {np.percentile(gap, 99):.4f}, "
          f"max {gap.max():.4f}")
    bucket_report("best-vs-second gap", margin, [0, 0.1, 0.5, 2, 5, 20, 101], gap, chose_wrong)
    bucket_report("pieces scored (game stage)", scored,
                  [0, 1, 3, 5, 7, 9, 15], gap, chose_wrong)
    bucket_report("position value", evaluation.best,
                  [0, 20, 40, 60, 80, 101], gap, chose_wrong)

    share = gap[margin >= 2].sum() / max(gap.sum(), 1e-12)
    print(f"\n  {100 * share:.1f}% of all regret comes from decisions where the best "
          f"move was ahead by 2+ points")


if __name__ == "__main__":
    main()
