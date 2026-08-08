#!/usr/bin/env python3
"""Gradient boosting on the same axes as the networks -- Q3 of the stage-2 plan.

The question is whether model families trace one envelope. If boosted trees,
dense nets and attention all land on the same rate-distortion curve, that curve
is a property of Ur and "this rule set needs N bits" is a claim about the game.
If they separate, it is a claim about inductive bias.

Capacity is trees x depth. Parameters are counted as internal nodes plus leaves,
which is the honest comparison to a weight count: both are numbers that must be
stored to reproduce the model.

Trained on a uniform sample of the table rather than all of it -- boosting is
not minibatch-incremental and 41M rows will not fit -- so the sample size is
reported and swept, since a tree model that is sample-limited rather than
capacity-limited would put it on the wrong part of the curve.

Usage:
    xgb_sweep.py <ruleset> <tensors.bin> <eval_successors.csv> [--rows N] [--out f.jsonl]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ur_tensors import Successors, load_tensors, unpack  # noqa: E402


def argument(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    import xgboost as xgb

    ruleset, tensors_path, eval_path = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = int(argument("--rows", 4_000_000))
    out_path = argument("--out", f"xgb_{ruleset}.jsonl")

    packed, values = load_tensors(tensors_path)
    rng = np.random.default_rng(0)
    index = rng.choice(len(packed), size=min(rows, len(packed)), replace=False)
    design = unpack(packed[index], ruleset)
    target = values[index].astype(np.float32) / 100.0
    print(f"{len(design)} training rows of {len(packed)} states, {design.shape[1]} features",
          flush=True)

    evaluation = Successors(eval_path, ruleset)
    eval_design = evaluation.design()

    handle = open(out_path, "a")
    for trees, depth in ((50, 4), (200, 6), (600, 8), (2000, 10)):
        started = time.time()
        model = xgb.XGBRegressor(
            n_estimators=trees, max_depth=depth, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            objective="reg:logistic", n_jobs=-1)
        model.fit(design, target)
        score = model.predict(eval_design).astype(np.float64) * 100.0
        report = evaluation.regret(score)
        # Nodes, not trees: what has to be stored to reproduce the model.
        # Counted from the text dump rather than `trees_to_dataframe`, which
        # needs pandas for what is a line count.
        dump = model.get_booster().get_dump()
        parameters = sum(
            1 for tree in dump for line in tree.splitlines() if line.strip())
        record = {"ruleset": ruleset, "family": "xgboost", "trees": trees, "depth": depth,
                  "parameters": parameters, "bits": parameters * 32,
                  "rows": len(design), "seconds": round(time.time() - started, 1), **report}
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        print(f"{trees}x{depth}: {parameters} nodes  regret {report['regret']:.4f}  "
              f"agreement {report['agreement']:.4f}  ({record['seconds']}s)", flush=True)
    handle.close()


if __name__ == "__main__":
    main()
