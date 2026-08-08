#!/usr/bin/env python3
"""Export a trained MLP to a flat text file the engine can read.

Regret is the cheap exact metric; win rate against the optimal agent is the
ground truth, and playing a game means choosing moves online, inside the
engine. Rather than link a tensor library into Rust, the weights are written
out and a matmul is hand-rolled there -- an MLP forward pass is a dozen lines,
and this keeps the engine dependency-free.

Format, all decimal, one number per line:

    n_layers
    in_0 out_0            # layer 0 shape
    W[0,0] W[0,1] ...     # row-major weights, out x in
    b[0] b[1] ...         # biases
    ... repeated per layer

The final layer has one output; the engine applies a sigmoid and scales to
[0, 100], matching `score_successors` in nn_sweep.py.

Usage:
    export_net.py <model.pt> <ruleset> <output.txt> [--features]
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ur_tensors import PIECES, usable_tiles  # noqa: E402


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    state = torch.load(sys.argv[1], map_location="cpu")
    ruleset, output = sys.argv[2], sys.argv[3]
    width = 3 * len(usable_tiles(ruleset)) + 2 * (PIECES[ruleset] + 1)
    if "--features" in sys.argv:
        width += 14

    layers = []
    index = 0
    while f"stack.{index}.weight" in state:
        layers.append((state[f"stack.{index}.weight"], state[f"stack.{index}.bias"]))
        index += 2          # Linear, ReLU, Linear, ReLU, ...
    if not layers:
        raise SystemExit("no Sequential 'stack' found; this exporter handles the dense model only")

    assert layers[0][0].shape[1] == width, (
        f"input width {layers[0][0].shape[1]} does not match {width} for {ruleset}; "
        "pass --features if the model was trained with the engineered block")
    assert layers[-1][0].shape[0] == 1, "expected a single-output value head"

    with open(output, "w") as handle:
        handle.write(f"{len(layers)}\n")
        for weight, bias in layers:
            out_dim, in_dim = weight.shape
            handle.write(f"{in_dim} {out_dim}\n")
            handle.write(" ".join(f"{v:.9g}" for v in weight.flatten().tolist()) + "\n")
            handle.write(" ".join(f"{v:.9g}" for v in bias.tolist()) + "\n")
    total = sum(w.numel() + b.numel() for w, b in layers)
    print(f"wrote {len(layers)} layers, {total} parameters to {output}")


if __name__ == "__main__":
    main()
