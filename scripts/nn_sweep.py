#!/usr/bin/env python3
"""Stage 2: the capacity sweep. Train MLPs of increasing size on the exact table.

One point on the rate-distortion curve per model size. The x axis is parameters
(equivalently bits, at 32 per fp32 weight); the y axis is policy regret against
the exact optimum, with value error reported alongside because stage 1 showed
they disagree.

Two objectives, which is the point of the sweep rather than a detail:

  * **value** -- binary cross-entropy against the exact win probability. The
    natural compression objective: reproduce the table.
  * **ordering** -- a listwise softmax over the candidate successors of a
    position, against the optimal move. The natural play objective.

Stage 1 found the value fit had the higher R^2 and played 25-34% worse at 15
parameters. That has to be a small-capacity effect -- as value error goes to
zero regret must follow -- so the interesting measurement is the capacity at
which the gap closes. Running both objectives at every size is what makes that
visible.

The whole dump is held in GPU memory (1.7 GB for Finkel, 6 GB for Masters
against an H200's 141 GB), so batching is pure device-side indexing and there is
no dataloader in the loop.

Usage:
    nn_sweep.py <ruleset> <tensors.bin> <eval_successors.csv> [options]
      --objective value|ordering    (default value)
      --train-successors <csv>      required for the ordering objective
      --sizes 64x2,256x2,1024x3     width x depth, comma separated
      --steps N                     optimiser steps per model (default 40000)
      --batch N                     (default 65536)
      --out results.jsonl
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ur_tensors import PIECES, Successors, load_tensors, usable_tiles  # noqa: E402


def argument(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


class Unpacker:
    """Packed words -> one-hot design, on the GPU.

    Kept as a bit-twiddle rather than a stored float matrix because the
    expansion is 20x: Finkel is 1.1 GB packed against 38 GB one-hot. Unpacking
    per batch costs a few hundred microseconds and buys the ability to hold the
    entire table on the device.
    """

    def __init__(self, ruleset: str, device: torch.device):
        self.tiles = torch.tensor(usable_tiles(ruleset), device=device)
        self.pieces = PIECES[ruleset]
        self.device = device
        self.width = 3 * len(self.tiles) + 2 * (self.pieces + 1)

    def __call__(self, packed: torch.Tensor) -> torch.Tensor:
        rows = packed.shape[0]
        shifts = (2 * self.tiles).view(1, -1)
        codes = (packed.view(-1, 1) >> shifts) & 3
        out = torch.zeros(rows, self.width, device=self.device)
        slots = torch.arange(len(self.tiles), device=self.device).view(1, -1)
        out.scatter_(1, 3 * slots + codes, 1.0)
        base = 3 * len(self.tiles)
        hand_width = self.pieces + 1
        mover = ((packed >> 48) & 7).clamp(0, self.pieces).view(-1, 1)
        other = ((packed >> 51) & 7).clamp(0, self.pieces).view(-1, 1)
        out.scatter_(1, base + mover, 1.0)
        out.scatter_(1, base + hand_width + other, 1.0)
        return out


def build(width: int, depth: int, inputs: int) -> torch.nn.Module:
    layers: list[torch.nn.Module] = []
    previous = inputs
    for _ in range(depth):
        layers += [torch.nn.Linear(previous, width), torch.nn.ReLU()]
        previous = width
    layers.append(torch.nn.Linear(previous, 1))
    return torch.nn.Sequential(*layers)


@torch.no_grad()
def score_successors(model, unpack, packed: np.ndarray, device, batch=1 << 20) -> np.ndarray:
    model.eval()
    words = torch.from_numpy(packed.astype(np.int64)).to(device)
    out = torch.empty(len(words), device=device)
    for start in range(0, len(words), batch):
        chunk = words[start:start + batch]
        out[start:start + batch] = torch.sigmoid(model(unpack(chunk)).squeeze(1)) * 100.0
    model.train()
    return out.cpu().numpy().astype(np.float64)


@torch.no_grad()
def value_error(model, unpack, packed, values, device, sample=1 << 21):
    """Mean and max absolute value error, in win-probability points."""
    index = torch.randint(0, len(packed), (min(sample, len(packed)),), device=device)
    predicted = torch.sigmoid(model(unpack(packed[index])).squeeze(1)) * 100.0
    error = (predicted - values[index]).abs()
    return float(error.mean()), float(error.max())


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    ruleset, tensors_path, eval_path = sys.argv[1], sys.argv[2], sys.argv[3]
    objective = argument("--objective", "value")
    sizes = argument("--sizes", "32x2,64x2,128x2,256x2,512x3,1024x3,2048x3")
    steps = int(argument("--steps", 40000))
    batch = int(argument("--batch", 1 << 16))
    out_path = argument("--out", f"nn_{ruleset}_{objective}.jsonl")
    train_successors = argument("--train-successors")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} ruleset={ruleset} objective={objective}", flush=True)

    unpack = Unpacker(ruleset, device)
    evaluation = Successors(eval_path, ruleset)
    print(f"eval set: {len(evaluation)} positions, {len(evaluation.packed)} successors")

    packed_np, values_np = load_tensors(tensors_path)
    print(f"table: {len(packed_np)} states", flush=True)
    packed = torch.from_numpy(packed_np.astype(np.int64)).to(device)
    values = torch.from_numpy(values_np.astype(np.float32)).to(device)

    # Hold out a twentieth of the table. At these capacities the gap is nil and
    # this is insurance rather than the point -- but at the top of the sweep a
    # model starts to have room to memorise, and that is exactly where a
    # rate-distortion claim would quietly become false.
    generator = torch.Generator(device="cpu").manual_seed(20250807)
    shuffled = torch.randperm(len(packed), generator=generator).to(device)
    holdout = shuffled[: len(packed) // 20]
    train = shuffled[len(packed) // 20:]
    print(f"train {len(train)}, holdout {len(holdout)}", flush=True)

    groups = None
    if objective == "ordering":
        if not train_successors:
            raise SystemExit("--train-successors is required for the ordering objective")
        training_set = Successors(train_successors, ruleset)
        # Pad ragged sibling groups into a rectangle so the listwise softmax is
        # one batched operation; padded slots are masked to -inf.
        widest = int(training_set.counts.max())
        rows = len(training_set)
        slot = np.full((rows, widest), -1, dtype=np.int64)
        for row, (offset, count) in enumerate(zip(training_set.offsets, training_set.counts)):
            slot[row, :count] = np.arange(offset, offset + count)
        groups = {
            "slot": torch.from_numpy(slot).to(device),
            "mask": torch.from_numpy(slot >= 0).to(device),
            "packed": torch.from_numpy(training_set.packed.astype(np.int64)).to(device),
            "passed": torch.from_numpy(training_set.passed.astype(np.float32)).to(device),
            "best": torch.from_numpy(
                np.argmax(np.where(slot >= 0, training_set.value[np.clip(slot, 0, None)], -1e9),
                          axis=1)).to(device),
        }
        print(f"ordering training set: {rows} positions, widest group {widest}", flush=True)

    handle = open(out_path, "a")
    for spec in sizes.split(","):
        width, depth = (int(part) for part in spec.lower().split("x"))
        model = build(width, depth, unpack.width).to(device)
        parameters = sum(p.numel() for p in model.parameters())
        optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=2e-3, total_steps=steps)
        started = time.time()

        for step in range(steps):
            if objective == "value":
                index = train[torch.randint(0, len(train), (batch,), device=device)]
                # BCE against the exact probability as a soft target: better
                # calibrated near 0 and 1 than MSE, which is where the decisive
                # positions live.
                logits = model(unpack(packed[index])).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, values[index] / 100.0)
            else:
                rows = torch.randint(0, groups["slot"].shape[0], (batch // 16,), device=device)
                slot = groups["slot"][rows]
                mask = groups["mask"][rows]
                flat = slot.clamp(min=0).reshape(-1)
                logit = model(unpack(groups["packed"][flat])).squeeze(1)
                value = torch.sigmoid(logit) * 100.0
                # Score from the chooser's perspective: reflect about 50 exactly
                # where the move hands the turn over.
                passed = groups["passed"][flat]
                value = torch.where(passed > 0, 100.0 - value, value)
                value = value.view(slot.shape)
                value = value.masked_fill(~mask, -1e9)
                loss = torch.nn.functional.cross_entropy(value, groups["best"][rows])

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            if step % 5000 == 0:
                print(f"  {spec} step {step}/{steps} loss {loss.item():.5f}", flush=True)

        scores = score_successors(model, unpack, evaluation.packed, device)
        report = evaluation.regret(scores)
        train_mae, train_max = value_error(model, unpack, packed[train], values[train], device)
        hold_mae, hold_max = value_error(model, unpack, packed[holdout], values[holdout], device)
        record = {
            "ruleset": ruleset, "objective": objective, "size": spec,
            "width": width, "depth": depth, "parameters": parameters,
            "bits": parameters * 32, "bits_per_state": parameters * 32 / len(packed_np),
            "steps": steps, "batch": batch, "seconds": round(time.time() - started, 1),
            "train_mae": train_mae, "train_max_error": train_max,
            "holdout_mae": hold_mae, "holdout_max_error": hold_max,
            **report,
        }
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        print(f"{spec}: {parameters} params  regret {report['regret']:.4f}  "
              f"agreement {report['agreement']:.4f}  holdout MAE {hold_mae:.4f}  "
              f"({record['seconds']}s)", flush=True)
        torch.save(model.state_dict(), f"model_{ruleset}_{objective}_{spec}.pt")

    handle.close()


if __name__ == "__main__":
    main()
