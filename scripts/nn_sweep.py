#!/usr/bin/env python3
"""Stage 2: the capacity sweep. Fit models of increasing size to the exact table.

One point on the rate-distortion curve per model size: parameters (equivalently
bits, at 32 per fp32 weight) against policy regret, with value error alongside
because stage 1 showed the two disagree.

**No holdout by default.** The task is to reproduce *this* table in fewer bits,
and every state in it is part of the object being compressed. Withholding a
slice would measure interpolation to unseen positions -- a real question, but a
different one, and answering it would mean deliberately compressing 95% of the
object and reporting error on the other 5%. Pass `--holdout 0.05` to run that
control; it is worth doing only at the top of the sweep, where the parameter
count starts to approach the state count and "compression" could quietly become
memorisation.

Three objectives, and the differences between them are the point:

  * **value** -- BCE against the exact win probability. One forward pass per
    candidate successor at play time.
  * **ordering** -- listwise softmax over a position's successors. Still a
    per-successor scorer, but trained on the ordering rather than the level.
  * **policy** -- a head over the mover's own path indices, from the position
    and the roll. One forward pass per *position*, and it never has to resolve
    the sibling value gap at all, only its sign. Stage 1 found gaps below 1e-3
    points deciding moves; a value net must represent those, a policy head must
    not.

Two architectures:

  * **mlp** -- per-tile one-hot into a dense stack. The generic approximator,
    and the honest reference.
  * **transformer** -- one token per path position for each player, with
    self-attention. Capture, blocking and exposure are all *relations between
    pieces*, which a dense net must learn from scratch and attention gets as an
    inductive bias. Positional embeddings restore the ordinal meaning of "how
    far along the path" that a per-tile one-hot throws away.

Usage:
    nn_sweep.py <ruleset> <tensors.bin> <eval_successors.csv> [options]
      --arch mlp|transformer        (default mlp)
      --objective value|ordering|policy
      --train-successors <csv>      required for ordering and policy
      --sizes 64x2,256x2,1024x3     width x depth, comma separated
      --steps N                     (default 40000)
      --batch N                     (default 65536)
      --holdout FRACTION            (default 0: train on everything)
      --out results.jsonl
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ur_tensors import (PATHS, PIECES, Successors, load_decisions,  # noqa: E402
                        load_tensors, usable_tiles)
from ur_features import StateFeatures  # noqa: E402

MAX_ROLL = 8


def argument(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


class Unpacker:
    """Packed words -> model inputs, on the GPU.

    Kept as a bit-twiddle rather than a stored float matrix because the
    expansion is 20x: Finkel is 1.1 GB packed against 38 GB one-hot. Unpacking
    per batch costs a few hundred microseconds and buys the ability to hold the
    entire table in device memory.
    """

    def __init__(self, ruleset: str, device: torch.device, features: bool = False):
        self.ruleset = ruleset
        self.device = device
        self.tiles = torch.tensor(usable_tiles(ruleset), device=device)
        self.pieces = PIECES[ruleset]
        light, dark = PATHS[ruleset]
        self.path_length = len(light)
        self.paths = torch.tensor(list(light) + list(dark), device=device)
        self.width = 3 * len(self.tiles) + 2 * (self.pieces + 1)
        # Stage 1's 14 state features, recomputed per batch rather than stored.
        # Scaled to roughly unit range so they do not dominate the one-hot block.
        self.features = StateFeatures(ruleset, device) if features else None
        if self.features is not None:
            self.width += 14
            self.scale = torch.tensor(
                [self.path_length * self.pieces, self.path_length * self.pieces,
                 self.pieces, self.pieces, self.pieces, self.pieces,
                 self.pieces, self.pieces, 1.0, 1.0, 1.0, 1.0,
                 self.path_length, self.path_length],
                device=device, dtype=torch.float32)

    def hands(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mover = ((packed >> 48) & 7).clamp(0, self.pieces)
        other = ((packed >> 51) & 7).clamp(0, self.pieces)
        return mover, other

    def check(self, packed: torch.Tensor) -> None:
        """Reject occupancy code 3, which the packing never produces.

        Worth an explicit check because it fails silently rather than loudly:
        in the one-hot it writes into 3*slot+3, which is the *next* tile's
        "empty" position, so a format drift would corrupt inputs instead of
        raising. Run once at startup, not per batch.
        """
        codes = (packed.view(-1, 1) >> (2 * self.tiles).view(1, -1)) & 3
        assert int(codes.max()) < 3, "occupancy code 3 present; packing format has drifted"

    def dense(self, packed: torch.Tensor) -> torch.Tensor:
        """One-hot over tiles and hands, for the dense architecture."""
        rows = packed.shape[0]
        codes = (packed.view(-1, 1) >> (2 * self.tiles).view(1, -1)) & 3
        out = torch.zeros(rows, self.width, device=self.device)
        slots = torch.arange(len(self.tiles), device=self.device).view(1, -1)
        out.scatter_(1, 3 * slots + codes, 1.0)
        base = 3 * len(self.tiles)
        hand_width = self.pieces + 1
        mover, other = self.hands(packed)
        out.scatter_(1, base + mover.view(-1, 1), 1.0)
        out.scatter_(1, base + hand_width + other.view(-1, 1), 1.0)
        if self.features is not None:
            out[:, -14:] = self.features(packed) / self.scale
        return out

    def tokens(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Occupancy along both players' paths, plus the two hand counts.

        Path indexing is what makes this worth doing: "position 5 of my path"
        means the same thing for both colours, so a positional embedding carries
        distance-to-home rather than an arbitrary board label.
        """
        occupancy = (packed.view(-1, 1) >> (2 * self.paths).view(1, -1)) & 3
        mover, other = self.hands(packed)
        return occupancy, torch.stack([mover, other], dim=1)


class Dense(torch.nn.Module):
    def __init__(self, width: int, depth: int, inputs: int, outputs: int):
        super().__init__()
        layers: list[torch.nn.Module] = []
        previous = inputs
        for _ in range(depth):
            layers += [torch.nn.Linear(previous, width), torch.nn.ReLU()]
            previous = width
        layers.append(torch.nn.Linear(previous, outputs))
        self.stack = torch.nn.Sequential(*layers)

    def forward(self, dense, tokens, hands, extra=None, summary=None):
        # `summary` is already inside `dense` for this architecture.
        if extra is not None:
            dense = torch.cat([dense, extra], dim=1)
        return self.stack(dense)


class PieceTransformer(torch.nn.Module):
    """Self-attention over path positions.

    One token per path position per player, so a token is "the square five steps
    along my route, occupied by whom". Attention then relates tokens directly,
    which is the shape of every tactical fact in this game: a capture is a
    relation between one of my pieces and one of theirs, a block is a relation
    between a piece and a route. A dense net has to synthesise those from
    independent per-tile weights.
    """

    def __init__(self, width: int, depth: int, path_length: int, pieces: int,
                 outputs: int, extra: int = 0):
        super().__init__()
        self.path_length = path_length
        tokens = 2 * path_length + 2
        self.occupancy = torch.nn.Embedding(3, width)
        self.position = torch.nn.Embedding(tokens, width)
        self.hand = torch.nn.Embedding(pieces + 1, width)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=width, nhead=max(1, width // 32), dim_feedforward=2 * width,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = torch.nn.TransformerEncoder(layer, depth)
        self.head = torch.nn.Linear(width + extra, outputs)

    def forward(self, dense, tokens, hands, extra=None, summary=None):
        board = self.occupancy(tokens)
        hand = self.hand(hands)
        sequence = torch.cat([board, hand], dim=1)
        sequence = sequence + self.position.weight.unsqueeze(0)
        pooled = self.encoder(sequence).mean(dim=1)
        for block in (extra, summary):
            if block is not None:
                pooled = torch.cat([pooled, block], dim=1)
        return self.head(pooled)


class PointerHead(torch.nn.Module):
    """A trunk that encodes the position once, scoring each candidate move.

    This exists because the plain policy head and the value model each give up
    something the other keeps. A value model scores successors, so it gets a
    free ply of search -- it sees what each move leads to, using the known
    dynamics -- but pays one forward pass per candidate. A policy head runs the
    trunk once but sees only the parent and the roll, so it has to internalise
    those dynamics, and it has nowhere to condition logit k on move k's own
    features.

    Here the trunk runs once and each candidate is scored by a small shared
    function of (position embedding, that move's features). One trunk pass, and
    move features are usable -- which stage 1 found were worth roughly as much
    as the state features on their own.

    Move features are supplied per class; illegal classes are masked out by the
    caller, so their scores never matter.
    """

    def __init__(self, trunk: torch.nn.Module, width: int, classes: int,
                 move_features: int):
        super().__init__()
        self.trunk = trunk
        self.classes = classes
        # A learned embedding per source class carries "which move is this",
        # the way a positional embedding carries "which square is this".
        self.move = torch.nn.Embedding(classes, width)
        self.project = torch.nn.Linear(move_features, width) if move_features else None
        self.score = torch.nn.Sequential(
            torch.nn.Linear(2 * width, width), torch.nn.ReLU(),
            torch.nn.Linear(width, 1))

    def forward(self, dense, tokens, hands, extra=None, summary=None, moves=None):
        context = self.trunk(dense, tokens, hands, extra=extra, summary=summary)
        rows = context.shape[0]
        candidates = self.move.weight.unsqueeze(0).expand(rows, -1, -1)
        if self.project is not None and moves is not None:
            candidates = candidates + self.project(moves)
        context = context.unsqueeze(1).expand(-1, self.classes, -1)
        return self.score(torch.cat([context, candidates], dim=2)).squeeze(2)


def build(arch: str, width: int, depth: int, unpack: Unpacker, outputs: int, extra: int):
    if arch == "pointer":
        trunk = Dense(width, depth, unpack.width + extra, width)
        return PointerHead(trunk, width, outputs, 0)
    if arch == "pointer-transformer":
        summary = 14 if unpack.features is not None else 0
        trunk = PieceTransformer(width, depth, unpack.path_length, unpack.pieces,
                                 width, extra + summary)
        return PointerHead(trunk, width, outputs, 0)
    if arch == "mlp":
        return Dense(width, depth, unpack.width + extra, outputs)
    if arch == "transformer":
        # The engineered block is a summary of the whole position, so it joins
        # after pooling rather than as another token.
        summary = 14 if unpack.features is not None else 0
        return PieceTransformer(width, depth, unpack.path_length, unpack.pieces,
                                outputs, extra + summary)
    raise SystemExit(f"unknown architecture: {arch}")


def run(model, unpack, packed, extra=None):
    """Feed a batch through whichever architecture is in use.

    The dense stack reads the engineered features straight out of its input
    vector; the transformer never sees that vector, so the same block is handed
    to it separately as a pooled summary. Passing it to both would count it
    twice and mismatch the input width.
    """
    dense = unpack.dense(packed)
    summary = dense[:, -14:] if unpack.features is not None else None
    return model(dense, *unpack.tokens(packed), extra=extra, summary=summary)


@torch.no_grad()
def score_successors(model, unpack, packed_np, device, batch=1 << 19) -> np.ndarray:
    model.eval()
    words = torch.from_numpy(packed_np.astype(np.int64)).to(device)
    out = torch.empty(len(words), device=device)
    for start in range(0, len(words), batch):
        chunk = words[start:start + batch]
        out[start:start + batch] = torch.sigmoid(run(model, unpack, chunk).squeeze(1)) * 100.0
    model.train()
    return out.cpu().numpy().astype(np.float64)


@torch.no_grad()
def value_error(model, unpack, packed, values, device, sample=1 << 21):
    index = torch.randint(0, len(packed), (min(sample, len(packed)),), device=device)
    predicted = torch.sigmoid(run(model, unpack, packed[index]).squeeze(1)) * 100.0
    error = (predicted - values[index]).abs()
    return float(error.mean()), float(error.max())


def roll_onehot(roll: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(roll.clamp(0, MAX_ROLL - 1), MAX_ROLL).float()


@torch.no_grad()
def policy_regret(model, unpack, evaluation, device) -> dict:
    """Regret for a head that picks a source directly from position and roll."""
    model.eval()
    mask, best_source, row_of = evaluation.policy_targets()
    parent = torch.from_numpy(evaluation.parent.astype(np.int64)).to(device)
    roll = torch.from_numpy(evaluation.roll).to(device)
    logits = torch.empty(len(parent), mask.shape[1], device=device)
    for start in range(0, len(parent), 1 << 17):
        chunk = slice(start, start + (1 << 17))
        logits[chunk] = run(model, unpack, parent[chunk], extra=roll_onehot(roll[chunk]))
    model.train()
    legal = torch.from_numpy(mask).to(device)
    chosen = logits.masked_fill(~legal, -1e9).argmax(dim=1).cpu().numpy()
    picked = row_of[np.arange(len(chosen)), chosen]
    assert (picked >= 0).all(), "policy chose a source with no successor row"
    gap = evaluation.best - evaluation.value[picked]
    return {"regret": float(np.mean(gap)), "p95": float(np.percentile(gap, 95)),
            "max": float(np.max(gap)), "agreement": float(np.mean(gap < 1e-9))}


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    ruleset, tensors_path, eval_path = sys.argv[1], sys.argv[2], sys.argv[3]
    arch = argument("--arch", "mlp")
    objective = argument("--objective", "value")
    sizes = argument("--sizes", "32x2,64x2,128x2,256x2,512x3,1024x3,2048x3")
    steps = int(argument("--steps", 40000))
    batch = int(argument("--batch", 1 << 16))
    holdout_fraction = float(argument("--holdout", 0.0))
    learning_rate = float(argument("--lr", 2e-3))
    loss_kind = argument("--loss", "bce")
    tag = argument("--tag", "")
    # Run-to-run spread is ~25% at a few thousand parameters and ~5% at half a
    # million: small models are far more sensitive to initialisation. Without a
    # controlled seed, two identical configurations differ by more than most of
    # the effects being measured, so every comparison needs repeats.
    seed = int(argument("--seed", 0))
    out_path = argument("--out", f"nn_{ruleset}_{arch}_{objective}.jsonl")
    train_successors = argument("--train-successors")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} ruleset={ruleset} arch={arch} objective={objective} "
          f"features={argument('--features', 'none')}", flush=True)

    use_features = argument("--features", "none") != "none"
    unpack = Unpacker(ruleset, device, features=use_features)
    evaluation = Successors(eval_path, ruleset)
    unpack.check(torch.from_numpy(evaluation.packed[:4096].astype(np.int64)).to(device))
    print(f"eval set: {len(evaluation)} positions, {len(evaluation.packed)} successors")

    # A policy head never looks at values, and the Masters table is 6 GB of
    # device memory that would otherwise sit idle beside a 14 GB decision set.
    needs_table = objective != "policy"
    if needs_table:
        packed_np, values_np = load_tensors(tensors_path)
        packed = torch.from_numpy(packed_np.astype(np.int64)).to(device)
        values = torch.from_numpy(values_np.astype(np.float32)).to(device)
        state_count = len(packed_np)
    else:
        packed = values = None
        state_count = len(np.memmap(tensors_path, dtype=np.uint8, mode="r")) // 12
        train = holdout = None
        print(f"table: {state_count} states (not loaded; policy head needs no values)",
              flush=True)

    if not needs_table:
        pass
    elif holdout_fraction > 0:
        generator = torch.Generator(device="cpu").manual_seed(20250807)
        shuffled = torch.randperm(len(packed), generator=generator).to(device)
        cut = int(len(packed) * holdout_fraction)
        holdout, train = shuffled[:cut], shuffled[cut:]
        print(f"table: {len(packed)} states; train {len(train)}, holdout {len(holdout)}")
    else:
        # Every state is part of the object being compressed.
        train = holdout = torch.arange(len(packed), device=device)
        print(f"table: {len(packed)} states, all used for fitting", flush=True)

    groups = None
    if objective in ("policy", "ordering"):
        # The whole decision space, not a sample of it. A model trained on a
        # sample compresses that sample; with on-policy sampling the evaluation
        # positions are a subset of it, and the measurement becomes circular.
        decisions_path = argument("--decisions")
        if not decisions_path:
            raise SystemExit(f"--decisions is required for the {objective} objective")
        max_decisions = argument("--max-decisions")
        (d_packed, d_roll, d_mask, d_best, d_succ, d_val, d_count,
         d_passed) = load_decisions(
            decisions_path, int(max_decisions) if max_decisions else None)
        if objective == "policy":
            # The mask stays packed into an int32 and is expanded per batch.
            groups = {
                "parent": torch.from_numpy(d_packed.astype(np.int64)).to(device),
                "roll": torch.from_numpy(d_roll.astype(np.int8)).to(device),
                "bits": torch.from_numpy(d_mask.astype(np.int32)).to(device),
                "best": torch.from_numpy(d_best.astype(np.int8)).to(device),
            }
        else:
            # Per-successor scoring over the same decisions the value objective
            # sees, so the two differ only in what they are asked to predict.
            slots = np.arange(d_succ.shape[1])[None, :]
            alive = slots < d_count[:, None]
            passed = ((d_passed[:, None] >> slots) & 1).astype(bool)
            groups = {
                "succ": torch.from_numpy(d_succ.astype(np.int64)).to(device),
                "value": torch.from_numpy(d_val.astype(np.float32)).to(device),
                "alive": torch.from_numpy(alive).to(device),
                "passed": torch.from_numpy(passed).to(device),
                "best": torch.from_numpy(
                    np.argmax(np.where(alive, d_val, -1e9), axis=1).astype(np.int64)).to(device),
            }
        print(f"{objective} training set: {len(d_packed)} decisions (the full space), "
              f"{sum(v.element_size() * v.numel() for v in groups.values()) / 1e9:.1f} GB",
              flush=True)

    classes = unpack.path_length + 1
    outputs = classes if objective == "policy" else 1
    extra = MAX_ROLL if objective == "policy" else 0

    handle = open(out_path, "a")
    for spec in sizes.split(","):
        width, depth = (int(part) for part in spec.lower().split("x"))
        torch.manual_seed(seed)
        model = build(arch, width, depth, unpack, outputs, extra).to(device)
        parameters = sum(p.numel() for p in model.parameters())
        optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=learning_rate, total_steps=steps)
        started = time.time()

        for step in range(steps):
            if objective == "value":
                index = train[torch.randint(0, len(train), (batch,), device=device)]
                logits = run(model, unpack, packed[index]).squeeze(1)
                target = values[index] / 100.0
                if loss_kind == "bce":
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                elif loss_kind == "mse":
                    loss = torch.nn.functional.mse_loss(torch.sigmoid(logits), target)
                else:
                    raise SystemExit(f"unknown loss: {loss_kind}")
            elif objective == "ordering":
                rows = torch.randint(0, groups["succ"].shape[0], (batch // 8,), device=device)
                succ = groups["succ"][rows]
                alive = groups["alive"][rows]
                flat = succ.reshape(-1)
                # The model returns the value to whoever moves in the
                # successor; the targets are in the chooser's frame. Reflect
                # about 50 exactly where the move handed the turn over.
                score = torch.sigmoid(run(model, unpack, flat).squeeze(1)) * 100.0
                score = score.view(succ.shape)
                score = torch.where(groups["passed"][rows], 100.0 - score, score)
                score = score.masked_fill(~alive, -1e9)
                loss = torch.nn.functional.cross_entropy(score, groups["best"][rows])
            else:
                rows = torch.randint(0, len(groups["parent"]), (batch // 4,), device=device)
                logits = run(model, unpack, groups["parent"][rows],
                             extra=roll_onehot(groups["roll"][rows].long()))
                legal = ((groups["bits"][rows].view(-1, 1).long()
                          >> torch.arange(classes, device=device).view(1, -1)) & 1).bool()
                logits = logits.masked_fill(~legal, -1e9)
                loss = torch.nn.functional.cross_entropy(logits, groups["best"][rows].long())

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            if step % 5000 == 0:
                print(f"  {spec} step {step}/{steps} loss {loss.item():.5f}", flush=True)

        if objective == "policy":
            report = policy_regret(model, unpack, evaluation, device)
            fit_mae = fit_max = hold_mae = hold_max = float("nan")
        else:
            report = evaluation.regret(score_successors(model, unpack, evaluation.packed, device))
            fit_mae, fit_max = value_error(model, unpack, packed[train], values[train], device)
            hold_mae, hold_max = value_error(model, unpack, packed[holdout], values[holdout], device)
        record = {
            "ruleset": ruleset, "arch": arch, "objective": objective, "size": spec,
            "width": width, "depth": depth, "parameters": parameters,
            "bits": parameters * 32, "bits_per_state": parameters * 32 / state_count,
            "steps": steps, "batch": batch, "holdout_fraction": holdout_fraction,
            "lr": learning_rate, "loss": loss_kind, "tag": tag, "seed": seed,
            "features": argument("--features", "none"),
            "seconds": round(time.time() - started, 1),
            "fit_mae": fit_mae, "fit_max_error": fit_max,
            "holdout_mae": hold_mae, "holdout_max_error": hold_max,
            **report,
        }
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        print(f"{spec}: {parameters} params  regret {report['regret']:.4f}  "
              f"agreement {report['agreement']:.4f}  MAE {fit_mae:.4f}  "
              f"({record['seconds']}s)", flush=True)
        torch.save(model.state_dict(), f"model_{ruleset}_{arch}_{objective}_{spec}.pt")

    handle.close()


if __name__ == "__main__":
    main()
