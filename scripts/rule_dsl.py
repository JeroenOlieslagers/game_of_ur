#!/usr/bin/env python3
"""A tiny language for stating Royal Game of Ur move rules, and a scorer.

A rule is a boolean expression over the features of a candidate move. Rules are
parsed into an expression tree and evaluated vectorised over the whole dataset,
so a rule can be graded against tens of thousands of positions in milliseconds.
Nothing is `eval`-ed: generated text is parsed by a recursive-descent parser and
anything outside the grammar is rejected, so proposals from a language model are
safe to run.

Grammar
-------
    expr    := or_expr
    or_expr := and_expr ("or" and_expr)*
    and_expr:= not_expr ("and" not_expr)*
    not_expr:= "not" not_expr | comparison
    comparison := sum (("<"|"<="|">"|">="|"=="|"!=") sum)?
    sum     := product (("+"|"-") product)*
    product := atom (("*"|"/") atom)*
    atom    := number | feature | "(" expr ")" | "-" atom

A bare arithmetic expression is truthy when positive, so `captures` alone is a
valid rule.

Why a rule and not a weight: rules are read as strategy ("prefer capturing"),
and a list of them is a policy a person can follow. Because the map is solved,
each one gets an exact score in win-probability points.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass

import numpy as np

TOKEN = re.compile(r"\s*(<=|>=|==|!=|<|>|\(|\)|\+|-|\*|/|[A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*)")


class RuleError(ValueError):
    """A rule that is not valid in the grammar, or names an unknown feature."""


@dataclass
class Parser:
    tokens: list
    position: int = 0

    def peek(self):
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self, expected=None):
        token = self.peek()
        if token is None:
            raise RuleError("unexpected end of rule")
        if expected is not None and token != expected:
            raise RuleError(f"expected {expected!r}, found {token!r}")
        self.position += 1
        return token

    def parse(self, columns):
        value = self.or_expr(columns)
        if self.position != len(self.tokens):
            raise RuleError(f"trailing input at {self.tokens[self.position]!r}")
        return value

    def or_expr(self, columns):
        value = self.and_expr(columns)
        while self.peek() == "or":
            self.take()
            value = value | self.and_expr(columns)
        return value

    def and_expr(self, columns):
        value = self.not_expr(columns)
        while self.peek() == "and":
            self.take()
            value = value & self.not_expr(columns)
        return value

    def not_expr(self, columns):
        if self.peek() == "not":
            self.take()
            return ~self.not_expr(columns)
        return self.comparison(columns)

    def comparison(self, columns):
        left = self.sum(columns)
        operator = self.peek()
        if operator in ("<", "<=", ">", ">=", "==", "!="):
            self.take()
            right = self.sum(columns)
            return {
                "<": left < right, "<=": left <= right,
                ">": left > right, ">=": left >= right,
                "==": np.isclose(left, right), "!=": ~np.isclose(left, right),
            }[operator]
        # A bare expression counts as true where it is positive.
        return left > 0 if left.dtype != bool else left

    def sum(self, columns):
        value = self.product(columns)
        while self.peek() in ("+", "-"):
            operator = self.take()
            right = self.product(columns)
            value = value + right if operator == "+" else value - right
        return value

    def product(self, columns):
        value = self.atom(columns)
        while self.peek() in ("*", "/"):
            operator = self.take()
            right = self.atom(columns)
            value = value * right if operator == "*" else value / np.where(right == 0, np.nan, right)
        return value

    def atom(self, columns):
        token = self.take()
        if token == "(":
            value = self.or_expr(columns)
            self.take(")")
            return value
        if token == "-":
            return -self.atom(columns)
        if re.fullmatch(r"\d+\.?\d*", token):
            return np.full(columns["__rows__"], float(token))
        if token in columns:
            return columns[token]
        raise RuleError(f"unknown feature {token!r}")


def evaluate(rule: str, columns) -> np.ndarray:
    """Parse and evaluate a rule, returning a boolean mask over candidate moves."""
    tokens = []
    position = 0
    while position < len(rule):
        match = TOKEN.match(rule, position)
        if not match:
            if rule[position:].strip() == "":
                break
            raise RuleError(f"cannot tokenise at {rule[position:][:20]!r}")
        tokens.append(match.group(1))
        position = match.end()
    if not tokens:
        raise RuleError("empty rule")
    result = Parser(tokens).parse(columns)
    if result.dtype != bool:
        result = result > 0
    return result


class Dataset:
    """Candidate moves grouped by position, with the exact value of each."""

    META = ("state", "move", "turn_passed", "value_mover", "occupancy")

    def __init__(self, path):
        rows = list(csv.DictReader(open(path)))
        self.names = [k for k in rows[0] if k not in self.META]
        self.columns = {
            name: np.array([float(r[name]) for r in rows]) for name in self.names
        }
        self.values = np.array([float(r["value_mover"]) for r in rows])
        state = np.array([int(r["state"]) for r in rows])
        self.offsets = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
        self.counts = np.diff(np.r_[self.offsets, len(state)])
        self.best = np.maximum.reduceat(self.values, self.offsets)
        self.columns["__rows__"] = len(rows)
        self.rows = len(rows)

    def regret(self, alive) -> float:
        """Mean regret of picking the first surviving move in each position."""
        index = np.arange(self.rows)
        pick = np.minimum.reduceat(np.where(alive, index, index.max() + 1), self.offsets)
        return float(np.mean(self.best - self.values[pick]))

    def apply_rule(self, alive, mask):
        """Narrow the candidates by a rule, but only where it leaves one alive."""
        candidate = alive & mask
        kept = np.repeat(
            np.add.reduceat(candidate.astype(np.int64), self.offsets), self.counts
        )
        return np.where(kept > 0, candidate, alive)

    def score_list(self, rules) -> float:
        alive = np.ones(self.rows, dtype=bool)
        for rule in rules:
            alive = self.apply_rule(alive, evaluate(rule, self.columns))
        return self.regret(alive)
