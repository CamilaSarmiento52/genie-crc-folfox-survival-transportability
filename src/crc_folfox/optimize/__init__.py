"""Optimization workflow: technical correction and exploratory optimization.

This package is a post hoc addition to a completed study. It never writes to
`analysis/out/current/model_training/` or `analysis/out/current/evaluation/`; every artifact it
produces lives under `analysis/out/current/the optimization artifact directory`. See
`correction-spec.md` and `optimization-spec.md`.
"""

from __future__ import annotations
