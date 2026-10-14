from __future__ import annotations

"""Lightweight training entry that delegates to the canonical experiment runner.

This module keeps backwards compatibility for callers that expect
``python -m dsmr.train`` while centralizing training logic in
``dsmr.run_experiment``.
"""

from dsmr.run_experiment import main as run_experiment_main


if __name__ == "__main__":
    run_experiment_main()
