"""Backward-compatible launcher.
Use `experiments/deep_rl/dqn/run_l6b_dqn_suite.py` as the canonical path.
"""

from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parent / "deep_rl" / "dqn" / "run_l6b_dqn_suite.py"

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
