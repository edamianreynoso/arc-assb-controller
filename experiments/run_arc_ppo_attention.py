"""Backward-compatible launcher.
Use `experiments/deep_rl/ppo/run_arc_ppo_attention.py` as the canonical path.
"""

from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parent / "deep_rl" / "ppo" / "run_arc_ppo_attention.py"

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
