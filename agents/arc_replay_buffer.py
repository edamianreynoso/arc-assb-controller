"""
ARC-gated replay buffer for Stable-Baselines3 off-policy algorithms.

Motivation:
- In tabular L6, ARC's "memory gating" blocks updates under internal overload.
- For Deep RL (e.g., DQN), practical analogs include:
  (1) gating experience storage when the ARC memory gate is closed
  (2) biasing replay sampling toward recent data during detected distribution shifts

This buffer reads ARC signals from `infos` (produced by `ARCGymWrapper`) and can
skip storing transitions when `arc_u_mem` is low (optional) and can bias replay
toward recent transitions when the agent is in "shift mode" (`arc_shift_active`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer


@dataclass
class ARCGatedReplayConfig:
    enable: bool = True

    # Storage gating (optional)
    skip_add_when_u_mem_low: bool = False
    u_mem_threshold: float = 0.2
    min_transitions_to_gate: int = 1000
    bypass_when_shift_active: bool = True

    # Shift-aware replay (optional)
    shift_sample_recent: bool = True
    shift_recent_window: int = 5000
    shift_recent_fraction: float = 1.0


class ARCGatedReplayBuffer(ReplayBuffer):
    """ReplayBuffer that can skip storing transitions when ARC gates memory."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        arc_config: Optional[ARCGatedReplayConfig] = None,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.arc_config = arc_config or ARCGatedReplayConfig()

        self.n_added: int = 0
        self.n_skipped: int = 0
        self.last_u_mem: float = 1.0
        self.last_shift_active: bool = False
        self.n_sample_calls: int = 0
        self.n_recent_sample_calls: int = 0

    def _sample_recent_batch_inds(self, batch_size: int, recent_window: int) -> np.ndarray:
        if batch_size <= 0:
            return np.array([], dtype=np.int64)

        size = self.size()
        if size <= 0:
            return np.array([], dtype=np.int64)

        window = int(min(max(1, recent_window), size))

        if self.full:
            offsets = np.random.randint(0, window, size=batch_size)
            batch_inds = (self.pos - 1 - offsets) % self.buffer_size
        else:
            start = max(0, self.pos - window)
            batch_inds = np.random.randint(start, self.pos, size=batch_size)

        return batch_inds.astype(np.int64)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        cfg = self.arc_config
        info0 = infos[0] if infos else {}
        self.last_u_mem = float(info0.get("arc_u_mem", self.last_u_mem))
        self.last_shift_active = bool(info0.get("arc_shift_active", self.last_shift_active))

        if (
            cfg.enable
            and cfg.skip_add_when_u_mem_low
            and infos
            and self.size() >= cfg.min_transitions_to_gate
        ):
            info0 = infos[0] if infos else {}
            u_mem = float(info0.get("arc_u_mem", 1.0))
            shift_active = bool(info0.get("arc_shift_active", False))

            if u_mem < cfg.u_mem_threshold and not (cfg.bypass_when_shift_active and shift_active):
                self.n_skipped += 1
                return

        super().add(obs=obs, next_obs=next_obs, action=action, reward=reward, done=done, infos=infos)
        self.n_added += 1

    def sample(self, batch_size: int, env=None):
        self.n_sample_calls += 1
        cfg = self.arc_config
        if (
            cfg.enable
            and cfg.shift_sample_recent
            and self.last_shift_active
            and self.size() > 0
            and cfg.shift_recent_window > 0
        ):
            recent_batch = int(round(float(batch_size) * float(cfg.shift_recent_fraction)))
            recent_batch = max(0, min(batch_size, recent_batch))
            recent_inds = self._sample_recent_batch_inds(recent_batch, cfg.shift_recent_window)

            if recent_batch == batch_size:
                self.n_recent_sample_calls += 1
                return self._get_samples(recent_inds, env=env)

            other_batch = batch_size - recent_batch
            other_inds = np.random.randint(0, self.size(), size=other_batch).astype(np.int64)
            batch_inds = np.concatenate([recent_inds, other_inds], axis=0)
            self.n_recent_sample_calls += 1
            return self._get_samples(batch_inds, env=env)

        return super().sample(batch_size=batch_size, env=env)

    def get_gate_stats(self) -> dict[str, float]:
        total = self.n_added + self.n_skipped
        skipped_ratio = (self.n_skipped / total) if total > 0 else 0.0
        return {
            "n_added": float(self.n_added),
            "n_skipped": float(self.n_skipped),
            "skipped_ratio": float(skipped_ratio),
            "last_u_mem": float(self.last_u_mem),
            "last_shift_active": float(1.0 if self.last_shift_active else 0.0),
            "sample_calls": float(self.n_sample_calls),
            "recent_sample_calls": float(self.n_recent_sample_calls),
            "recent_sample_ratio": float(self.n_recent_sample_calls / self.n_sample_calls) if self.n_sample_calls > 0 else 0.0,
        }
