"""Synthetic RTS-shaped data; contains no game assets, private data or models."""

import numpy as np

from neko_data.contract import TypedSample


def source(episodes=4):
    for episode in range(episodes):
        units = [np.zeros((count, 8), dtype=np.float32) for count in (0, 3 + episode, 2)]
        yield TypedSample(
            sample_key=f"episode-{episode}-segment-0",
            data={
                "observation": {"units": units, "spatial": np.zeros((3, 9, 16, 16), dtype=np.uint8)},
                "actions": ({"command": 1, "members": np.array([0, 2], dtype=np.int64)}, None),
                "legal_masks": [np.ones(len(step), dtype=np.bool_) for step in units],
                "behavior_logp": np.array([-0.5, -0.8, -0.2], dtype=np.float32),
                "reward": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "terminated": np.array([False, False, True]),
                "truncated": np.zeros(3, dtype=np.bool_),
                "bootstrap": None,
            },
            metadata={"episode_id": episode, "policy_version": 7, "outcome": "win", "synthetic": True},
        )
