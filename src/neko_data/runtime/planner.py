"""Deterministic epoch, node, rank, and worker shard assignment."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, replace
from typing import Mapping

from ..contract.schema import ShardRecord


@dataclass(frozen=True)
class RuntimeContext:
    """Distributed placement information for one training process."""

    world_size: int = 1
    global_rank: int = 0
    node_rank: int = 0
    num_nodes: int = 1
    local_world_size: int | None = None
    local_rank: int | None = None
    worker_id: int = 0
    num_workers: int = 1

    def __post_init__(self) -> None:
        if self.world_size <= 0 or self.num_nodes <= 0 or self.num_workers <= 0:
            raise ValueError("world_size, num_nodes, and num_workers must be positive")
        if not 0 <= self.global_rank < self.world_size:
            raise ValueError("global_rank must be within world_size")
        if not 0 <= self.node_rank < self.num_nodes:
            raise ValueError("node_rank must be within num_nodes")
        local_world_size = self.local_world_size or max(1, self.world_size // self.num_nodes)
        local_rank = self.local_rank if self.local_rank is not None else self.global_rank % local_world_size
        if not 0 <= local_rank < local_world_size:
            raise ValueError("local_rank must be within local_world_size")
        if not 0 <= self.worker_id < self.num_workers:
            raise ValueError("worker_id must be within num_workers")
        object.__setattr__(self, "local_world_size", local_world_size)
        object.__setattr__(self, "local_rank", local_rank)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "RuntimeContext":
        """Build a process context from torchrun or SLURM environment variables."""
        values = os.environ if environ is None else environ

        def integer(*names: str, default: int) -> int:
            for name in names:
                value = values.get(name)
                if value is not None and value != "":
                    return int(value)
            return default

        world_size = integer("WORLD_SIZE", "SLURM_NTASKS", default=1)
        global_rank = integer("RANK", "SLURM_PROCID", default=0)
        local_world_size = integer("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE", default=world_size)
        local_world_size = max(1, min(local_world_size, world_size))
        local_rank = integer(
            "LOCAL_RANK",
            "SLURM_LOCALID",
            default=global_rank % local_world_size,
        )
        inferred_nodes = max(1, (world_size + local_world_size - 1) // local_world_size)
        node_rank = integer(
            "GROUP_RANK",
            "NODE_RANK",
            "SLURM_NODEID",
            default=global_rank // local_world_size,
        )
        num_nodes = integer(
            "GROUP_WORLD_SIZE",
            "NNODES",
            "SLURM_NNODES",
            default=inferred_nodes,
        )
        num_nodes = max(num_nodes, node_rank + 1)
        return cls(
            world_size=world_size,
            global_rank=global_rank,
            node_rank=node_rank,
            num_nodes=num_nodes,
            local_world_size=local_world_size,
            local_rank=local_rank,
        )

    def for_worker(self, worker_id: int, num_workers: int) -> "RuntimeContext":
        return replace(self, worker_id=worker_id, num_workers=max(1, num_workers))


class ShardPlanner:
    def __init__(self, shards: list[ShardRecord], seed: int = 42) -> None:
        self.shards = list(shards)
        self.seed = seed

    def shuffled(self, epoch: int) -> list[ShardRecord]:
        shards = list(self.shards)
        random.Random(self.seed + epoch).shuffle(shards)
        return shards

    def plan(self, epoch: int, context: RuntimeContext) -> list[ShardRecord]:
        shards = self.shuffled(epoch)
        node_shards = shards[context.node_rank::context.num_nodes]
        rank_shards = node_shards[context.local_rank::context.local_world_size]
        return rank_shards[context.worker_id::context.num_workers]
