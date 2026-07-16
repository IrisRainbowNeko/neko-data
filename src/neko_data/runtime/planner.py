"""Deterministic epoch, node, rank, and worker shard assignment."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

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

