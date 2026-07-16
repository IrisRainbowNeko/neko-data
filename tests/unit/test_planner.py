from neko_data.contract.schema import ShardRecord
from neko_data.runtime.planner import RuntimeContext, ShardPlanner


def _shards(count):
    return [ShardRecord(f"{index}.tar", "train", 1, 1, str(index)) for index in range(count)]


def test_planner_covers_each_shard_once_across_nodes():
    shards = _shards(24)
    planner = ShardPlanner(shards, seed=10)
    assignments = []
    for node in range(2):
        for local_rank in range(2):
            context = RuntimeContext(
                world_size=4,
                global_rank=node * 2 + local_rank,
                node_rank=node,
                num_nodes=2,
                local_world_size=2,
                local_rank=local_rank,
            )
            assignments.extend(planner.plan(0, context))
    assert {item.sha256 for item in assignments} == {str(index) for index in range(24)}


def test_planner_changes_order_each_epoch():
    planner = ShardPlanner(_shards(12), seed=10)
    first = [item.sha256 for item in planner.plan(0, RuntimeContext())]
    second = [item.sha256 for item in planner.plan(1, RuntimeContext())]
    assert first != second


def test_planner_covers_each_shard_once_across_workers_per_epoch():
    shards = _shards(24)
    planner = ShardPlanner(shards, seed=10)
    expected = {str(index) for index in range(24)}
    for epoch in (0, 1):
        assignments = []
        for worker_id in range(3):
            context = RuntimeContext(worker_id=worker_id, num_workers=3)
            assignments.extend(planner.plan(epoch, context))
        values = [item.sha256 for item in assignments]
        assert len(values) == len(set(values))
        assert set(values) == expected

