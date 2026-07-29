from neko_data.runtime import RuntimeContext


def test_runtime_context_from_torchrun_environment():
    context = RuntimeContext.from_env({
        "WORLD_SIZE": "16",
        "RANK": "10",
        "LOCAL_WORLD_SIZE": "8",
        "LOCAL_RANK": "2",
        "GROUP_RANK": "1",
        "GROUP_WORLD_SIZE": "2",
    })

    assert context.world_size == 16
    assert context.global_rank == 10
    assert context.num_nodes == 2
    assert context.node_rank == 1
    assert context.local_world_size == 8
    assert context.local_rank == 2


def test_runtime_context_from_slurm_environment():
    context = RuntimeContext.from_env({
        "SLURM_NTASKS": "8",
        "SLURM_PROCID": "5",
        "SLURM_NTASKS_PER_NODE": "4",
        "SLURM_LOCALID": "1",
        "SLURM_NODEID": "1",
        "SLURM_NNODES": "2",
    })

    assert context.world_size == 8
    assert context.global_rank == 5
    assert context.node_rank == 1
    assert context.num_nodes == 2
    assert context.local_world_size == 4
    assert context.local_rank == 1
