import os

import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:" + str(port),
        rank=rank,
        world_size=world_size
    )


def cleanup():
    dist.destroy_process_group()


def run_target(target_subroutine, world_size, CF):
    mp.spawn(
        target_subroutine,
        args=(world_size, CF),
        nprocs=world_size,
        join=True
    )
