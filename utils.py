import os
import torch

def setup_distributed_env():
    # initialize distributed environment
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        num_gpus = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
    else:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            num_gpus = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
        else:
            rank = 0
            num_gpus = 1
            local_rank = 0

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.distributed.init_process_group(backend='nccl', 
                                        init_method='env://',
                                        world_size=num_gpus,
                                        rank=rank)
    return device, rank, num_gpus

def cleanup_distributed_env():
    torch.distributed.destroy_process_group()