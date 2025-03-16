#!/bin/bash
#SBATCH --output=slurm_out/%x-%j.out
#SBATCH --nodes=2
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem-per-gpu=60G
#SBATCH --ntasks-per-node=1
#SBATCH --time 00:10:00

set -e # stop bash script on first error

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=$((${SLURM_JOB_ID} % 16384 + 49152))
export MASTER_PORT=29500

# export NCCL_DEBUG="INFO"
# use the 100Gbps network interface
export NCCL_SOCKET_IFNAME=fabric

# For environments
source ~/.bashrc
mamba activate AIObs
source set_env.sh # for dynolog

# Run dynolog daemon and export to output file according to the node
# srun dynolog --enable-ipc-monitor=true > slurm_out/dynolog-${SLURM_JOB_NODELIST}.out 2>&1 &
# sleep 5
# srun python main.py
srun bash srun_wrapper.sh