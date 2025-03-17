#!/bin/bash

# Start Dynolog in the background
dynolog --enable-ipc-monitor=true > slurm_out/dynolog-${SLURM_PROCID}.out 2>&1 &

# Start the Python script in the foreground
python main.py --data_ratio 0.05 --batch_size 4 --freeze_layers_ratio 0.7 --model_id meta-llama/Llama-3.2-1B