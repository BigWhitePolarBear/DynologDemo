#!/bin/bash

# Start Dynolog in the background
dynolog --enable-ipc-monitor=true > slurm_out/dynolog-${SLURM_PROCID}.out 2>&1 &

# Start the Python script in the foreground
python main.py

# Wait for all background processes to finish
# wait