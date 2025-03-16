#!/bin/bash

# For dynolog server
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/lib

# For Pytorch to register with Dynolog
export KINETO_USE_DAEMON=1
export KINETO_DAEMON_INIT_DELAY_S=1