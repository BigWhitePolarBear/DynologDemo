# Introduction

This a simple experimental test for dynolog, which is a profiling tool for PyTorch. The settings in this repo mainly considers the slurm environment at ROCS Testbed.

# Prerequisite

First, [dynolog](https://github.com/facebookincubator/dynolog) needs to be installed. `dyno` and `dynolog` commands should be avaliable.

Dynolog need two libs file from openssl 1.1.1, which has be deprecated for a long time. They are placed in this directory, otherwise it will be more bothersome to keep a old version openssl in your environment.

Run the follwing command if you wish to do with a interactive way:

```shell
source set_env.sh
```

# Run

- For single node, `python single-node.py` can start a finetuning procedure. However, which means you should start a dynolog daemon yourself.
- For two nodes, use `sbatch start_dp_train.sh` to submit it onto slurm cluster.

Here is a example of start tracing some data from client:

```shell
dyno --hostname mcnode23 --port 1778 gputrace --job-id 3088 --log-file trace_out/trace.json --duration-ms 5000
```

, where job-id is the slurm job-id. The trace output can be opened with [Perfettor](https://ui.perfetto.dev/) or other trace viewer.

For more details, please refer to the dynolog official documentation.