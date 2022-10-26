#!/usr/bin/env bash

set -x
now=$(date +"%Y%m%d-%H%M%S")

partition=$1
config=$2
gpus=$3
jobtag=$4

workspace=$(cat "$config" | grep -E '^save:' | cut -d' ' -f2 | tr -d "'" | tr -d '"')
mkdir -p $workspace

g=$(($gpus<8?$gpus:8))

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
PYTHONPATH=..:$PYTHONPATH \
srun --mpi=pmi2 -p $partition --job-name=$jobtag -x SH-IDC1-10-5-38-[33-42] \
  --gres=gpu:$g -n$gpus --ntasks-per-node=$g --cpus-per-task=5 \
  python -u imagenet.py --config $config 2>&1 | tee $workspace/train.log-$now
