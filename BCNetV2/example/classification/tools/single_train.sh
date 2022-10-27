#!/bin/bash
PARTITION=$1
CONFIG=$2
MODEL=$3
MODEL_CONFIG=$4
PY_ARGS=${@:5}

PYTHONPATH=$PWD:$PYTHONPATH PYTHONWARNINGS=ignore GLOG_logtostderr=-1 GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
    srun --mpi=pmi2 --job-name retrain --partition=${PARTITION} -n1 --gres=gpu:1 --ntasks-per-node=1 \
        python train.py -c ${CONFIG} --model ${MODEL} --model-config ${MODEL_CONFIG} ${PY_ARGS}
