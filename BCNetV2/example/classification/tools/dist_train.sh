#!/bin/bash
PARTITION=$1
GPUS=$2
CONFIG=$3
MODEL=$4
MODEL_CONFIG=$5
PY_ARGS=${@:6}

PYTHONPATH=$PWD:$PYTHONPATH PYTHONWARNINGS=ignore GLOG_logtostderr=-1 GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
    srun --mpi=pmi2 --job-name retrain --partition=${PARTITION} -n1 --gres=gpu:${GPUS} --ntasks-per-node=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
        train.py -c ${CONFIG} --model ${MODEL} --model-config ${MODEL_CONFIG} ${PY_ARGS}
