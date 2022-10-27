#!/usr/bin/env bash
PARTITION=$1
GPUS=$2
CONFIG=$3
MODEL=$4
MODEL_CONFIG=$5
PY_ARGS=${@:6}

N=${GPUS}
if [ ${GPUS} -gt 8 ]
then
    echo "multi machine"
    N=8
fi
# -x BJ-IDC1-10-10-16-[44,48]

PYTHONPATH=$PWD:$PYTHONPATH PYTHONWARNINGS=ignore GLOG_logtostderr=-1 GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
    srun --mpi=pmi2 --job-name train -x BJ-IDC1-10-10-16-[46,48,49,51,52,53,60,71,58,61-62,64,66,68,69,84] --partition=${PARTITION} -n${GPUS} --gres=gpu:${N} --ntasks-per-node=${N} \
        python -u train.py -c ${CONFIG} --model ${MODEL} --model-config ${MODEL_CONFIG} --slurm ${PY_ARGS}
