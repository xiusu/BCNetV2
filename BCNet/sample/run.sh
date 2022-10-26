#!/usr/bin/env bash
#export PYTHONPATH=$ROOT:/mnt/lustre/yangmingmin/cellular:$PYTHONPATH
now=$(date +"%Y%m%d_%H%M%S")
GLOG_logtostderr=-1 GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
    srun --mpi=pmi2 --job-name retrain -x SH-IDC1-10-198-6-[131,132,138] --partition=$1 -n$2 --gres=gpu:$2 --ntasks-per-node=$2 \
        python -u tools/agent_run.py $3

