# ImageNet Classification
Framework for training classification networks on ImageNet dataset.

## Supported Features  
* Multi-machine & multi-GPU distributed training
* Rand augmentation
* Model Exponential Moving Average (EMA)
* Optimizer (RMSProp, SGD)
* LR scheduler (warm-up, step, cosine)
* Auxiliary Loss Tower

## Supported Models
* NAS models with MobileNetV2(SE)-based macro search space
* NAS models with DARTS-based micro search space

---

## Usage

Put ImageNet train, val, and meta directories into ./data .

### Training
  
```shell
sh tools/dist_train.sh <partition> <gpus> <strategy_config> <model> <model_config> 
```

* For slurm users:  
```shell
sh tools/slurm_train.sh <partition> <gpus> <strategy_config> <model> <model_config> 
```

For example:
```
sh tools/slurm_train.sh VA 8 config/strategies/mbv2_se_aa.yaml nas_model config/models/GreedyNAS/GreedyNAS-B.yaml
```
---

## Benchmark

