# Microbenchmarks

## Setup

Set up a v6e TPU VM:
```
gcloud compute tpus tpu-vm create $TPU_NAME /
        --zone=$ZONE /
        --accelerator-type=v6e-1  /
        --version=v2-alpha-tpuv6e
```
See https://cloud.google.com/tpu/docs/regions-zones for available zones.

SSH into the VM:
```
gcloud compute ssh $TPU_NAME --zone=$ZONE
```

Clone the repo and install the dependencies:
```bash
git clone https://github.com/chishuen/tpu-recipes.git
cd tpu-recipes/microbenchmarks
pip install -r requirements.txt
```

## Run Matmul Benchmark

Usage example:
```
python benchmark_matmul.py \
  --dim 4096 4096 4096 \
  --libtpu_args=--xla_tpu_scoped_vmem_limit_kib=65536 \
  --matcher="jit_matmul.*"
```

Example output:
```
dtype: bfloat16, matrix Dimensions: (16384, 16384, 16384), time taken (median): 10.584555 ms, TFLOPs/sec: 831.0309712791892
```

Run `python benchmark_matmul.py -h` to view the how to set the arguments.


## HBM Bandwidth Benchmark

Usage example:
```
python benchmark_hbm.py \
  --num_elements=16777216 \
  --matcher="jit_my_copy.*"
```

Example output:
```
Tensor size: 32.0 MBs, time taken (median): 0.0461 ms, bandwidth: 1454.54 GBps
```

Run `python benchmark_hbm.py -h` to view the how to set the arguments.