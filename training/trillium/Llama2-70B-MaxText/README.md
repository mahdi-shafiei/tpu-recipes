# Instructions for training LLAMA2-70B-Maxtext on TPU trillium

## XPK setup
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/training/trillium/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxtext 
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/training/trillium/MAXTEXT_README.md) to install maxtext and build docker image

## Run Maxtext LLAMA2-70B workloads on GKE

### Test Env
jaxlib=0.4. 35

libtpu-nighly=20241028

[maxtext](https://github.com/AI-Hypercomputer/maxtext.git)@2e1ebad7c660e45d2f020ef025d74cc90e2f0eb3

### Starting workload

From the MaxText root directory, start your LLAMA2-70B workload

```
python3 benchmarks/benchmark_runner.py --project=${PROJECT} --zone={zone} --device_type=v6e-256 --num_slices=1  --cluster_name=${CLUSTER_NAME} --base_output_directory=${OUTPUT_DIR} \
--model_name="llama2_70b_4096" --libtpu_version=20241028 --base_docker_image=maxtext_base_image
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 17, seconds: 17.373, TFLOP/s/device: 419.193, Tokens/s/device: 943.086, total_weights: 4194304, loss: -0.004
```