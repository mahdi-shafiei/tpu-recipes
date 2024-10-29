# Instructions for training Mixtral-8x7B-MaxText on TPU trillium

## XPK setup
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/training/trillium/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxtext 
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/training/trillium/MAXTEXT_README.md) to install maxtext and build docker image

## Run Maxtext Mixtral-8x7B workloads on GKE

### Starting workload

From the MaxText root directory, start your Mixtral workload.

Bf16 run:
```
python3 benchmarks/benchmark_runner.py --project=${PROJECT} --zone={zone} --device_type=v6e-256 --num_slices=1  --cluster_name=${CLUSTER_NAME} --base_output_directory=${OUTPUT_DIR} \
--model_name="mixtral_8x7b_dropped" --libtpu_version=20241009 --base_docker_image=maxtext_base_image
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 19, seconds: 8.982, TFLOP/s/device: 302.566, Tokens/s/device: 3648.273, total_weights: 8388608, loss: 0.030
```

Int8 run:
```
python3 benchmarks/benchmark_runner.py --project=${PROJECT} --zone={zone} --device_type=v6e-256 --num_slices=1  --cluster_name=${CLUSTER_NAME} --base_output_directory=${OUTPUT_DIR} \
--model_name="mixtral_8x7b_dropped_int8" --libtpu_version=20241009 --base_docker_image=maxtext_base_image
```