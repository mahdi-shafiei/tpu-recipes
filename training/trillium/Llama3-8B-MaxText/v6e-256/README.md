# Instructions for training Llama3.1-8B-MaxText on TPU trillium (v6e-256)

## XPK setup
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxtext 
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/MAXTEXT_README.md) to install maxtext and build docker image.
Be sure to use the jax-stable-stack image containing jax0.4.37.

## Run Maxtext Llama3.1-8B workloads on GKE

### Test Env
jaxlib=0.4.37

libtpu-nightly=20241209

[maxtext](https://github.com/AI-Hypercomputer/maxtext.git)@3ad02ba70b122cec488aa5d017925aa00f5ef15f

### Starting workload

From the MaxText root directory, start your Llama3.1-8B workload. Note: this benchmark uses a different model name than the equivalent v6e-8 recipe.
```
python3 benchmarks/benchmark_runner.py --project=$PROJECT --zone=$ZONE --device_type=v6e-256 --num_slices=1  --cluster_name=${CLUSTER_NAME} --base_output_directory=${OUTPUT_DIR} \
--model_name="llama3_1_8b_8192" --libtpu_version=20241209 --base_docker_image maxtext_base_image
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 7, seconds: 4.225, TFLOP/s/device: 449.171, Tokens/s/device: 7755.989, total_weights: 8388608, loss: 0.000
```
If you would like to run on multiple slices of v6e-256, you may modify the `--num_slices` flag.