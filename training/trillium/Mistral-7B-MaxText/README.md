# Instructions for training Mistral-7B-MaxText on TPU trillium (v6e-8)

## XPK setup
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxtext

### Test Env
Use the MaxText [tpu-recipes-v0.1.0](https://github.com/AI-Hypercomputer/maxtext/releases/tag/tpu-recipes-v0.1.0) tag to run this recipe. This can be done using the following command from your local MaxText directory:

```
git checkout tpu-recipes-v0.1.0
```

### Install MaxText and Build Docker Image
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/MAXTEXT_README.md) to install maxtext and build docker image. In step 2, be sure to use the jax-stable-stack image containing JAX 0.4.37:

```
BASE_IMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.4.37-rev1
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE=${BASE_IMAGE}
```

## Run Maxtext Mistral-7B workloads on GKE

### Starting workload

From the MaxText root directory, start your Mistral-7B workload.
```
python3 benchmarks/benchmark_runner.py xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-8 \
    --num_slices=1  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="mistral_7b" \
    --base_docker_image=maxtext_base_image
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 7, seconds: 6.051, TFLOP/s/device: 451.236, Tokens/s/device: 8123.462, total_weights: 393216, loss: 6.765
```
If you would like to run on multiple slices of v6e-8, you may modify the `--num_slices` flag.
