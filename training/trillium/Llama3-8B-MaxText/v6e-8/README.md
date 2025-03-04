# Instructions for training Llama3.1-8B-MaxText on TPU trillium (v6e-8)

## XPK setup
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/XPK_README.md) to create your GKE cluster with XPK

## Prep for Maxtext 
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/MAXTEXT_README.md) to install maxtext and build docker image. In step 2, be sure to use the jax-stable-stack image containing JAX 0.4.37:

```
BASE_IMAGE=BASEIMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.4.37-rev1
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE=${BASE_IMAGE}
```

## Run Maxtext Llama3.1-8B workloads on GKE

### Test Env
Use the MaxText [tpu-recipes-v0.1.0](https://github.com/AI-Hypercomputer/maxtext/releases/tag/tpu-recipes-v0.1.0) tag to run this recipe. This can be done using the following command from you local MaxText directory:

```
git checkout tpu-recipes-v0.1.0
```

### Starting workload

From the MaxText root directory, start your Llama3.1-8B workload. Note: this benchmark uses a different model name than the equivalent v6e-256 recipe.
```
python3 benchmarks/benchmark_runner.py xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-8 \
    --num_slices=1  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="llama3_1_8b_8192_no_collective_matmul" \
    --base_docker_image=maxtext_base_image
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 7, seconds: 3.416, TFLOP/s/device: 416.601, Tokens/s/device: 7193.579, total_weights: 196608, loss: 6.634
```
If you would like to run on multiple slices of v6e-8, you may modify the `--num_slices` flag.
