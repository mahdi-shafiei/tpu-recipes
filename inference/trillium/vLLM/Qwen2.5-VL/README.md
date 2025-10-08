# Serve Qwen2.5-VL with vLLM on TPU VMs

In this guide we show how to serve
[Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

## Step 0: Install `gcloud cli`

You can reproduce this experiment from your dev environment
(e.g. your laptop).
You need to install `gcloud` locally to complete this tutorial.

To install `gcloud cli` please follow this guide:
[Install the gcloud CLI](https://cloud.google.com/sdk/docs/install#mac)

Once it is installed, you can login to GCP from your terminal with this
command: `gcloud auth login`.

## Step 1: Create a v6e TPU instance

We create a single VM. For Qwen2.5-VL-7B, 1 chip is sufficient. If you need a different number of
chips, you can set a different value for `--topology` such as `1x1`,
`2x4`, etc.

To learn more about topologies: [v6e VM Types](https://cloud.google.com/tpu/docs/v6e#vm-types).

> **Note:** Acquiring on-demand TPUs can be challenging due to high demand. We recommend using [Queued Resources](https://cloud.google.com/tpu/docs/queued-resources) to ensure you get the required capacity.

### Option 1: Create an on-demand TPU VM

This command attempts to create a TPU VM immediately.

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project

# this command creates a tpu vm with 1 Trillium (v6e) chip - adjust it to suit your needs
gcloud alpha compute tpus tpu-vm create $TPU_NAME \
    --type v6e --topology 1x1 \
    --project $PROJECT --zone $ZONE --version v2-alpha-tpuv6e
```

### Option 2: Use Queued Resources (Recommended)

With Queued Resources, you submit a request for TPUs and it gets fulfilled
when capacity is available.

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project
export QR_ID=your-queued-resource-id # e.g. my-qr-request

# This command requests a v6e-1 (1 chip). Adjust accelerator-type for different sizes.
gcloud alpha compute tpus queued-resources create $QR_ID \
    --node-id $TPU_NAME \
    --project $PROJECT --zone $ZONE \
    --accelerator-type v6e-1 \
    --runtime-version v2-alpha-tpuv6e
```

You can check the status of your request with:

```bash
gcloud alpha compute tpus queued-resources list --project $PROJECT --zone $ZONE
```

Once the state is `ACTIVE`, your TPU VM is ready and you can proceed to the next steps.

## Step 2: ssh to the instance

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone=$ZONE
```

## Step 3: Use the latest vllm docker image for TPU

```bash
export DOCKER_URI=vllm/vllm-tpu:nightly
```

> **!!Important!!:** As of 10/07/2025, the `vllm/vllm-tpu:nightly` Docker image does not yet include the necessary `tpu_inference` updates to support multi-modal models like Qwen2.5-VL. The following instructions require installing [vllm-tpu](https://docs.vllm.ai/en/latest/getting_started/installation/google_tpu.html#set-up-using-python) and [tpu-inference](https://github.com/vllm-project/tpu-inference) manually on the TPU VM and run directly from the source (user can also choose to build a local Docker image) instead of using Docker published images. For production environments, we recommend waiting for an official `vllm-tpu` Docker image release that includes this support.

## Step 4: Run the docker container in the TPU instance

```bash
sudo docker run -it --rm --name $USER-vllm --privileged --net=host \
    -v /dev/shm:/dev/shm \
    --shm-size 17gb \
    -p 8000:8000 \
    --entrypoint /bin/bash ${DOCKER_URI}
```

Set `--shm-size` based on the model weights; 17GB is sufficient for a 7B model.

## Step 5: Set up env variables

Export your hugging face token along with other environment variables inside
the container.

```bash
export HF_HOME=/dev/shm
export HF_TOKEN=<your HF token>
```

## Step 6: Serve the model

Now we start the vllm server.
Make sure you keep this terminal open for the entire duration of this experiment.

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.98 \
    --max-model-len 16384 \
    --limit-mm-per-prompt '{"image": 10, "video": 0}' \
    --mm-processor-kwargs '{"max_pixels": 1003520}' \
    --guided-decoding-backend "xgrammar" \
    --disable-chunked-mm-input
```

Use `--limit-mm-per-prompt` to set the maximum number of images per request and `--mm-processor-kwargs '{"max_pixels": 1003520}'` to define the maximum pixel size for each image. Ensure these settings match the requirements of your requests.

It takes a few minutes to prepare the multi modal server.
Once you see the below snippet in the logs, it means that the server is ready
to serve requests or run benchmarks:

```bash
INFO:     Started server process [461368]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## Step 7: Prepare the test environment

Open a new terminal to test the server and run the benchmark (keep the previous terminal open).

First, we ssh into the TPU vm via the new terminal:

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project

gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone=$ZONE
```

## Step 8: access the running container

```bash
sudo docker exec -it $USER-vllm bash
```

## Step 9: Run the benchmarking

Finally, we are ready to run the benchmark:

```bash
export HF_TOKEN=<your HF token>

cd /workspace/vllm

vllm bench serve \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset-name random-mm \
    --num-prompts 128 \
    --backend openai-chat \
    --endpoint "/v1/chat/completions" \
    --random-mm-bucket-config '{(736, 736, 1): 1.0}' \
    --random-mm-base-items-per-request 6 \
    --random-mm-num-mm-items-range-ratio 0.67 \
    --random-mm-limit-mm-per-prompt '{"image": 10, "video": 0}'
```

This benchmark uses the `random-mm` dataset to generate requests with random combinations of images and text. The following parameters control the generation:

* `--random-mm-bucket-config`: Specifies image resolutions (H, W, T=1) and their probabilities (sum to 1).
* `--random-mm-base-items-per-request (n)`: Base number of images per request.
* `--random-mm-num-mm-items-range-ratio (r)`: Varies the number of images per request within the range [floor(n·(1−r)), ceil(n·(1+r))].
* `--random-mm-limit-mm-per-prompt`: Sets a hard limit on items per modality (e.g., `{'image': 10}`).

In the example command, requests use a fixed image size of 736x736, and each request contains 2-10 images (`n=6`, `r=0.67`), with an upper limit of 10 images via `--random-mm-limit-mm-per-prompt`. For parameter details, see [random multi-modal APIs](https://github.com/vllm-project/vllm/pull/23119).

The snippet below is what you’d expect to see - the numbers vary based on the vllm version, the model size and the TPU instance type/size.

```bash
============ Serving Benchmark Result ============
Successful requests:                     xxxxxxx
Benchmark duration (s):                  xxxxxxx
Total input tokens:                      xxxxxxx
Total generated tokens:                  xxxxxxx
Request throughput (req/s):              xxxxxxx
Output token throughput (tok/s):         xxxxxxx
Total Token throughput (tok/s):          xxxxxxx
---------------Time to First Token----------------
Mean TTFT (ms):                          xxxxxxx
Median TTFT (ms):                        xxxxxxx
P99 TTFT (ms):                           xxxxxxx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xxxxxxx
Median TPOT (ms):                        xxxxxxx
P99 TPOT (ms):                           xxxxxxx
---------------Inter-token Latency----------------
Mean ITL (ms):                           xxxxxxx
Median ITL (ms):                         xxxxxxx
P99 ITL (ms):                            xxxxxxx
==================================================
```
