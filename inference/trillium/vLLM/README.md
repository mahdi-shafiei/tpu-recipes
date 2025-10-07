# Serve vLLM on Trillium TPUs (v6e)

This repository provides examples demonstrating how to deploy and serve vLLM on Trillium TPUs using GCE (Google Compute Engine) for a select set of models.

- [Llama3.1-8B/70B](./Llama3.1/README.md)
- [Qwen2.5-32B](./Qwen2.5-32B/README.md)
- [Qwen2.5-VL-7B](./Qwen2.5-VL/README.md)
- [Qwen3-4B/32B](./Qwen3/README.md)

These models were chosen for demonstration purposes only. You can serve any model from this list: [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

If you are looking for GKE-based deployment, please refer to this documentation: [Serve an LLM using TPU Trillium on GKE with vLLM](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-vllm-tpu)
