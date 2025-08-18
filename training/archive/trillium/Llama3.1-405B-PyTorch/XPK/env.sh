#!/bin/bash

# Environment variables associated with XPK on GCP.
export ZONE=...
export PROJECT=...
export TPU_TYPE=v6e-256
export NUM_SLICE=2
export CLUSTER_NAME=xpk-$USER-... # use existing CLUSTER if you have

# Environment variables associated with training config.
export BATCH_PER_DEVICE=1
export SEQUENCE_LENGTH=8192
export MAX_STEP=50
export WORKLOAD_NAME=${USER}-xpk-${TPU_TYPE}-... # Your workload name. Need to update for different run.
export BASE_DOCKER_IMAGE=us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-tpu-llama@sha256:d3a4c09cd13dab2af8129e8438b0acf3f8b5a2370b94b69e2e3aac16530e3664
export PROFILE_LOG_DIR=... # GSC bucket to store profile in form of gs://...
export HF_TOKEN=... # Add your own Hugging face token to download model
