#!/bin/bash

COMMAND=$(cat <<EOF
cd /root/ && export PJRT_DEVICE=TPU && export XLA_USE_SPMD=1 && export ENABLE_PJRT_COMPATIBILITY=true && export XLA_IR_DEBUG=1 && export XLA_HLO_DEBUG=1 && export PROFILE_EPOCH=0 && export PROFILE_STEP=3 && export PROFILE_DURATION_MS=100000 && export PROFILE_LOGDIR=${PROFILE_LOG_DIR} && export XLA_PERSISTENT_CACHE_PATH=/app/xla_cache/ && export TPU_LIBRARY_PATH=/root/_libtpu.so && export NUM_SLICE=${NUM_SLICE} && huggingface-cli login --token=${HF_TOKEN} && python3 transformers/examples/pytorch/language-modeling/run_clm.py --dataset_name=wikitext --dataset_config_name=wikitext-103-raw-v1 --per_device_train_batch_size=1024 --do_train --output_dir=test-clm --overwrite_output_dir --config_name=config_70b.json --cache_dir=cache --tokenizer_name=meta-llama/Meta-Llama-3-70B --block_size=8192 --optim=adafactor --save_strategy=no --logging_strategy=no --fsdp="full_shard" --fsdp_config=fsdp_config.json --torch_dtype=bfloat16 --dataloader_drop_last=yes --flash_attention --max_steps=50
EOF
)

python3 xpk/xpk.py workload create \
--cluster ${CLUSTER_NAME} \
--base-docker-image=${BASE_DOCKER_IMAGE} \
--workload=${WORKLOAD_NAME} \
--tpu-type=v6e-256 \
--num-slices=${NUM_SLICE} \
--on-demand \
--zone=$ZONE \
--project=$PROJECT \
--enable-debug-logs \
--command="$COMMAND"
