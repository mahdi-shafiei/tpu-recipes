#!/bin/bash

# XPK will create a new docker and copy env.sh file under /app/
source /app/env.sh

# Calculate the global batch size
# Extract the number after '-' in TPU_TYPE
TPU_NUM=$(echo "$TPU_TYPE" | grep -oP '(?<=-)\d+')
# Calculate GLOBAL_BATCH_SIZE
GLOBAL_BATCH_SIZE=$(( TPU_NUM * BATCH_PER_DEVICE * NUM_SLICE ))
export GLOBAL_BATCH_SIZE
echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"

# Note --per_device_train_batch_size is the global batch size since we overwrite the dataloader in the HF trainer.
cd /workspace/ && \
export PJRT_DEVICE=TPU && \
export XLA_USE_SPMD=1 && \
export ENABLE_PJRT_COMPATIBILITY=true && \
export XLA_IR_DEBUG=1 && \
export XLA_HLO_DEBUG=1 && \
export PROFILE_EPOCH=0 && \
export PROFILE_STEP=3 && \
export PROFILE_DURATION_MS=450000 && \
export PROFILE_LOGDIR=${PROFILE_LOG_DIR} && \
export XLA_PERSISTENT_CACHE_PATH=/app/xla_cache/ && \
export TPU_LIBRARY_PATH=/workspace/_libtpu.so && \
export NUM_SLICE=${NUM_SLICE} && \

export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2 --megascale_graph_hang_threshold=30m --megascale_graph_within_launch_hang_threshold=30m --megascale_grpc_enable_xor_tracer=false --megascale_grpc_premap_memory_bytes=68719476736 --megascale_grpc_use_chaotic_good=true --megascale_grpc_use_event_engine_allocator=true --grpc_enable_tcp_recv_zerocopy=false --grpc_enable_rpc_receive_coalescing=true"

huggingface-cli login --token=${HF_TOKEN} && \
python3 transformers/examples/pytorch/language-modeling/run_clm.py \
    --dataset_name=wikitext \
    --dataset_config_name=wikitext-103-raw-v1 \
    --per_device_train_batch_size=${GLOBAL_BATCH_SIZE} \
    --do_train \
    --output_dir=test-clm \
    --overwrite_output_dir \
    --config_name=/app/config_405b.json \
    --cache_dir=cache \
    --tokenizer_name=meta-llama/Meta-Llama-3.1-405B \
    --block_size=${SEQUENCE_LENGTH} \
    --optim=adafactor \
    --save_strategy=no \
    --logging_strategy=no \
    --torch_dtype=bfloat16 \
    --dataloader_drop_last=yes \
    --flash_attention \
    --spmd_2d_sharding=4 \
    --max_steps=${MAX_STEP}
