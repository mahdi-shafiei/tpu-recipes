python3 ~/xpk/xpk.py workload create \
    --cluster=${CLUSTER_NAME} \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --device-type=v6e-256 \
    --command="git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git && cd accelerator-microbenchmarks && git checkout trillium-collectives && pip install -r requirements.txt && echo '4096 41943040 314572800' > /proc/sys/net/ipv4/tcp_rmem && export LIBTPU_INIT_ARGS='--megascale_grpc_premap_memory_bytes=17179869184 --xla_tpu_enable_sunk_dcn_allreduce_done_with_host_reduction=true' && python src/run_benchmark.py --config=configs/1x_v6e_256.yaml" \
    --num-slices=1 \
    --docker-image=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.4.37-rev1 \
    --workload=${WORKLOAD_NAME}
