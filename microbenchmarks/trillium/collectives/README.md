# Instructions for running Collectives Benchmark on TPU trillium (v6e-256)

## XPK setup
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/trillium/XPK_README.md) to create your GKE cluster with XPK

## Run Collectives on v6e-256

### Starting workload

Launch the XPK workload, example to run on 1 slice of v6e-256:
```
python3 ~/xpk/xpk.py workload create \
    --cluster=${CLUSTER_NAME} \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --device-type=v6e-256 \
    --command="git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git && cd accelerator-microbenchmarks && git checkout trillium-collectives && pip install -r requirements.txt && echo '4096 41943040 314572800' > /proc/sys/net/ipv4/tcp_rmem && export LIBTPU_INIT_ARGS='--megascale_grpc_premap_memory_bytes=17179869184 --xla_tpu_enable_sunk_dcn_allreduce_done_with_host_reduction=true' && python src/run_benchmark.py --config=configs/1x_v6e_256.yaml" \
    --num-slices=1 \
    --docker-image=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1 \
    --workload=${WORKLOAD_NAME}
```

To run on more than 1 slice, modify the `--num_slices` and `--config` flags to use the target number of slices and the corresponding yaml config file e.g
```
--num_slices=2 --config=configs/2x_v6e_256.yaml 
```

From your workload logs, you should start seeing benchmark logs:
```
psum_dcn: Matrix size: 17408x17408, dtype=<class 'jax.numpy.bfloat16'>, matrix_size_gbyte=0.606076928,achieved_bandwidth_gbyte_s=4.1130934137328214
psum_ici: Matrix size: 17408x17408, dtype=<class 'jax.numpy.bfloat16'>, matrix_size_gbyte=0.606076928,achieved_bandwidth_gbyte_s=235.7595345022845
```

Results will be printed out and also stored at `/tmp/microbenchmarks/collectives`. You can save the stored results to GCS by adding the following to `--command` in the XPK command:
```
gsutil cp -r /tmp/microbenchmarks/collectives gs://<your-gcs-bucket>
```


