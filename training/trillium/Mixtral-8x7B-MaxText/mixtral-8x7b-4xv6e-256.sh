python3 benchmarks/benchmark_runner.py --project=$PROJECT --zone=$ZONE --device_type=v6e-256 --num_slices=4  --cluster_name=${CLUSTER_NAME} --base_output_directory=${OUTPUT_DIR} \
--model_name="mixtral_8x7b_dropped" --libtpu_version=20241119 --base_docker_image maxtext_base_image
