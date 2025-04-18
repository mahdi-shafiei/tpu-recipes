# Prep for MaxText workloads on GKE

> **_NOTE:_** We recommend running these instructions and kicking off your recipe 
workloads from a VM in GCP using Python 3.10.

1. Clone [MaxText](https://github.com/google/maxtext) repo and move to its directory
```shell
git clone https://github.com/google/maxtext.git
cd maxtext
# Checkout either the commit id or MaxText tag. 
# Example: `git checkout tpu-recipes-v0.1.2`
git checkout ${MAXTEXT_COMMIT_ID_OR_TAG}
```

2. Install MaxText dependencies
```shell
bash setup.sh
```

Optional: Use a virtual environment to setup and run your workloads. This can help with errors
like `This environment is externally managed`:
```shell
## One time step of creating the venv
VENV_DIR=~/venvp3
python3 -m venv $VENV_DIR
## Enter your venv.
source $VENV_DIR/bin/activate
## Install dependencies
bash setup.sh
```

> **_NOTE:_** If you use a virtual environment, you must use the same one when running the 
[XPK Installation](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#installation) 
steps linked in the [XPK_README](XPK_README.md) as well as your relevant tpu-recipe workloads.

3. Run the following commands to build the docker image
```shell
# Example BASE_IMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1
BASE_IMAGE=<stable_stack_image_with_desired_jax_version>
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE=${BASE_IMAGE}
```

4. Upload your docker image to Container Registry
```shell
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${USER}_runner
```

5. Create your GCS bucket
```shell
OUTPUT_DIR=gs://v6e-demo-run #<your_GCS_folder_for_results>
gcloud storage buckets create ${OUTPUT_DIR}  --project ${PROJECT}
```

6. Specify your workload configs
```shell
export PROJECT=#<your_compute_project>
export ZONE=#<your_compute_zone>
export CLUSTER_NAME=v6e-demo #<your_cluster_name>
export OUTPUT_DIR=gs://v6e-demo/ #<your_GCS_folder_for_results>
```