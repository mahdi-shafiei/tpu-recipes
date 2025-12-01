## Initialization

> **_NOTE:_** We recommend using Python 3.12 for MaxText workloads, as it is our primary supported version. Other Python versions may encounter compatibility issues.

1. Run the following commands to initialize the project and zone.
```shell
export PROJECT=#<your_project_id>
export ZONE=#<zone>
gcloud config set project $PROJECT
gcloud config set compute/zone $ZONE
```

2. Install the prerequisites for XPK by following the [instructions](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#prerequisites). Also ensure you have the proper [GCP permissions](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#installation).

* In order to run the tpu-recipes as-is, run the `git clone` command from your home (~/) directory:
```shell
# tpu-recipes requiring XPK will look for it in the home directory
cd ~/
git clone https://github.com/google/xpk.git
```

3. Run the rest of these commands from the cloned XPK directory:

```shell
cd xpk # Should be equivalent to cd ~/xpk
git checkout v0.14.3
```

## GKE Cluster Creation 
Trillium GKE clusters can be [created](https://cloud.google.com/tpu/docs/v6e-intro#create_an_xpk_cluster_with_multi-nic_support) and 
[deleted](https://cloud.google.com/tpu/docs/v6e-intro#delete_xpk_cluster) by following the public GCP documentation.

> **_NOTE:_** in order to run the training and microbenchmarks tpu-recipes, you should not need to run sections outside of
`Create an XPK cluster with multi-NIC support` when creating your cluster. You can skip the following sections like `Framework setup`.
