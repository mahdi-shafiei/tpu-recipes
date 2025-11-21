# Training Recipes for Ironwood TPU

The training recipes contained in this folder are optimized for Ironwood TPU. Here is a summary of the included recipes.

| <div style="width:100px;">Model ID</div>        | Number of chips | GBS | Sequence length | Precision | Step time (seconds) | Tokens/sec/chip |
|-----------------|--------------------|--------------|--------------------------|--------------------|-------------------|---------------------------|
| deepseek-v3     | 128                | 2048         | 4096                     | bf16               | 27.91552391       | 2,347.65                  |
| deepseek-v3     | 128                | 2048         | 4096                     | fp8_full           | 22.83807576       | 2,869.59                  |
| deepseek-v3     | 256                | 4096         | 4096                     | bf16               | 29.35336316       | 2,232.66                  |
| deepseek-v3     | 256                | 4096         | 4096                     | fp8_full           | 26.51635157       | 2,471.53                  |
| gpt-oss-120b    | 64                 | 1280         | 8192                     | bf16               | 17.77661018       | 9,216.61                  |
| gpt-oss-120b    | 256                | 5120         | 8192                     | bf16               | 18.77993546       | 8,724.20                  |
| llama3.1-405b   | 256                | 1536         | 8192                     | bf16               | 99.66454824       | 493.17                    |
| llama3.1-405b   | 256                | 1536         | 8192                     | fp8_full           | 65.02921753       | 755.84                    |
| llama3.1-70b    | 64                 | 256          | 8192                     | bf16               | 12.51527348       | 2,618.24                  |
| llama3.1-70b    | 64                 | 256          | 8192                     | fp8_full           | 8.908863386       | 3,678.13                  |
| llama3.1-70b    | 256                | 1024         | 8192                     | bf16               | 12.78735822       | 2,562.53                  |
| llama3.1-70b    | 256                | 1024         | 8192                     | fp8_full           | 9.384045601       | 3,491.88                  |
| llama3.1-70b    | 256                | 64           | 131072                   | bf16               | 34.72535706       | 943.63                    |
| llama3.1-70b    | 256                | 64           | 131072                   | fp8_full           | 31.47576637       | 1,041.05                  |
| qwen3-235b-a22b | 256                | 8192         | 4096                     | bf16               | 33.81617737       | 3,876.01                  |