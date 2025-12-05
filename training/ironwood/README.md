# Training Recipes for Ironwood TPU

The training recipes contained in this folder are optimized for Ironwood TPU. Here is a summary of the included recipes.

| <div style="width:100px;">Model ID</div> | Number of chips | GBS | Sequence length | Precision | Step time (seconds) | TFLOPs/sec/chip | Tokens/sec/chip |
|-----------------|--------------------|--------------|--------------------------|--------------------|-------------|--------------|-----------------------|
| deepseek-v3     | 128                | 2048         | 4096                     | bf16               | 27.91       | 587.91       | 2,347.65              |
| deepseek-v3     | 128                | 2048         | 4096                     | fp8_full           | 22.83       | 718.57       | 2,869.59              |
| deepseek-v3     | 256                | 4096         | 4096                     | bf16               | 29.35       | 559.18       | 2,232.66              |
| deepseek-v3     | 256                | 4096         | 4096                     | fp8_full           | 26.51       | 618.95       | 2,471.53              |
| gpt-oss-120b    | 64                 | 1280         | 8192                     | bf16               | 17.77       | 317.63       | 9,216.61              |
| gpt-oss-120b    | 256                | 5120         | 8192                     | bf16               | 18.77       | 300.64       | 8,724.20              |
| llama3.1-405b   | 256                | 1536         | 8192                     | bf16               | 99.66       | 1,244.67     | 493.17                |
| llama3.1-405b   | 256                | 1536         | 8192                     | fp8_full           | 65.02       | 1,907.81     | 755.84                |
| llama3.1-70b    | 64                 | 256          | 8192                     | bf16               | 12.51       | 1,176.27     | 2,618.24              |
| llama3.1-70b    | 64                 | 256          | 8192                     | fp8_full           | 8.90        | 1,652.29     | 3,678.13              |
| llama3.1-70b    | 256                | 1024         | 8192                     | bf16               | 12.78       | 1,151.14     | 2,562.53              |
| llama3.1-70b    | 256                | 1024         | 8192                     | fp8_full           | 9.38        | 1,568.72     | 3,491.88              |
| llama3.1-70b    | 256                | 64           | 131072                   | bf16               | 34.72       | 879.83       | 943.63                |
| llama3.1-70b    | 256                | 64           | 131072                   | fp8_full           | 31.47       | 970.72       | 1,041.05              |
| qwen3-235b-a22b | 256                | 8192         | 4096                     | bf16               | 33.81       | 574.87       | 3,876.01              |
