# Nemotron-Nano-30B BATCHED PREFILL — per-op profiling map

- Device: GB10 sm_121 (GB10 Blackwell)
- S (prompt tokens): 2048
- Clean batched throughput: **74.3 tok/s** (13.46 ms/tok)
- Profiled pass wall (sync-bracketed, inflated): 28.465s; summed op time: 13.536s
- vLLM-on-GB10 reference: pp2048@d0=6395, @d8192=4993, @d32768=2734 tok/s
- Tensor-core peak assumed: 1000 TFLOP/s (bf16 dense)

| op | ms | % | calls | TFLOP/s | %peak |
|---|---:|---:|---:|---:|---:|
| moe_experts | 7137.65 | 52.7% | 69 | 0.790 | 0.08% |
| proj_gemm | 3764.71 | 27.8% | 70 | 1.121 | 0.11% |
| moe_shared | 1680.60 | 12.4% | 46 | 1.119 | 0.11% |
| ssm_scan | 673.69 | 5.0% | 23 | 0.147 | 0.01% |
| sdpa_prefill | 202.75 | 1.5% | 6 | 1.017 | 0.10% |
| moe_router | 32.37 | 0.2% | 23 | 1.001 | 0.10% |
| slice/cast | 31.36 | 0.2% | 140 | — | — |
| rms_norm | 12.01 | 0.1% | 52 | — | — |
| lm_head | 1.01 | 0.0% | 1 | 0.699 | 0.07% |

## Gap analysis
- `proj_gemm`/`lm_head` running far below %peak → projection GEMMs not at tensor-core roofline (f32 matmul, not bf16-MMA; dequant overhead separate).
- `ssm_scan` high % → the sequential-in-T `ssm_step_record` is the Mamba bottleneck → Milestone B: chunked/parallel SSD scan.
- `moe_experts`/`moe_shared` high % with many calls → per-token MoE gather loop → Milestone B: Q4 batched-expert GEMM over S.
- `host_conv` time is CPU (host-bridged) → move causal conv on-device for S-batched.
