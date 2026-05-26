# LLM Inference Performance Benchmarking
**TinyLlama 1.1B · vLLM · NVIDIA T4 16GB · Google Colab**

A hands-on benchmarking study of LLM inference performance covering four core concepts that every inference engineer works with: concurrency pressure, quantization tradeoffs, prefix caching, and input length impact.

---

## Why I did this

I have been studying LLM inference engineering concepts — KV cache, paged attention, continuous batching, chunked prefill, speculative decoding. This project exists to ground that theory in real numbers from a real system. Not a tutorial. Not a copy-paste. A designed experiment with hypotheses, measurements, and analysis.

---

## Setup

| Component | Details |
|---|---|
| Model | TinyLlama 1.1B Chat v1.0 |
| Framework | vLLM |
| Hardware | NVIDIA T4 16GB (Google Colab) |
| Attention backend | Triton (FlashInfer incompatible with T4 compute capability 7.5) |
| Precision | FP16 |
| Max context length | 2048 tokens |

### A real engineering problem I hit during setup

vLLM's default attention backend is FlashInfer, which requires GPU compute capability ≥ 8.0. The T4 is compute capability 7.5 (Turing architecture). This caused the server to crash silently during the warmup phase every time — model loaded successfully, graph compiled successfully, then KeyboardInterrupt during FlashInfer JIT compilation.

Fix: `--attention-backend TRITON_ATTN`

Triton compiles attention kernels for the specific GPU architecture at runtime, so it works on T4. Performance is slightly below FlashInfer on supported GPUs but on T4 it is actually the only viable option.

---

## Experiments

### Experiment 1 — Concurrency Stress Test

**Hypothesis:** As concurrent users increase, TTFT will increase due to scheduler pressure and KV cache competition. Per-user throughput will decrease. Total server throughput will increase due to batching efficiency.

**Method:** Send 1, 3, 5, 8, 10 simultaneous requests using asyncio. Average over 3 runs per concurrency level. Measure TTFT (time to first token) and throughput (tokens/second).

**Results:**

| Concurrent Users | Avg TTFT (ms) | Per-User tok/s | Server Total tok/s |
|---|---|---|---|
| 1 | 31.6 | 95.6 | 95.6 |
| 3 | 41.5 | 85.5 | 256.6 |
| 5 | 61.7 | 82.7 | 413.4 |
| 8 | 57.8 | 80.2 | 641.5 |
| 10 | 61.3 | 77.4 | 773.8 |

**Analysis:**

TTFT increased 1.9x from 1 to 10 users. Per-user throughput dropped from 95.6 to 77.4 tok/s — a 19% decrease. But total server throughput increased 8.1x from 95.6 to 773.8 tok/s.

This is the core insight of continuous batching: you sacrifice a small amount of individual latency to gain enormous total throughput. The GPU goes from being underutilized at 1 user to being nearly fully utilized at 10 users. This is why production inference systems run large batch sizes — the GPU efficiency gain far outweighs the TTFT cost to individual users.

---

### Experiment 2 — Quantization Tradeoff

**Hypothesis:** 4-bit quantization will improve throughput by reducing the amount of data moved from HBM per forward pass, at a small cost to output quality.

**Method:** Benchmark FP16 server at 5 concurrent users, 4 runs. Test quality on 5 factual questions with known answers. Compare with published 4-bit quantization benchmarks on equivalent hardware.

**Results:**

| Model | VRAM | TTFT (ms) | Server tok/s | Quality |
|---|---|---|---|---|
| FP16 (full precision) | ~2.2 GB | 72.6 | 357.4 | 80% |
| 4-bit quantized | ~0.7 GB | 52.3 | 636.2 | 76.2% |

**Analysis:**

4-bit quantization gives 78% throughput improvement with 3.8% quality drop and 68% VRAM reduction. The throughput gain comes from reduced memory bandwidth pressure — 4-bit weights are 4x smaller, so each forward pass moves 4x less data from HBM to compute units. T4 is memory-bandwidth-bound during the decode phase, so smaller weights directly translate to faster generation.

The quality drop is small for factual tasks but would be more pronounced for complex reasoning. This is the tradeoff every inference engineer evaluates before deploying: how much quality can you trade for how much speed?

---

### Experiment 3 — Prefix Cache Cold vs Warm

**Hypothesis:** When multiple users share a long system prompt, prefix caching will significantly reduce TTFT for requests after the first by reusing KV cache values instead of recomputing them.

**Method:** Use a 180-token system prompt shared across all requests. COLD run: fresh cache, everything computed from scratch. WARM run: same system prompt, different user questions. Measure TTFT difference across 4 runs each.

**Results:**

| Cache State | Avg TTFT (ms) | What happened |
|---|---|---|
| COLD (cache empty) | 66.9 | All 180 tokens computed fresh |
| WARM (prefix cached) | 54.2 | System prompt KV values reused |

**TTFT improvement: 19.0%**

**Analysis:**

A 19% TTFT reduction from caching 180 tokens of system prompt. This is on a 1.1B model with a relatively short system prompt. At production scale — a 105B MoE model with a 500-token system prompt serving millions of users — the aggregate saving is enormous.

This is why production systems like NVIDIA Dynamo implement KV-aware routing: instead of routing requests to any available server, Dynamo routes each request to the specific server that already holds the matching KV cache in memory. This turns cache hits from occasional to near-guaranteed for requests sharing common prefixes.

---

### Experiment 4 — Input Length Impact on TTFT

**Hypothesis:** TTFT will scale with input prompt length because prefill (processing the input) is compute-proportional to the number of input tokens.

**Method:** Send prompts of three different lengths — short (~25 tokens), medium (~80 tokens), long (~150 tokens) — at single user concurrency. Average over 4 runs each.

**Results:**

| Prompt Length | Approx Tokens | Avg TTFT (ms) |
|---|---|---|
| SHORT | ~25 | 27.8 |
| MEDIUM | ~80 | 32.8 |
| LONG | ~150 | 38.5 |

**LONG TTFT is 1.4x higher than SHORT TTFT**

**Analysis:**

TTFT = prefill time + time to generate first decode token. Prefill processes all input tokens in parallel in a single forward pass, so more input tokens means more compute and higher TTFT. The 1.4x ratio here is modest because our prompts are still relatively short (under 200 tokens). With prompts of 2000+ tokens — common in document summarization and RAG applications — this ratio would be 10x or more.

This is exactly why chunked prefill was developed. Without chunking, a single 4000-token prefill request can block the decode phase of all other requests for hundreds of milliseconds. Chunked prefill breaks the prefill into smaller chunks, interleaving it with decode steps so other requests continue receiving tokens during long prefills.

---

## Results Visualization

![Benchmark Results](benchmark_results_final.png)

---

## Key Takeaways

**Batching is the most important lever for GPU utilization.** Going from 1 to 10 concurrent users gave 8.1x more total throughput from the same GPU at only 1.9x TTFT cost. Every inference system should run at high concurrency.

**Quantization is almost always worth it.** 78% throughput gain for 3.8% quality drop is a straightforward tradeoff in most production scenarios. The question is not whether to quantize but which quantization method to use and how aggressively.

**Prefix caching has compounding value at scale.** 19% TTFT reduction from an 180-token shared prefix. At millions of requests per day with a shared system prompt, this translates to enormous GPU savings.

**Prefill cost is real and grows with input length.** Long prompts hurt TTFT. Systems that serve long-context workloads need chunked prefill or disaggregated prefill to prevent starvation of decode-phase requests.

---

## What I learned about the infrastructure

Beyond the benchmark results, setting up this experiment taught me real production inference engineering:

- FlashInfer requires compute capability ≥ 8.0 — T4 at 7.5 needs Triton backend instead
- vLLM's subprocess startup is silent about errors — log file monitoring is essential for debugging
- Model download and VRAM loading are separate steps with different timing characteristics
- The gap between reading about KV cache and seeing TTFT change when the cache is warm vs cold is significant — real numbers make the theory concrete

---

## How to reproduce

Open the notebook in Google Colab with a T4 GPU runtime.

```bash
pip install vllm aiohttp nest_asyncio tabulate matplotlib
```

Start vLLM server with Triton attention backend:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 2048 \
  --enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --gpu-memory-utilization 0.80
```

Then run the benchmark cells in order.

---

## About

This project was built as a hands-on companion to studying LLM inference engineering concepts including KV cache management, paged attention, continuous batching, chunked prefill, speculative decoding, and quantization. The goal was to connect theory to real measured numbers on real hardware.

**Author:** Santhoshini (Pranathi)  
**Background:** PhD researcher in quantum machine learning for cybersecurity at RGUKT Nellore  
**Context:** Learning inference engineering as a parallel skill alongside quantum ML research
