# Transformer Inference Engine

A transformer inference serving engine built from scratch — implementing
KV-cache reuse, continuous batching scheduling, and streaming token
generation with full benchmark suite and performance analysis.

Built to understand how production inference systems like vLLM work
at the architectural level — not by using them, but by building the
core mechanisms independently.

---

## What This Is

Most ML projects train models. This project **serves** them.

The gap between a trained transformer and a system that efficiently
serves that transformer to many users simultaneously is enormous.
This project fills that gap by implementing the three core components
that make production inference engines fast:

- **KV-Cache** — eliminates redundant recomputation during decode
- **Continuous Batching** — eliminates idle GPU slots across requests  
- **Streaming** — delivers tokens to users as they are generated

---

## Results

| Metric | Result |
|--------|--------|
| Decode latency reduction (KV cache) | **36.3%** avg TPOT reduction |
| Throughput improvement (continuous vs static batching) | **2.74x** measured, **~4x** theoretical upper bound |
| GPU utilization — static batching | **49.7%** (half the GPU wasted) |
| GPU utilization — continuous batching | **100%** |
| Priority scheduling — HIGH vs LOW wait time | **61 ms vs 719 ms** |
| KV cache memory at 32K context (real model scale) | **32 GB per request** |

All results measured on **NVIDIA T4 GPU** (Google Colab).

---

## Architecture
transformer-inference-engine/
│
├── minimal_tranformer_inference_engine.ipynb    ← complete implementation, one notebook
│
└── plots/
├── plot_7_1_ttft_vs_prompt.png
├── plot_7_2_kvcache_latency.png
├── plot_7_3_memory_wall.png
├── plot_7_4_throughput.png
├── plot_7_5_utilization_timeline.png
├── plot_7_6_queue_depth.png
└── plot_7_7_priority.png
---

## Notebook Structure

The entire project lives in one Colab notebook with clearly labeled sections.

### Section 0 — Setup
Device detection, imports.

### Section 1 — Transformer Model
Multi-head attention with causal masking, MLP, transformer block,
full ToyTransformer. Tensor shapes: `[batch, heads, seq, head_dim]`.
KV cache integration directly inside the attention forward pass.

### Section 2 — KV Cache System
Per-request `KVCache` objects with exact byte-level memory tracking.
`KVCacheManager` handles all concurrent request caches together
and enforces a GPU memory budget via memory-aware admission control.

### Section 3 — Prefill and Decode Runtime
Two fundamentally different execution paths correctly separated.
`prefill()` measures **TTFT** (Time To First Token).
`decode_step()` measures **TPOT** (Time Per Output Token).

### Section 4 — Scheduler
`Request` dataclass tracking full request lifecycle.
`PriorityRequestQueue` — min-heap with priority + FIFO tie-breaking.
`StaticBatchingScheduler` — correct baseline implementation.
`ContinuousBatchingScheduler` — four-phase execution loop
(prefill → decode → retire → admit) with memory-aware slot management.

### Section 5 — Streaming Token Generation
Python generator yielding one token per decode step.
Simulates ChatGPT-style word-by-word output delivery.

### Section 6 — Benchmarks
Seven benchmarks producing all performance claims:
- TTFT vs prompt length
- TPOT with cache vs without cache
- KV cache memory scaling and memory wall
- Static vs continuous batching throughput
- GPU slot utilization timeline
- Queue depth vs system concurrency
- Priority scheduling fairness

### Section 7 — Plots
Seven publication-quality plots, one per benchmark.

---

## Key Concepts

### Why KV Cache

During decode, the model generates one token at a time. Without
caching, every step recomputes Key and Value vectors for ALL previous
tokens from scratch — O(N) redundant work per step, O(N²) total.

With KV cache, Key and Value vectors are computed once during prefill
and stored. Each decode step reads from cache and appends one new
entry. Compute per step stays O(1) regardless of sequence length.

**Measured result: 36.3% average TPOT reduction at sequences of
50–500 tokens.**

### Why Continuous Batching

Static batching processes one fixed batch at a time, waiting for ALL
requests to finish before accepting new ones. When requests have
mixed output lengths — some finish in 5 tokens, others need 150 —
the finished slots sit completely idle. Measured GPU utilization:
**49.7%** — half the GPU wasted.

Continuous batching refills slots the moment any request finishes.
The scheduler runs a four-phase loop every decode step. Phase 1:
prefill newly admitted requests. Phase 2: decode all active requests.
Phase 3: retire finished requests and free their caches. Phase 4:
admit new requests from the priority queue into freed slots.

GPU slots are never idle as long as the queue has requests.
**Measured GPU utilization: 100%.**

### Why TTFT and TPOT Are Separate Metrics

**TTFT** (Time To First Token) — how long before the user sees
anything. Dominated by prefill length. Grows with prompt size.
Optimized by efficient prefill execution.

**TPOT** (Time Per Output Token) — how long between each subsequent
token. Dominated by decode stage. Should stay flat with KV cache.
Optimized by cache reuse and efficient decode scheduling.

Total latency = TTFT + (output_tokens × TPOT).
Knowing which to optimize requires knowing which dominates your
workload — long prompts vs long outputs.

### The Memory Wall

KV cache memory grows linearly with context length. At real model
scale (d_model=4096, 32 heads, 32 layers):

| Context Length | KV Cache Memory |
|----------------|-----------------|
| 1K tokens | 1 GB per request |
| 4K tokens | 4 GB per request |
| 32K tokens | **32 GB per request** |

A T4 GPU has 16 GB total. At 32K context, one request's cache
exceeds the entire GPU memory. This is the memory wall.

Paged attention (vLLM's core innovation) solves this by storing KV
cache in fixed-size non-contiguous pages — like virtual memory in
an operating system — eliminating fragmentation and enabling
prefix sharing across requests.

---

## Honest Limitations

This is a pedagogically correct implementation, not a production system.

| What this engine does | What vLLM/TRT-LLM does |
|----------------------|------------------------|
| Contiguous KV cache layout | Paged KV cache (PagedAttention) |
| PyTorch high-level ops | Custom CUDA / Triton kernels |
| Single GPU | Multi-GPU tensor parallelism |
| Greedy argmax sampling | Temperature, top-p, top-k, penalties |
| Integer token lists | Full HuggingFace tokenizer integration |
| Simulated throughput claims | Real production workload benchmarks |

The architectural concepts — prefill/decode separation, KV cache
update mechanism, continuous batching scheduler, memory-aware
admission control, TTFT/TPOT measurement — are identical to
production systems. The implementation details are simplified
to keep the code readable and the concepts clear.

---

## How to Run

Open `inference_engine.ipynb` in Google Colab with a T4 GPU runtime.

Runtime → Change runtime type → T4 GPU

Run all cells in order. Each section builds on the previous.
Section 6 benchmarks take approximately 5–8 minutes to complete.

---

## Concepts This Project Demonstrates

- Transformer inference execution (prefill vs decode)
- KV cache design and memory management
- GPU utilization analysis (arithmetic intensity, memory bandwidth)
- Serving system scheduler design (static vs continuous batching)
- Production inference metrics (TTFT, TPOT, throughput, queue depth)
- Memory-aware admission control
- Priority scheduling and fairness tradeoffs
- Performance benchmarking methodology

---

## Related Production Systems

- **vLLM** — PagedAttention, continuous batching at scale
- **TensorRT-LLM** — NVIDIA's production inference stack
- **DeepSpeed-FastGen** — Microsoft's inference optimization
- **FlashAttention** — IO-aware attention kernel this engine would
  need at production scale

---

*Built on NVIDIA T4 GPU. PyTorch 2.x. Python 3.10.*
