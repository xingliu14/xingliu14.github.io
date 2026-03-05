---
title: "Nano-vLLM Study Notes"
published: true
---

I've been recently studying the nano-vLLM codebase and writing this article to document my notes and learnings for future reference.

### config.py
- model: path to the model weight and config files.
- max_num_batched_tokens: The maximum total number of tokens that can be processed in a **single batch across all sequences**.
- max_num_seqs: The maximum number of sequences (requests) that can be processed simultaneously in a batch.
- max_model_len: The maximum length (in tokens) of a single sequence.
- gpu_memory_utilization: The fraction of GPU memory to use (90% by default).
- tensor_parallel_size: tensor parallelism.
- enforce_eager: eager mode.
- hf_config: The HuggingFace model configuration object.
- eos: likely the end-of-sequence token ID.
- kvcache_block_size: The size of each block in the key-value cache. Must be multiple of 256.
- num_kvcache_blocks: likely the total number of KV-cache blocks to allocate.

### sampling_params.py
- temperature
- max_tokens
- ignore_eos

### bench.py
The script generates random token sequences, feeds them to a language model, and measures throughput.

### engine/sequence.py
This file implements a Sequence class that represents a single request/prompt being processed by the inference engine.

Attributes:
- seq_id: Unique id.
- status: waiting, running, finished.
- token_ids: full list of tokens.
- rest of the token ids, sampling params.

Properties:
- Multiple token ids, block counts.

### engine/block_manager.py
Block class for a single block:
- block_id
- ref_count: int (for prefix caching)
- hash: int (for prefix caching)
- token_ids: list[int]

BlockManager class for orchestration and prefix caching:  
Attributes:
- block_size: int
- blocks: list[Block]
- hash_to_block_id: dict[int, int]
- free_block_ids: deque[int]
- used_block_ids: deque[int]
- allocate: 

Methods:  
- allocate
- deallocate
- can_allocate
- can_append
- may_append
- compute_hash

### utils/context.py
Context to share globle state info across different part of the inference engine.

Attributes:
- is_prefill: whether it's in prefill phase
- cu_seqlens_q: Cumulative sequence lengths for queries for FlashAttention
- cu_seqlens_k: Cumulative sequence lengths for keys for FlashAttention
- max_seqlen_q: Maximum query sequence length in the current batch
- max_seqlen_k: Maximum key sequence length in the current batch
- slot_mapping: token to physical KV cache slots mapping
- context_lens: Number of cached tokens for each sequence
- block_tables: Maps each sequence to its allocated memory blocks

Glable Context instance, set and reset.


### engine/scheduler.py
The brain of continuous batching. It maintains queues for waiting and running sequences. It decides how many sequences from the waiting queue can be processed in the "prefill" phase, and which running sequences can proceed in the "decode" phase without exceeding the GPU's KV cache capacity. It also handles "preemption" (pausing a sequence if the engine runs out of memory)

Methods:
- is_finished(): Returns True if both queues are empty, meaning the engine has no more work to do.
- add(seq: Sequence): Accepts a new incoming request (Sequence) and puts it at the back of the waiting queue.
- Schedule(): This method is called before every forward pass of the model. It decides what the model will compute next. It returns a tuple containing the list of sequences to process and a boolean indicating if it's doing a Prefill (True) or Decode (False).
- preempt(): When a sequence is preempted due to lack of memory.
- postprocess(): After the model generates the next tokens (token_ids), this method applies them to the sequences.

Notes:  
- max_num_seqs limits how many requests (sequences) can be scheduled in one iteration, while max_num_batched_tokens limits total prompt tokens during prefill. Prefill is batched: in one scheduling step, it can schedule multiple waiting requests together, stopping when it hits sequence limit, token limit, or KV-cache allocation limit. Decode also schedules multiple running sequences (typically one token each), and is constrained by max_num_seqs plus KV-cache append capacity rather than max_num_batched_tokens.


### engine/llm_engine.py
It defines the LLMEngine class — the central orchestrator of the nano-vllm inference engine. It ties together model execution, scheduling, tokenization, and multi-process tensor parallelism into a unified interface for running LLM inference.  

#### Engine Initialization
- Config construction: Filters kwargs to only pass recognized fields to the Config dataclass
- Tensor parallelism setup: Rank0 has its own ModelRunner, worker processes created for rank 1 and above.
- Tokenizer
- Scheduler

#### exit
Sends an "exit" command to the model runner (which propagates to all worker processes), then waits for all child processes to terminate with join().

#### add_request
- Accepts a prompt as either a string (which gets tokenized) or pre-tokenized token IDs.
- Wraps it in a Sequence object (which tracks state like generated tokens, finish status, etc.).
- Adds it to the scheduler's queue.

#### step
- scheduler.schedule(): Selects a batch of sequences to run and determines whether this is a prefill or decode
- execute forward pass with model runner
- scheduler.postprocess(): Updates sequences with the newly generated tokens and marks finished sequences.
- Return: completed sequences and a throughput metric: positive for prefill, negative for decode.

#### generate
This is the user-facing method that runs end-to-end generation.  
- Input normalization: If a single SamplingParams is provided, it's broadcast to all prompts.
- Enqueuing: All prompts are added to the scheduler via add_request.
- Main loop: Repeatedly calls step() until all sequences finish.
- Throughput tracking (with tqdm)
- Output assembly


### engine/model_runner.py
ModelRunner is the workhorse that actually runs the LLM on a GPU. It handles model loading, KV cache allocation, input preparation for both prefill and decode phases, CUDA graph capture for decode acceleration, and tensor-parallel coordination across multiple GPUs via NCCL and shared memory.

#### Initialization
- NCCL init: establishes GPU-to-GPU communication
- Device binding
- Model creation: Instantiates model with the HuggingFace config, then loads pretrained weights via load_model()
- Sampler creation: Creates a Sampler that converts logits → token IDs using temperature-scaled softmax + Gumbel sampling.
- Warmup: Runs a dummy forward pass to trigger all lazy CUDA/Triton kernel compilation
- KV cache allocation: allocate_kv_cache() — uses the peak memory measurement to compute how many KV cache blocks fit in remaining GPU memory
- CUDA graph capture
- Tensor parallel setup: If world_size > 1, rank 0 creates a 1 MB shared memory segment (SharedMemory)

#### Tensor Parallel Communication
- The multi-GPU design uses a leader-follower pattern with shared memory + multiprocessing Events
- `write_shm`: Serializes a method name + arguments with pickle, writes a 4-byte length header + payload into shared memory, then signals all follower processes via their Events.
- `read_shm`: Blocks until signaled, deserializes the command, then clears the event.
- `call`: When rank 0 calls call("run", seqs, is_prefill), it first broadcasts the command to followers, then executes locally. Followers receive the command in loop() and execute the same method. This ensures all ranks execute the same operations in lock-step.
- `loop`: Non-zero ranks sit in this infinite loop, only breaking on an "exit" command.

#### Model Warmup  
Creates worst-case dummy sequences (maximum length, maximum batch) and runs a full prefill forward pass.  
- Triggers all lazy CUDA kernel compilation
- Records peak memory usage so allocate_kv_cache knows how much memory the model needs

#### KV Cache Allocation

#### Input Preparation
- prepare_prefill: This builds inputs compatible with Flash Attention's variable-length API
- prepare_decode: All tensors are created with pin_memory=True and transferred with .cuda(non_blocking=True) for overlapped CPU→GPU transfers.
- prepare_sample: Collects per-sequence temperature values into a tensor for the sampler.

#### Model Execution
- Prefill or eager or batch > 512: Eager execution
- Decode with batch ≤ 512: CUDA graph replay

#### CUDA Graph Capture


### layers/attention.py  
This file implements the attention layer for a minimal vLLM-style inference engine. It has three components:
- A Triton GPU kernel for writing KV data into a paged KV cache
- A Python wrapper that launches that kernel
- An Attention module that orchestrates KV caching and dispatches to FlashAttention for both prefill and decode phases

#### The Triton Kernel
A GPU kernel written in Triton that copies newly computed key/value vectors into the paged KV cache.

#### The Python Wrapper
- Shape extraction: key has shape [N, num_heads, head_dim] where N = total tokens in the batch.
- Stride assertions: Verifies the tensors are contiguous in the expected layout — the last dimension (head_dim) must be contiguous (stride 1), and the num_heads dimension must have stride head_dim. This ensures the kernel can treat num_heads × head_dim as a single flat vector of size D.
- Launch: [(N,)] launches N thread blocks — one per token.

#### The Attention Module
- Stores attention hyperparameters
- self.k_cache and self.v_cache are initialized as empty tensors. They get replaced later by the ModelRunner with properly sized paged cache buffers allocated by the block manager.


### layers/linear.py & layers/embed_head.py
# Tensor Parallelism in nano-vllm

## Overview

These two files implement **Megatron-LM style tensor parallelism (TP)**, splitting model weights across GPUs to serve large language models. The core pattern is:

> **Column-parallel (no comm) → local compute → Row-parallel (all-reduce)**

Each transformer sub-block (attention, MLP) requires only **one all-reduce**, minimizing communication overhead.

---

## Linear Layers (`linear.py`)

| Class | Sharding | Communication | Use Case |
|---|---|---|---|
| `ReplicatedLinear` | None (full copy) | None | Small layers not worth splitting |
| `ColumnParallelLinear` | Output dim (`dim=0`) | None | First projection (Q/K/V, gate, up) |
| `MergedColumnParallelLinear` | Output dim (multiple merged) | None | Fused gate + up proj |
| `QKVParallelLinear` | Output dim (Q/K/V merged) | None | Fused QKV with GQA support |
| `RowParallelLinear` | Input dim (`dim=1`) | `all_reduce` | Second projection (o_proj, down_proj) |

### Key Mechanisms

- **`weight_loader` hook**: Attached to each parameter; called during checkpoint loading to extract the correct shard per GPU via `.narrow()` / `.chunk()`.
- **Column-parallel**: Splits output dim → each GPU produces a slice of the output, no communication needed.
- **Row-parallel**: Splits input dim → each GPU computes a partial result, summed via `all_reduce`. Bias added only on rank 0 to avoid double-counting.

### QKV Weight Layout (per GPU)

For GQA (e.g., 8 Q heads, 2 KV heads, tp_size=2):

```
[Q_shard (4 heads) | K_shard (1 head) | V_shard (1 head)]
```

Loaded via `loaded_shard_id ∈ {"q", "k", "v"}` with computed offsets.

---

## Embedding & LM Head (`embed_head.py`)

| Class | Sharding | Communication | Purpose |
|---|---|---|---|
| `VocabParallelEmbedding` | Vocab rows | `all_reduce` | Input token embeddings |
| `ParallelLMHead` | Vocab rows | `gather` to rank 0 | Output logits for sampling |

### Embedding Forward

1. **Mask** tokens outside this rank's vocab range.
2. **Remap** global token IDs to local indices.
3. **Lookup** embeddings, zero out masked positions.
4. **All-reduce** to combine (each token is non-zero on exactly one rank).

### LM Head Forward

1. **Prefill optimization**: Extract only the last token per sequence (only those need logits).
2. **Linear projection** with this rank's vocab shard → partial logits.
3. **Gather to rank 0** and concatenate → only rank 0 has full logits for sampling. Other ranks return `None`.

---

## End-to-End Transformer Layer Flow

```
Input (identical on all GPUs)
  │
  ├─► QKVParallelLinear (column, no comm)
  │     └─► Attention (local)
  │           └─► RowParallelLinear/o_proj (all_reduce) ──► + residual
  │
  ├─► MergedColumnParallelLinear/gate+up (column, no comm)
  │     └─► SiLU(gate) * up (local)
  │           └─► RowParallelLinear/down_proj (all_reduce) ──► + residual
  │
  └─► (repeat N layers) ─► ParallelLMHead (gather to rank 0) ─► Sampling
```

---

## Design Principles

1. **Minimize communication**: Column + Row pairing ensures only one `all_reduce` per sub-block.
2. **Decoupled loading**: `weight_loader` hooks let each GPU extract its shard from full checkpoint weights independently.
3. **Asymmetric head output**: Embedding uses `all_reduce` (all GPUs need embeddings), LM head uses `gather` (only rank 0 samples), saving memory.


### 

