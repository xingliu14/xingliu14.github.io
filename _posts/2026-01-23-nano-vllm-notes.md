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
- ref_count: int
- hash: int
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
