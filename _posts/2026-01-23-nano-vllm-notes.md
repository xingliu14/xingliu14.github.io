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



