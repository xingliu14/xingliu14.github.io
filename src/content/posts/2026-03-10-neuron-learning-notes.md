---
title: "Neuron Learning Notes"
published: true
---

## Trainium Memory Hierarchy
### Host Memory
- external, DRAM
- Linear Memory: multi-dimensional tensors must be stored in a flattened manner
- ~1TB, ~16GB/s
- NKI kernels don't provide API to access it

### Device Memory
- external, HMB
- Linear Memory
- ~50GB, 0.5TB/s per NC
- Input and output parameters of a NKI kernel must from and to Device Memory

### SBUF (State Buffer)
- internal, two-dimensional memory
- 128 partition
- ~25MB, ~10TB/s
- NKI kernel load data from HBM to SBUF, and result from SBUF to HBM

### PSUM (Partial Sum Buffer)
- internal, two-dimensional memory
- 128 partition
- ~2MB, 10TB/s
- Store MatMul result from TensorE
- PSUM can be accessed by ScalarE and VectorE, but for performance, leave it to TensorE.


## Data Representation
### Representing data in NKI
A `tensor` has a name, a shape that describes the number and size of each of its dimensions, an element data type (or “dtype”), and a description of the physical location of the underlying data on the NeuronCore.

For tensor placement, a physical location is defined by **memory type + offset/size metadata**.

- In **HBM (1D memory)**, you use one `(offset, size)`.
- In **SBUF/PSUM (2D memories)**, you use two `(offset, size)` pairs describing a rectangle:
  1. **Partition offset/size** (units: partitions) — which partitions are used.
  2. **In-partition offset/size** (units: bytes) — where data starts and how many bytes are used inside each selected partition.

For a `128×64` `float16` tensor mapped row-wise to SBUF:

- partition `(offset, size)` could be `(0, 128)` (128 rows across 128 partitions),
- in-partition `(offset, size)` could be `(1024, 128)` because each row has `64 × 2 = 128` bytes.


## Indexing
### Basic Tensor Indexing
NKI supports basic indexing of tensors using integers as indexes.
```
x = nl.ndarray((2, 128, 1024), dtype=nl.float32, buffer=nl.hbm)

# `x[1, :, :]` is the same as `x[1]`
assert x[1, :, :].shape == [128, 1024]

# Get a smaller view of the third dimension
assert x[1, :, 0:512].shape == [128, 512]

# `x[:, 1, 0:2]` returns a view of x with shape of [2, 2]
# [[x[0, 1, 0], x[0, 1 ,1]], [x[1, 1, 0], x[1, 1 ,1]]]
assert x[:, 1, 0:2].shape == [2, 2]
```


## Tiling
### Tile-based operations
SBUF and PSUM memories have 128 partitions, most APIs are limited to tiles with a first dimension (also called the “Partition Dimension”) no larger than 128 elements.

### Layout considerations
SBUF and PSUM all have first dimension as the partition dimention with size 128. The second dimension is the free dimension.

ow-major layouts place elements within each row in contiguous memory, and column-major layouts place elements within each column in contiguous memory.

The NeuronCore compute engines impose two layout constraints (LC):

- [Layout Constraint #1] For matrix multiplication operations, the contraction axis of both input tiles must be mapped to the Partition (P or P_DIM) dimension which is typically 128 for current hardware.
- [Layout Constraint #2] For operations that are not matrix multiplication operations, such as scalar or vector operations, the parallel axis should be mapped to the Partition (P or P_DIM) dimension. 

### Tile Size Considerations
Besides layout constraints, NeuronCore hardware further imposes three tile-size constraints (TC) in NKI:
- [Tile-Size Constraint#1] The P dimension size of a tile in both SBUF and PSUM must never exceed nki.tile_size.pmax == 128.
- [Tile-Size Constraint#2] For tiles in PSUM, the F dimension size must not exceed nki.tile_size.psum_fmax == 512.
- [TileSize Constraint#3] Matrix multiplication input tiles F dimension size must not exceed nki.tile_size.gemm_stationary_fmax == 128 on the left-hand side (LHS), or nki.tile_size.gemm_moving_fmax == 512 on the right-hand side (RHS).


## Introduction to Direct Memory Access (DMA)
Direct Memory Access (DMA) engines in Neuron enable efficient data movement between different memory types, primarily between the device memory (HBM) and on-chip SRAM buffers (SBUF).

### Basic DMA Capabilities
- Each DMA transfer starts with a DMA trigger from a NeuronCore and ends with a semaphore update from the DMA engine to signal the completion of transfer back to the NeuronCore
- DMA transfers can perform both copy and transpose transfers into SBUF.
- DMA transfers can move data in multiple directions: bidirectionally between HBM to SBUF, within HBM or within SBUF
- DMA engines also support scatter-gather operations, allowing a single transfer to gather data from multiple non-contiguous source buffers or scatter to multiple non-contiguous destination buffers.
- Bandwidth
	- 27.2 GB/s for NeuronCore-v2 and -v3
	- 38.4 GB/s for NeuronCore-v4

### DMA Triggers
DMA transfers can be triggered by any engine sequencer in the NeuronCore.

### DMA Queues
DMA transfers are submitted to DMA queues for the DMA Engines to consume. There are 16 DMA queues per DMA engine (ID 0-15).

### Performance Considerations
Higher the bytes per partition, higher the bandwidth. Each DMA engine are responsible for 8 partitions.


## Using Logical Neuron Cores (LNC)
The Neuron SDK supports running NKI kernels on multiple logical cores.










