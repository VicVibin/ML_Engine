# ML Library from Scratch

An ever evolving implementation of a machine learning library that can be used to build almost any structure of DNN such as a transformer,diffusion model, and reinforcement learning frameworks. This codebase was built from the ground up in C++, with the use of GPU accelerated CUDA kernels for any parallelizable computation, with automatic differentiation from GraphOperations. There are many more optimizations and architecture reward to make the API more usable across a wide range of implementations.

## Features

- **Custom Backpropagation Engine**: Full computational graph with automatic differentiation
- **GPU Acceleration**: CUDA-powered vector and matrix operations
- **Transformer Architecture**: Complete encoder-decoder implementation with:
  - Multi-head self-attention
  - Cross-attention mechanisms
  - Layer normalization
  - Feed-forward networks
  - Positional encoding

- **Diffusion Architecture**: complete DDPM structure with:
  - Time-step positional encoding
  - U_NET architecture
  - Langevin Sampling
  - Convolutional Blocks, etc

- **Reinforcement Learning Architecture**: complete DDPM structure with:
  - Q-Learning
  - PPO

- **Advanced Optimizers**: SGD, Adam and AdamW optimizer with momentum and bias correction
- **Text Processing**: Built-in tokenization and textual embeddings
- **Image Processing** - Uses OpenCV library to convert images to vectors/matrices
- **Flexible Inference**: TopK sampling with configurable parameters and KV Caching for improved attention efficiency (more methods to come)

## Architecture Overview

### Computational Graph
- **NodeBackProp**: Base computation unit with forward/backward/parameter updates passes and a built in memmory manager using free  
- **AdamParameter**: Special Node with Advanced optimizer with momentum and RMSprop with added correction
- **GraphOperations**: Complete set of differentiable operations stored in a graph. The memory is statically allocated at graph build and       computations are called using lambdas

## Quick Start

### Prerequisites
- CUDA-capable GPU
- CUDA toolkit
- C++17 compiler
- Standard dependencies:'engine', 'kernels', and 'debugging_utils` libraries

## Prebuilt use-cases are in use-cases folder to see some sample examples of the library for machine learning

## Key Operations

### Textual Embedding Operations
- `updateVocabulary()`: Updates the vocabulary of the embedder
- `encodeText()`: Encodes all texts of a given batch index saved in the internal keys. You can manually customize for your usecase
- `rencodeText()`: Encodes only the last text of a given batch index and start saved in the internal keys (recursive inference)
- `encodeBatch()`: Encodes a batch of texts
- `forward()`: Pushes the embedding matrix into the output of the given shared ptr node 
- `rforward`:  Pushes the embedding only the last vector of the embedding matrix into the output of the given shared ptr node 
- `encodeText()`: Encodes all texts of a given batch index saved in the internal keys
- `one_hot_forward()`: Pushes the embedding matrix of the one hot encoding of the embedded keys of the encoded texts
- `EmbeddingUpdate()`: Encodes a batch of texts

### Activation Functions
- `GraphOperations.RELU()->forward()`: Rectified Linear Unit
- `GraphOperations.SILU()->forward()`: Sigmoid Linear Unit
- `GraphOperations.SIGMOID()->forward()`: Sigmoid function
- `GraphOperations.TANH()->forward()`: Hyperbolic Tangent function
- `GraphOperations.GELU()->forward()`: Gaussian Error Linear Unit
- More will be added as needed since they are really easy to implement

### Special Graph Operation Function
# Note: these parameters can take in the node directly and sort to the top of all connected nodes then run
- `zero_grad()`: Zeros the gradient kernels of all nodes to avoid accumulation with
- `forward()`: Forward passes the topologically sorted nodes.
- `backward()`: Backward passes the topologically sorted nodes.
- `ParameterUpdate()`: Updates the AdamParameters.
- `accumulate()`: Accumulates the gradients for clipNorm.
- `clipNorm()`: clipnorms based on accumulated gradients. call after accumulate()
- `printParams()`: Prints the parameters of each AdamParameter.
- `PrintNodes()`: Prints the names of each Node within the graph
- `clear_graph()`: Clears the computational graph created.

### Convolutional Operations
- `Convolute2D->forward()->forward()`: Performs convolution using internals weights and biases that can be modified for specifics
- `Convolute2DT->foward()->forward()`: Performs transposed convolutions with similar internals as Convolute2D

## Performance Optimizations

- **GPU Streams**: Everything is preallocated on the GPU before computation and can easily flow through cuda
- **Memory Management**: Efficient graph clearing and memory reuse in go.clear_graph() which preserves parameters, clean_clear_graph() which frees the parameters also;
- **Asynchronous Execution**: Async optimizations soon to be added using graph tree
- **Custom CUDA Kernels**: Optimized all round operations

## Memory Requirements

- **Model Parameters**: depending on configuration
- **GPU Memory**: 2-8GB recommended for training
- **Dataset**: Flexible, processes text files of any size and images of any size and number. You can create your custom datahandler

## Troubleshooting

### Common Issues
- **CUDA Out of Memory**: SafeCudaMalloc catches an error when the kernel allocated is too large

### Debug Features
- Can check memory before and after clearing graph to find leaks
- Custom input parameters for weight initialization, gradient normalization and other factors in the header of debug.h

## Contributing

This implementation serves as an educational foundation for understanding how machine learning works under the hood and removing the black box, hidden nature of LLMs, Diffusion models, and other state of the art architectures. This learning experience shows that machine learning is not as complicated as it seems but requires understanding of data structures, mathematical operations and their derivatives and high performance computing. 
Key areas for extension:
- Multi-GPU training
- Cuda Streams for parallel data transfers and executions for independent kernels
- A new structure of GraphOperations that can run kernels on changed node dimensions for changing kernels without having to clear and reallocate every time
- Faster kernels and improved cache hits
- Better memory management and destructor for out of use allocations.. The current method I use is the most feasable method I have come up with does not clear useful memory

## License
 All usage and modifications should be made open source

---
**Note**: This implementation prioritizes educational clarity over production efficiency, though still highly efficient in some areas.
