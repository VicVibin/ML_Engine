# ML Library from Scratch

A partially complete implementation of a machine learning library that can be used to build almost any structure of DNN such as a transformer and a diffusion model. This codebase was built from the ground up in C++, with the use of GPU accelerated CUDA kernels for any parallelizable computation, with automatic differentiation from GraphOperations.

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

- **Advanced Optimizers**: Adam and AdamW optimizer with momentum and bias correction
- **Text Processing**: Built-in tokenization and textual embeddings
- **Image Processing** - Uses OpenCV library to convert images to vectors/matrices
- **Flexible Inference**: TopK sampling with configurable parameters and KV Caching for improved attention efficiency (more    methods to come)

## Architecture Overview

The implementation follows the standard Transformer architecture:

```
Encoder: Input → Attention → Add & Norm → FFN → Add & Norm
Decoder: Input → Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm → Linear → Softmax
```

## Core Components

### Computational Graph
- **NodeBackProp**: Base computation unit with forward/backward passes
- **AdamParameter**: Special Node with Advanced optimizer with momentum and RMSprop with added correction
- **GraphOperations**: Complete set of differentiable operations stored in a graph

### Attention Mechanisms
- **Attention**: Single attention head with Q, K, V projections
- **MultiHeadAttention**: Soon to be added MultiHead Attention with row-wise concatenation
- **Cross-Attention**: Encoder-decoder attention for sequence-to-sequence tasks

## Quick Start

### Prerequisites
- CUDA-capable GPU
- CUDA toolkit
- C++17 compiler
- Custom dependencies: `Dataloader`, `Engine`, `Debugging_utils`, `kernels` libraries

### Basic Usage for Transformer using graphOperations

```cpp
#include "includes/engine.h"
#include "includes/dataloader.h"


class TransformerBlock
{
    private:
    GraphOperations &go;
    public:
    const int embed_dim;
    const int hidden;
    Linear *q, *k, *v, *out;
    const str type;
    TransformerBlock(GraphOperations &go_ref, const int embed_dim, const int t_hidden): go(go_ref), 
    embed_dim(embed_dim), type(type), hidden(t_hidden)
    {
        q = new Linear(go, embed_dim, t_hidden, "Transformer Q");
        k = new Linear(go, embed_dim, t_hidden, "Transformer K");
        v = new Linear(go, embed_dim, t_hidden, "Transformer V");
        out = new Linear(go, t_hidden, embed_dim, "Transformer Output");
    }

    void save(std::ofstream& f) const
    {
        q->save(f);
        k->save(f);
        v->save(f);
        out->save(f);
    }

    void load(std::ifstream& f)
    {
        q->load(f);
        k->load(f);
        v->load(f);
        out->load(f);
    }
    
    graph forward(const graph&X, const bool mask = false)
    {
        auto Q = q->forward(X);
        auto K = k->forward(X);
        auto V = v->forward(X);
        auto scores = go.BMMABT(Q, K);
        auto scaled_scores = go.Scale(scores, 1.0f / sqrtf((float)(hidden)));
        auto attn_weights = mask ? go.SOFTMASK(scaled_scores,1): go.SOFTMAX(scaled_scores,1);
        auto attn_output = go.BMM(attn_weights, V);
        auto output_node = out->forward(attn_output);
        output_node->op_name = type + "Transformer Block Output";
        auto skip = go.LayerNorm(go.Add(output_node, X));
        return skip;
    }

    graph cross_forward(const graph& X, const graph& Y)
    {
        /*
        @author Cross attention implementation, query from X, key and value from Y
        */
        auto Q = q->forward(X);
        auto K = k->forward(Y);
        auto V = v->forward(Y);
        auto scores = go.BMMABT(Q, K);
        auto scaled_scores = go.Scale(scores,1.0f/sqrtf((float)(embed_dim)));
        auto attn_weights =  go.SOFTMASK(scaled_scores,1);
        auto attn_output = go.BMM(attn_weights, V); 
        auto output_node = out->forward(attn_output);
        output_node->op_name = " Cross-Attention Transformer Block Output";
        auto skip = go.LayerNorm(go.Add(output_node, X));
        return skip;
    }

    graph cached_forward(const graph& X_new, KVCache&cache, const int start_idx, bool mask = true)
    { 
    auto K_new = k->forward(X_new);    
    auto V_new = v->forward(X_new);

    // Handling Cache Update ======== // Can add MemcpyAsync for SpeedUp;
    K_new->forward();
    V_new->forward();

    int offset = cache.current_len * cache.hidden;
    cudaMemcpy(cache.K+offset,K_new->output,cache.hidden*sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy(cache.V+offset,V_new->output,cache.hidden*sizeof(float),cudaMemcpyDeviceToDevice);
    cache.current_len += K_new->dim[2];

    K_new->clear();
    V_new->clear();

    graph K_full = std::make_shared<NodeBackProp>("KV_K",1,1,cache.current_len,cache.hidden,0);
    graph V_full = std::make_shared<NodeBackProp>("KV_V",1,1,cache.current_len,cache.hidden,0);

    K_full->output = cache.K;
    V_full->output = cache.V;

    // ============================= // 
    auto pos    = go.MatrixPositionalEncoding(X_new, start_idx);
    auto Q_new  = q->forward(pos);
    auto scores = go.BMMABT(Q_new, K_full);
    auto scaled  = go.Scale(scores, 1.0f / sqrtf((float)hidden));
    auto weights = mask ? go.SOFTMASK(scaled, 1) : go.SOFTMAX(scaled, 1);
    auto attn_out = go.BMM(weights, V_full);
    auto output_node = out->forward(attn_out);
    auto skip = go.LayerNorm(go.Add(output_node, X_new));
    return skip;

    }

    graph cached_cross_forward(const graph& X_new, KVCache& cache)
    {
    auto Q_new = q->forward(X_new);   
    graph K_full = std::make_shared<NodeBackProp>("KV_K",1,1,cache.current_len,cache.hidden,0);
    graph V_full = std::make_shared<NodeBackProp>("KV_V",1,1,cache.current_len,cache.hidden,0);
    K_full->output = cache.K;
    V_full->output = cache.V;
    auto scores = go.BMMABT(Q_new, K_full);
    auto scaled  = go.Scale(scores, 1.0f / sqrtf((float)hidden));
    auto weights = go.SOFTMASK(scaled, 1);
    auto attn_out = go.BMM(weights, V_full);
    auto output_node = out->forward(attn_out);
    auto skip = go.LayerNorm(go.Add(output_node, X_new));
    return skip;
    }
    
};

```

### Basic Usage for Embedder 

```cpp
#include "includes/engine.h"
#include "includes/dataloader.h"

int main(){
    const int MAX_VOCAB_SIZE = 25000; // Maximum vocabulary size of the embedder
    const int MAX_BATCH_SIZE = 16; // Maximum batch size of the embedder
    const int MAX_CONTEXT_LEN = 16384; // Maximum context length of the embedder
    const int EMBED_DIM = 128; // Embedding dimension of the transformer model (n x d_model)
    const int context_len = 32; // Context length used for training | <= MAX_CONTEXT_LEN
    const int HIDDEN_DIM = 256; // Size of the hidden dimension as used by custom LLM implementation

    GraphOperations go; // Declare graph Operations class
    TextualEmbedding embedder(EMBED_DIM,MAX_BATCH_SIZE, MAX_CONTEXT_LEN, MAX_VOCAB_SIZE); //Declare and initialize embedder
    Text Db = LoadStory("text_training_data"); // Loads all the text data from folder
    Text Database = read_words(Db, 0, Db.size());
    embedder.updateVocabulary(Database); // Always call update vocabulary to add the vectorize the database
    printf("Total Vocabulary Size: %i | Total Word Count: %i \n", embedder.Vocabulary.size(), Database.size());
    // Assume you have a created class structure with training definition
    LLM model(go, embedder, Database, MAX_BATCH_SIZE, context_len, HIDDEN_DIM);
      
    model.train(20000, 100);
    model.generate({"so","sherlock", "holmes", "said"}, 30);
}

```


### Basic Usage for Diffusion assuming Custom U-NET and Langevin Sampler 

```cpp
#include "includes/engine.h"
#include "includes/dataloader.h"


int main()
{
    GraphOperations go; // Declaring Graph Operations
    const int T = 1000, init_depth = 64, t_hidden = 128, img_size = 64, epochs = 30000; // Initializing parameters
    int t;

    std::random_device rd;
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, T-1); // Initializing random t values for the diffusion model


    auto base = Bi2n("image_training_data", 1, img_size, img_size); // Loads a batch of images and converts to a node 
    StandardNorm(base); // Normalizes using (X/255 - 0.5) / 0.5
    auto input  = go.like(base, "Input Image"), target = go.like(base,"Target Image"); // Creates node structures similar to base

    U_NET model(go,input,target,nullptr,3,3,init_depth,t_hidden); // Custom built U-NET model, you would need to build your own
    model.build_train(); // builds the computational graph

    const bool sample_only = false;
    if (!sample_only)
    {
    for(int epoch = 0; epoch < epochs; ++epoch)
    {
        model.t = t = dist(gen); 
        prepare(base, input, target, t, T, ((uint64_t)rd() << 32) | rd());
        /*Copies the base values to the input kernel, fills the target with gaussian noise and uses DDPM cosine scheduling algorithm to fill the input with the needed 
        */
        model.train();
        if (epoch % 10 == 0) {printf("Epoch %i, Loss: %f at t = %i \n", epoch+1, model.loss, t);}
    }
    model.save("unet_model.bin");
    go.clear_graph();
    }
    else
    {
        model.load("unet_model.bin");
    }

    
    model.build_inference(input);
    auto test = go.like(base, "Test Image");
    Noise(test);
    Sampling<U_NET> sampler(model, test, T-1,T);
    sampler.loop(T-1, 1);
    BPrintImage(test);
    BPrintImage(base, 512, 512);
    sampler.display(512,512);
    return 0;
}

```

## Key Operations

### Embedding Operations
- `updateVocabulary()`: Updates the vocabulary of the embedder
- `encodeText()`: Encodes all texts of a given batch index saved in the internal keys
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
- More will be added as needed since they are really easy to implement

### Special Graph Operation Function
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
- **Memory Management**: Efficient graph clearing and memory reuse in go.clear_graph();
- **Asynchronous Execution**: Async optimizations soon to be added
- **Custom CUDA Kernels**: Optimized all round operations

## Example Output

```
Epoch: 1000
Epoch 0 Loss: 2.45
Generating Text
the quick brown fox jumps over the lazy dog and runs
_____________________
```

## Memory Requirements

- **Model Parameters**: depending on configuration
- **GPU Memory**: 2-8GB recommended for training
- **Dataset**: Flexible, processes text files of any size and images of any size and number

## Troubleshooting

### Common Issues
- **CUDA Out of Memory**: SafeCudaMalloc catches an error 
- **No actual training**: If custom graph are built always call go.nodes = topological_sort(last_node), then train or else no the graph built will not be tracked
- **Slow Training**: Working on it, the LLM is on par with pytorch with speed but Error checking and synchronizations everywhere is slowing down training for Convolutional Kernels

### Debug Features
- Can check memory before and after clearing graph to find leaks
- Custom input parameters for weight initialization, gradient normalization and other factors in the header of debug.h

## Contributing

This implementation serves as an educational foundation for understanding how machine learning works under the hood and removing the black box, hidden nature of LLMs, Diffusion models, and other state of the art architectures. This learning experience shows that machine learning is not as complicated as it seems but requires understanding of data structures, mathematical operations and their derivatives and high performance computing. 
Key areas for extension:
- Learning rate scheduling
- Multi-GPU training
- Cuda Streams for parallel data transfers and executions for independent kernels
- EmbeddingUpdate fixing to update vocabulary by differentiating keys for update
- Faster kernels and improved cache hits
- Better memory management and destructor for out of use allocations

## License
 All usage and modifications should be made open source

---

**Note**: This implementation prioritizes educational clarity over production efficiency, though still highly efficient in some areas.
