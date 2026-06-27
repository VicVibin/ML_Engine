#pragma once
#include "engine.h"

using Text = std::vector<str>;
using BatchText = std::vector<Text>;

struct BatchTexts
{
    BatchText encoder;
    BatchText decoder;
    BatchText target;
    BatchTexts(int batch_size,int clen):encoder(batch_size,Text(clen+1)),decoder(batch_size,Text(clen+1)),target(batch_size,Text(clen+1)){}
};

struct KVCache 
{
    float* K;   // [max_len, hidden]
    float* V;   // [max_len, hidden]
    int current_len = 0;
    int max_len;
    int hidden;
    KVCache(int max_len, int hidden);
    void init(int max_len, int hidden);
    void free();

};

class TextualEmbedding
{
public:
    /*
    Parameters:
    1.) embed_dim: dimension of each word embedding
    2.) batch_size: maximum batch size for input text
    3.) max_c_len: maximum context length (number of words in each input text)
    4.) max_vocab_size: maximum vocabulary size (number of unique words in the vocabulary
    */
    std::random_device rd;
    float* EmbedSpace;      // GPU embedding memory space [MAX_VOCAB_SIZE x embed_dim]
    int* encoder_keys;      // GPU word index mapping to the embedding: [MAX_BATCH_SIZE x MAX_CONTEXT_LEN], changes each epoch
    int* decoder_keys;      // GPU word index mapping to the embedding: [MAX_BATCH_SIZE x MAX_CONTEXT_LEN], changes each epoch
    int* target_keys;       // GPU word index mapping to the embedding: [MAX_BATCH_SIZE x MAX_CONTEXT_LEN], changes each epoch
    Text input_text;        // Current input text batch
    std::unordered_map<str, int> WordSpace; // CPU word to key mapping
    std::unordered_map<int, str> KeySpace;  // CPU key to word mapping
    std::unordered_set<str> Vocabulary;     // Total set of all unique words which the index is the key
    const int MAX_BATCH_SIZE, MAX_CONTEXT_LEN, MAX_VOCAB_SIZE;
    const int embed_dim;
    const int tpb = THREADSPERBLOCK; 
    
private:
    str preprocessWord(const str& word);
public:
    TextualEmbedding(const int embed_dim, const int batch_size=128, const int max_c_len=64, const int max_vocab_size=10000, const bool for_LLM=false);
    ~TextualEmbedding();
    void updateVocabulary(const Text& texts);
    void encodeText(const Text& texts, const str key, const int batch_idx = 0);
    void rencodeText(const Text& texts,const str key, const int batch_idx = 0, const int start_idx = 0);
    void encodeBatch(const BatchText&  batch_texts, const str key);
    void forward(const graph& X, const str key);
    void rforward(const graph&X, const str key, const int start_idx = 0);
    void EmbeddingUpdate(const graph&X, const str key);
    void save(const str& filepath) const;
    void load(const str& filepath);
    void load_replace(const str& filepath);
};

class DataLoading
{
private:
    TextualEmbedding& embedder;
    const Text& Database;
    std::mt19937 gen;
    std::uniform_int_distribution<int> dist;
    std::unordered_set<int> used_indices; 
    const str START_TOKEN = "<START>";
    const str END_TOKEN = "<END>";
    const str PAD_TOKEN = "<PAD>";
    const str UNK_TOKEN = "<UNK>";

public:
    const int batch_size;
    const int context_len;
    DataLoading(TextualEmbedding& embed_ref, const Text& db, const int batch, const int context);
    BatchTexts load_data();
    graph forward(std::shared_ptr<BatchTexts> dataset, const str type = "E");
};

class SingleHeadAttention
{
public:
    const int embed_dim;
    const int hidden;
    Linear q, k, v, out;
    const str type;
    SingleHeadAttention(const int embed_dim, const int t_hidden, const int num_heads = 0);
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    graph forward(const graph&X, const bool mask = false);
    graph cross_forward(const graph& X, const graph& Y);
    graph cached_forward(const graph& X_new, KVCache&cache, bool mask = true);
    graph cached_cross_forward(const graph& X_new, KVCache& cache);
    
};

class MultiHeadAttention
{
public:
    const int embed_dim;
    const int hidden;
    const int num_heads;
    int head_dim;
    str name;
    Linear q, k, v, o;

    MultiHeadAttention(const int embed_dim, const int hidden, const int num_heads, const str& name = "MultiHeadAttention"); 
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);

    graph forward(const graph& X, const bool mask = false); 
    graph cross_forward(const graph& X, const graph& Y);
    graph cached_forward(const graph& X_new, KVCache&cache, bool mask = true);
    graph cached_cross_forward(const graph& X_new, KVCache& cache);
};

class LLM
{
private:
    GraphOperations go;
    const int embed_dim;
    TextualEmbedding& embedder;
    DataLoading Dataload;
    MultiHeadAttention T1, T2, T3;
    TimeMLPBlock fc1, fc2;
    Linear proj;
    int num_heads;
 
public:
    LLM(TextualEmbedding& embed, const Text& Database, int batch, int clen, int hidden_dim=128, int num_heads=8);
    std::tuple<graph, graph, graph> build_train(std::shared_ptr<BatchTexts> data);
    void generate(const Text& prompt, const int max_len = 0);
    void train(const int num_batches, const int percent = 1, const bool model_save = true, const str mod_path = "../LLM.bin", 
    const bool embed_save = true, const str embed_path = "../embedding.temp");
    void save(const str& filename) const;
    void load(const str& filename);
};

