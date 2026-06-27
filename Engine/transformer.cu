#include "transformer.h"

KVCache::KVCache(int max_len, int hidden) : max_len(max_len), hidden(hidden) 
{
SafeCudaMalloc("KVCache K", K, max_len*hidden);
SafeCudaMalloc("KVCache V", V, max_len*hidden);
}

void KVCache::init(int max_len, int hidden)
{
    SafeCudaMalloc("KVCache K", K, max_len*hidden);
    SafeCudaMalloc("KVCache V", V, max_len*hidden);
}

void KVCache::free() {cudaFree(K);cudaFree(V);}

str TextualEmbedding::preprocessWord(const str& word)
{
        str word_lower = word;
        size_t dot_pos = word.find_last_of('.');
        if (dot_pos != str::npos && dot_pos == word.length() - 1) {word_lower = word.substr(0, dot_pos);}
        std::transform(word_lower.begin(), word_lower.end(),word_lower.begin(), ::tolower);
        
        return word_lower;
};

TextualEmbedding::TextualEmbedding(const int embed_dim, const int batch_size, const int max_c_len, const int max_vocab_size, const bool for_LLM)
    : MAX_BATCH_SIZE(batch_size), MAX_CONTEXT_LEN(max_c_len), MAX_VOCAB_SIZE(max_vocab_size), embed_dim(embed_dim)
{
        const uint64_t seed =  ((uint64_t)rd() << 32) | rd();

        SafeCudaMalloc("EmbedSpace", EmbedSpace, MAX_VOCAB_SIZE * embed_dim);
        SafeCudaMalloc("Encoder Keys", encoder_keys, MAX_BATCH_SIZE * MAX_CONTEXT_LEN);
        SafeCudaMalloc("Decoder Keys", decoder_keys, MAX_BATCH_SIZE * MAX_CONTEXT_LEN);
        SafeCudaMalloc("Targets Keys",  target_keys, MAX_BATCH_SIZE * MAX_CONTEXT_LEN);

        GaussianNoise<<<(embed_dim  * MAX_VOCAB_SIZE+tpb-1)/tpb, tpb>>>(EmbedSpace, 0.f, 1.f, embed_dim * MAX_VOCAB_SIZE, seed);
        fillKernel<<<(MAX_BATCH_SIZE*MAX_CONTEXT_LEN+tpb-1)/tpb, tpb>>>(encoder_keys, INT_MIN, MAX_BATCH_SIZE * MAX_CONTEXT_LEN);
        fillKernel<<<(MAX_BATCH_SIZE*MAX_CONTEXT_LEN+tpb-1)/tpb, tpb>>>(decoder_keys, INT_MIN, MAX_BATCH_SIZE * MAX_CONTEXT_LEN);
        fillKernel<<<(MAX_BATCH_SIZE*MAX_CONTEXT_LEN+tpb-1)/tpb, tpb>>>(target_keys,  INT_MIN, MAX_BATCH_SIZE * MAX_CONTEXT_LEN);
        if (for_LLM) updateVocabulary({"<START>", "<END>", "<UNK>", "<PAD>"});
};

TextualEmbedding::~TextualEmbedding()
{
    cudaFree(EmbedSpace);
    cudaFree(encoder_keys);
    cudaFree(decoder_keys);
    cudaFree(target_keys);
}

void TextualEmbedding::updateVocabulary(const Text& texts)
{       
        /*
        @author Updates the vocabulary with new words from the input texts. 
        If new words are added, their embeddings are initialized with Gaussian noise.
        */
        const uint64_t seed =  ((uint64_t)rd() << 32) | rd();

        std::vector<int> new_indices;
        for(const auto& word : texts) 
        {
            str word_lower = preprocessWord(word);
            if (Vocabulary.find(word_lower) == Vocabulary.end()) {
            int index = WordSpace.size();
            if(index >= MAX_VOCAB_SIZE) 
            {
            std::cout << "Warning: Vocabulary size exceeded maximum limit of "<< MAX_VOCAB_SIZE << ". Skipping word: " << word_lower << "\n";
            continue;
            }

            WordSpace[word_lower] = index;
            KeySpace[index] = word_lower;
            Vocabulary.insert(word_lower);
            new_indices.push_back(index);
            }}
             
};
    
void TextualEmbedding::encodeText(const Text& texts,const str key, const int batch_idx)
{
    /*
    @author Encodes a single text input into its corresponding indices in the embedding space.
    The encoded indices are stored in the keys tensor at the position corresponding to the batch index.
    */
    if(key != "E" && key != "D" && key !="T")
    {
        std::cout << "Warning... Input to TextualEmbedding text encoder MUST be either E D or T... received " << key << " which is invalid \n";
        std::cout << "Failed to Embed and X will be all initial values (most likely 0s) \n";
    }

    if(batch_idx >= MAX_BATCH_SIZE) 
    {
        std::cout << "Error: batch_idx " << batch_idx << " exceeds MAX_BATCH_SIZE " << MAX_BATCH_SIZE << "\n";
        return;
    }
    
    const int offset = batch_idx * MAX_CONTEXT_LEN;
    if (key == "E") fillKernel<<<(MAX_CONTEXT_LEN+tpb-1)/tpb, tpb>>>(encoder_keys + offset, INT_MIN, MAX_CONTEXT_LEN);
    if (key == "D") fillKernel<<<(MAX_CONTEXT_LEN+tpb-1)/tpb, tpb>>>(decoder_keys + offset, INT_MIN, MAX_CONTEXT_LEN);
    if (key == "T") fillKernel<<<(MAX_CONTEXT_LEN+tpb-1)/tpb, tpb>>>(target_keys  + offset, INT_MIN, MAX_CONTEXT_LEN);

    const int num_tokens = std::min((int)texts.size(), MAX_CONTEXT_LEN);
    
    for(int i = 0; i < num_tokens; ++i)
    {
        str word_lower = preprocessWord(texts[i]);
        if (word_lower == "" || word_lower == " ") continue;
        int index = -1;
        auto it = WordSpace.find(word_lower);
        if(it != WordSpace.end()) index = it->second;
        //else std::cout << "Warning: Word '" << word_lower << "' not in vocabulary. Using masked value.\n";
        if(index >= MAX_VOCAB_SIZE || index < 0) 
        {
            std::cout << "Error: Invalid index " << index << " for word: " << word_lower << "\n";
            continue;
        }

        if(key == "E"){ cudaMemcpy(&encoder_keys[offset + i], &index, sizeof(int), cudaMemcpyHostToDevice); continue;}
        if(key == "D"){ cudaMemcpy(&decoder_keys[offset + i], &index, sizeof(int), cudaMemcpyHostToDevice); continue;}
        if(key == "T"){ cudaMemcpy( &target_keys[offset + i], &index, sizeof(int), cudaMemcpyHostToDevice); continue;}
    }
}

void TextualEmbedding::rencodeText(const Text& texts, const str key, const int batch_idx, const int start_idx)
{
    /*
    @author Recursively encodes a single text input into its corresponding indices in the embedding space
    starting from the given start index. The encoded indices are stored in the keys tensor at the position 
    corresponding to the batch index.
    */
    if(key !="E" && key !="D" && key !="T")
    {
            std::cout << "Warning... Input to TextualEmbedding recursive encoder pass MUST be either E D or T... received " << key << " which is invalid \n";
            std::cout << "Failed to Embed and X will be all initial values (most likely 0s) \n";
    }

    if(batch_idx >= MAX_BATCH_SIZE) 
    {
        std::cout << "Error: batch_idx " << batch_idx << " exceeds MAX_BATCH_SIZE " << MAX_BATCH_SIZE << "\n";
        std::exit(1);
    }
    
    const int offset = batch_idx * MAX_CONTEXT_LEN + start_idx;
    const int num_tokens = std::min((int)texts.size(), MAX_CONTEXT_LEN - start_idx);
    
    for(int i = 0; i < num_tokens; ++i)
    {
    str word_lower = preprocessWord(texts[i]);
    int index = -1;
    auto it = WordSpace.find(word_lower);
    if(it != WordSpace.end()) {index = it->second;} 
    else {std::cout << "Warning: Word '" << word_lower << "' not in vocabulary. Using masked value.\n";}
    if(index >= MAX_VOCAB_SIZE || index < 0) {
    std::cout << "Error: Invalid index " << index << " for word: " << word_lower << "\n";continue;}
    if(key == "E") {cudaMemcpy(&encoder_keys[offset + i], &index, sizeof(int), cudaMemcpyHostToDevice); continue;}
    if(key == "D") {cudaMemcpy(&decoder_keys[offset + i], &index, sizeof(int), cudaMemcpyHostToDevice); continue;}
    if(key == "T") {cudaMemcpy(&target_keys[offset + i], &index, sizeof(int), cudaMemcpyHostToDevice); continue;}
    }
    
    //CheckError("Text encoding");
    }

void TextualEmbedding::encodeBatch(const BatchText& batch_texts, const str key)
{
    /* @author 
    Encodes batches of texts into their corresponding indices in the embedding space. 
    The encoded indices are stored in the keys tensor at the position corresponding to the batch index.
    */
    const int batch_size = std::min((int)batch_texts.size(), MAX_BATCH_SIZE);
    for(int b = 0; b < batch_size; ++b) 
    {
        encodeText(batch_texts[b], key,b);
    }}

void TextualEmbedding::forward(const graph&X, const str key) 
{
        /*
        @author Performs the forward pass to retrieve embeddings for the encoded texts.
        It gathers embeddings from the embedding space based on the keys tensor and returns the output tensor matrices.
        */

        if(X->dim[0] > MAX_BATCH_SIZE || X->dim[2] > MAX_CONTEXT_LEN) 
        {
            printf("Dimension Mismatch... Received (Batch x Context): (%i, %i), Max: (%i, %i)", 
            X->dim[0], X->dim[2], MAX_BATCH_SIZE, MAX_CONTEXT_LEN);
            std::exit(1);
        }
        const int bpg = (X->total+tpb-1)/tpb;
        if(key == "E") GatherEmbeddings<<<bpg, tpb>>>(X->output, EmbedSpace, encoder_keys, X->dim[2], MAX_CONTEXT_LEN, embed_dim, X->total);
        if(key == "D") GatherEmbeddings<<<bpg, tpb>>>(X->output, EmbedSpace, decoder_keys, X->dim[2], MAX_CONTEXT_LEN, embed_dim, X->total);
        if(key == "T")
        {
            const int size = X->dim[0] * X->dim[2];
            fillKernel<<<bpg, tpb>>>(X->output, 0.0f, X->total);
            OneHotEmbeddings<<<bpg,tpb>>>(X->output, target_keys, X->dim[2], MAX_CONTEXT_LEN, Vocabulary.size(), size);
        }

    if(key !="E" && key != "D" && key !="T")
    {
            std::cout << "Warning... Input to TextualEmbedding Forward pass MUST be either E D or T... received " << key << " which is invalid \n";
            std::cout << "Failed to Embed and X will be all initial values (most likely 0s) \n";
    }
        //CheckError("Forward pass - gather embeddings");
}

void TextualEmbedding::rforward(const graph&X,const str key, const int start_idx)
{
        /*
        @author Performs the recursive forward pass to retrieve embeddings for the encoded texts at the bottom of X.
        It gathers embeddings from the embedding space based on the keys tensor and returns the output tensor matrices.
        It also assumes you're only working on batch 1.
        */
        if(X->dim[0] != 1 || X->dim[2] > MAX_CONTEXT_LEN) 
        {
        printf("Dimension Mismatch... Received (Batch x Context): (%i, %i), Max: (%i, %i), can only recursively call on single batches", 
            X->dim[0], X->dim[2], MAX_BATCH_SIZE, MAX_CONTEXT_LEN);std::exit(1);
        }
        if(key !="E" && key != "D" && key !="T")
        {
            std::cout << "Warning... Input to TextualEmbedding recursive forward pass MUST be either E D or T... received " << key << " which is invalid \n";
            std::cout << "Failed to Embed and X will be all initial values (most likely 0s) \n";
        }

        const int bpg = (embed_dim+tpb-1)/tpb;
        const int xOffset = start_idx * embed_dim;
        if (key == "E") GatherEmbeddings<<<bpg, tpb>>>(X->output + xOffset, EmbedSpace, encoder_keys + start_idx, 1, 1, embed_dim, embed_dim);
        if (key == "D") GatherEmbeddings<<<bpg, tpb>>>(X->output + xOffset, EmbedSpace, decoder_keys + start_idx, 1, 1, embed_dim, embed_dim);
                
        //CheckError("Forward pass - gather embeddings");
    }
        
void TextualEmbedding::EmbeddingUpdate(const graph& X, const str key)
{
        /*
        @author Updates the embedding space using gradients from backpropagation. 
        If custom keys are provided, they are used for the update; otherwise, the internal keys are used
        */
       const int tpb = THREADSPERBLOCK; 
       const int bpg = (X->total+tpb-1)/tpb;
       if(key == "E") KeyUpdate<<<bpg,tpb>>>(EmbedSpace, X->grad, encoder_keys,X->dim[2],MAX_CONTEXT_LEN, embed_dim,LEARNING_RATE,X->total);
       if(key == "D") KeyUpdate<<<bpg,tpb>>>(EmbedSpace, X->grad, decoder_keys,X->dim[2],MAX_CONTEXT_LEN, embed_dim,LEARNING_RATE,X->total);
       //CheckError("Key Update in Embedding Space update in Textual Embedding");
        
}

void TextualEmbedding::load_replace(const str& filepath)
{
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open())
    {
        std::cerr << "TextualEmbedding::load_replace — could not open '" << filepath << "'\n";
        return;
    }
    uint32_t magic = 0, version = 0;
    f.read(reinterpret_cast<char*>(&magic),   sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x54454D42u)
    {
        std::cerr << "TextualEmbedding::load_replace — bad magic number; not a TEMB file\n";
        return;
    }

    int saved_dim = 0, saved_vocab_size = 0;
    f.read(reinterpret_cast<char*>(&saved_dim),        sizeof(saved_dim));
    f.read(reinterpret_cast<char*>(&saved_vocab_size), sizeof(saved_vocab_size));

    if (saved_dim != embed_dim)
    {
        std::cerr << "TextualEmbedding::load_replace — embed_dim mismatch: "
                  << "file=" << saved_dim << " object=" << embed_dim << "\n";
        return;
    }

    // ── read vocab + build word→saved_index map ─────────────
    // We only care about words that exist in the current WordSpace.
    // Collect (local_index, saved_index) pairs for matched words.
    struct IndexPair { int local_idx; int saved_idx; };
    std::vector<IndexPair> matches;

    for (int i = 0; i < saved_vocab_size; ++i)
    {
        uint32_t wlen = 0;
        f.read(reinterpret_cast<char*>(&wlen), sizeof(wlen));

        str word(wlen, '\0');
        f.read(word.data(), wlen);

        int saved_idx = 0;
        f.read(reinterpret_cast<char*>(&saved_idx), sizeof(saved_idx));

        auto it = WordSpace.find(word);
        if (it != WordSpace.end())
            matches.push_back({ it->second, saved_idx });
    }

    if (matches.empty())
    {
        std::cout << "TextualEmbedding::load_replace — no vocabulary overlap found, nothing updated\n";
        return;
    }

    // ── stream embedding rows and copy matched ones ──────────
    // Read the saved embed matrix row by row; when a row index is in our
    // match set, copy that row to the correct local slot on the GPU.
    // Sorting matches by saved_idx lets us seek through the file sequentially.
    std::sort(matches.begin(), matches.end(),
              [](const IndexPair& a, const IndexPair& b){ return a.saved_idx < b.saved_idx; });

    std::vector<float> row(embed_dim);
    int match_cursor = 0;

    for (int s = 0; s < saved_vocab_size && match_cursor < (int)matches.size(); ++s)
    {
        f.read(reinterpret_cast<char*>(row.data()), embed_dim * sizeof(float));
        if (!f.good())
        {
            std::cerr << "TextualEmbedding::load_replace — read error at embedding row " << s << "\n";
            return;
        }

        if (s == matches[match_cursor].saved_idx)
        {
            const int local_idx = matches[match_cursor].local_idx;
            float* dst = EmbedSpace + static_cast<size_t>(local_idx) * embed_dim;

            cudaError_t err = cudaMemcpy(dst, row.data(),
                                         embed_dim * sizeof(float),
                                         cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                std::cerr << "TextualEmbedding::load_replace — cudaMemcpy failed for word at saved_idx="
                          << s << ": " << cudaGetErrorString(err) << "\n";

            ++match_cursor;
        }
    }

    std::cout << "TextualEmbedding::load_replace — replaced " << match_cursor
              << "/" << (int)WordSpace.size()
              << " embeddings from '" << filepath << "'\n";
}

void TextualEmbedding::save(const str& filepath) const
{
    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open())
    {
        std::cerr << "TextualEmbedding::save — could not open '" << filepath << "' for writing\n";
        return;
    }

    const uint32_t MAGIC   = 0x54454D42u;   // "TEMB"
    const uint32_t VERSION = 1u;
    f.write(reinterpret_cast<const char*>(&MAGIC),   sizeof(MAGIC));
    f.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
    f.write(reinterpret_cast<const char*>(&embed_dim), sizeof(embed_dim));

    const int vocab_size = static_cast<int>(WordSpace.size());
    f.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));

    // ── vocabulary ──────────────────────────────────────────
    // Iterate WordSpace so we get (word → index) pairs directly.
    for (const auto& [word, idx] : WordSpace)
    {
        const uint32_t wlen = static_cast<uint32_t>(word.size());
        f.write(reinterpret_cast<const char*>(&wlen), sizeof(wlen));
        f.write(word.data(), wlen);
        f.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
    }

    // ── embedding matrix ────────────────────────────────────
    // Pull the used rows from the GPU (vocab_size rows × embed_dim floats).
    const size_t embed_bytes = static_cast<size_t>(vocab_size) * embed_dim * sizeof(float);
    std::vector<float> host_embed(static_cast<size_t>(vocab_size) * embed_dim);

    cudaError_t err = cudaMemcpy(host_embed.data(), EmbedSpace,
                                 embed_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "TextualEmbedding::save — cudaMemcpy D→H failed: "
                  << cudaGetErrorString(err) << "\n";
        return;
    }

    f.write(reinterpret_cast<const char*>(host_embed.data()),
            static_cast<std::streamsize>(embed_bytes));

    if (!f.good())
        std::cerr << "TextualEmbedding::save — write error on '" << filepath << "'\n";
    else
        std::cout << "TextualEmbedding saved: vocab=" << vocab_size
                  << " embed_dim=" << embed_dim
                  << " → '" << filepath << "'\n";
}

void TextualEmbedding::load(const str& filepath)
{
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open())
    {
        std::cerr << "TextualEmbedding::load — could not open '" << filepath << "'\n";
        return;
    }

    uint32_t magic = 0, version = 0;
    f.read(reinterpret_cast<char*>(&magic),   sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x54454D42u)
    {
        std::cerr << "TextualEmbedding::load — bad magic number; not a TEMB file\n";
        return;
    }
    if (version != 1u)
    {
        std::cerr << "TextualEmbedding::load — unsupported version " << version << "\n";
        return;
    }

    int saved_dim = 0, vocab_size = 0;
    f.read(reinterpret_cast<char*>(&saved_dim),   sizeof(saved_dim));
    f.read(reinterpret_cast<char*>(&vocab_size),  sizeof(vocab_size));

    if (saved_dim != embed_dim)
    {
        std::cerr << "TextualEmbedding::load — embed_dim mismatch: "
                  << "file=" << saved_dim << " object=" << embed_dim << "\n";
        return;
    }
    if (vocab_size > MAX_VOCAB_SIZE)
    {
        std::cerr << "TextualEmbedding::load — saved vocab (" << vocab_size
                  << ") exceeds MAX_VOCAB_SIZE (" << MAX_VOCAB_SIZE << ")\n";
        return;
    }

    // ── vocabulary ──────────────────────────────────────────
    WordSpace.clear();
    KeySpace.clear();
    Vocabulary.clear();

    for (int i = 0; i < vocab_size; ++i)
    {
        uint32_t wlen = 0;
        f.read(reinterpret_cast<char*>(&wlen), sizeof(wlen));

        str word(wlen, '\0');
        f.read(word.data(), wlen);

        int idx = 0;
        f.read(reinterpret_cast<char*>(&idx), sizeof(idx));

        WordSpace[word] = idx;
        KeySpace[idx]   = word;
        Vocabulary.insert(word);
    }

    // ── embedding matrix ────────────────────────────────────
    const size_t embed_bytes = static_cast<size_t>(vocab_size) * embed_dim * sizeof(float);
    std::vector<float> host_embed(static_cast<size_t>(vocab_size) * embed_dim);

    f.read(reinterpret_cast<char*>(host_embed.data()),
           static_cast<std::streamsize>(embed_bytes));

    if (!f.good())
    {
        std::cerr << "TextualEmbedding::load — read error on '" << filepath << "'\n";
        return;
    }

    cudaError_t err = cudaMemcpy(EmbedSpace, host_embed.data(),
                                 embed_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "TextualEmbedding::load — cudaMemcpy H→D failed: "
                  << cudaGetErrorString(err) << "\n";
        return;
    }

    std::cout << "TextualEmbedding loaded: vocab=" << vocab_size
              << " embed_dim=" << embed_dim
              << " ← '" << filepath << "'\n";
}

DataLoading::DataLoading(TextualEmbedding& embed_ref, const Text& db, const int batch, const int context):
    embedder(embed_ref), Database(db), batch_size(batch), context_len(context), gen(std::random_device{}())
{
    if(batch > embedder.MAX_BATCH_SIZE || context > embedder.MAX_CONTEXT_LEN){
        printf("Cannot load data for batches or context > embedders construct");
        printf("Requested (batch, context): (%i, %i), Maximum: (%i, %i)", batch, context, embedder.MAX_BATCH_SIZE, embedder.MAX_CONTEXT_LEN);
    }
    dist = std::uniform_int_distribution<int>(0, Database.size() - context);
}

BatchTexts DataLoading::load_data()
{
    BatchTexts batch_data(batch_size, context_len);
    used_indices.clear();

    for (int b = 0; b < batch_size; ++b)
    {
        int start_idx;
        do {
            start_idx = dist(gen);
        } while (used_indices.find(start_idx) != used_indices.end());

        used_indices.insert(start_idx);
        batch_data.decoder[b][0] = START_TOKEN;

        for (int i = 0; i < context_len; ++i)
        {
            batch_data.encoder[b][i] = Database[start_idx + i];
            batch_data.decoder[b][i + 1] = Database[start_idx + i];
            batch_data.target[b][i] = Database[start_idx + i];
        }
        batch_data.encoder[b][context_len] = END_TOKEN;
        batch_data.target[b][context_len] =  END_TOKEN;
    }
    return batch_data;
}
    
graph DataLoading::forward(std::shared_ptr<BatchTexts> dataset, const str type)
{
    if (type != "E" && type != "D" && type != "T") {
        std::cerr << "Invalid type: " << type << "\n";
        std::exit(1);
    }
    const int batch_size = dataset->encoder.size();
    const int row = context_len + 1;

    const int col = (type != "T") ? embedder.embed_dim : embedder.Vocabulary.size();

    graph node = (type == "T")
        ? std::make_shared<NodeBackProp>("Target One-Hot",    batch_size, 1, row, col, 1)
        : std::make_shared<NodeBackProp>(type + " Embeddings", batch_size, 1, row, col, 1);

    node->inputs = {};
    node->forward = [=]()  
    {
        const BatchText& data = (type == "E") ? dataset->encoder
                              : (type == "D") ? dataset->decoder
                              :                 dataset->target;
        embedder.encodeBatch(data, type);
        embedder.forward(node, type);
    };
    node->free      = [=](){ node->clear(); };
    node->zero_grad = [=](){ Zerograd(node); };
    return node;
}


SingleHeadAttention::SingleHeadAttention(const int embed_dim, const int t_hidden, const int num_heads):
embed_dim(embed_dim), type(type), hidden(t_hidden),
q(embed_dim, t_hidden, "Transformer Q"), k(embed_dim, t_hidden, "Transformer K"),
v(embed_dim, t_hidden, "Transformer V"), out(t_hidden, embed_dim, "Transformer Output"){}
void SingleHeadAttention::save(std::ofstream& f) const {q.save(f); k.save(f); v.save(f); out.save(f);}
void SingleHeadAttention::load(std::ifstream& f){ q.load(f); k.load(f); v.load(f);out.load(f);}   
graph SingleHeadAttention::forward(const graph&X, const bool mask)
{
    auto Q = q.forward(X);
    auto K = k.forward(X);
    auto V = v.forward(X);
    auto scores = GraphOperations::BMMABT(Q, K);
    auto scaled_scores = GraphOperations::Scale(scores, 1.0f / sqrtf((float)(hidden)));
    auto attn_weights = mask ? GraphOperations::SOFTMASK(scaled_scores,1): GraphOperations::SOFTMAX(scaled_scores,1);
    auto attn_output = GraphOperations::BMM(attn_weights, V);
    auto output = out.forward(attn_output);
    output->op_name = type + "Transformer Block Output";
    return output;
}
graph SingleHeadAttention::cross_forward(const graph& X, const graph& Y)
{
    /*@author Single Head Cross attention implementation, query from X, key and value from Y*/
    auto Q = q.forward(X);
    auto K = k.forward(Y);
    auto V = v.forward(Y);
    auto scores = GraphOperations::BMMABT(Q, K);
    auto scaled_scores = GraphOperations::Scale(scores,1.0f/sqrtf((float)(embed_dim)));
    auto attn_weights =  GraphOperations::SOFTMASK(scaled_scores,1);
    auto attn_output = GraphOperations::BMM(attn_weights, V); 
    auto output = out.forward(attn_output);
    output->op_name = " Cross SHA Block Output";
    return output;
}
graph SingleHeadAttention::cached_forward(const graph& X_new, KVCache&cache, bool mask)
{ 
    auto K_new = k.forward(X_new);    
    auto V_new = v.forward(X_new);

    K_new->forward();
    V_new->forward();

    int offset = cache.current_len * cache.hidden;
    const int tpb = THREADSPERBLOCK;
    const int bpg = (cache.hidden+tpb-1)/tpb;
    Write<<<bpg,tpb>>>(K_new->output, cache.K+offset,cache.hidden);
    Write<<<bpg,tpb>>>(V_new->output, cache.V+offset,cache.hidden);
    cache.current_len += K_new->dim[2];

    K_new->clear();
    V_new->clear();

    graph K_full = std::make_shared<NodeBackProp>("KV_K",1,1,cache.current_len,cache.hidden,0);
    graph V_full = std::make_shared<NodeBackProp>("KV_V",1,1,cache.current_len,cache.hidden,0);

    K_full->output = cache.K;
    V_full->output = cache.V;

    // ============================= // 
    auto Q_new  = q.forward(X_new);
    auto scores = GraphOperations::BMMABT(Q_new, K_full);
    auto scaled  = GraphOperations::Scale(scores, 1.0f / sqrtf((float)hidden));
    auto weights = mask ? GraphOperations::SOFTMASK(scaled, 1) : GraphOperations::SOFTMAX(scaled, 1);
    auto attn_out = GraphOperations::BMM(weights, V_full);
    auto output = out.forward(attn_out);
    output->op_name = "SHA Cached Forward";
    return output;
}
graph SingleHeadAttention::cached_cross_forward(const graph& X_new, KVCache& cache)
{
    auto Q_new = q.forward(X_new);   
    graph K_full = std::make_shared<NodeBackProp>("KV_K",1,1,cache.current_len,cache.hidden,0);
    graph V_full = std::make_shared<NodeBackProp>("KV_V",1,1,cache.current_len,cache.hidden,0);
    K_full->output = cache.K;
    V_full->output = cache.V;
    auto scores = GraphOperations::BMMABT(Q_new, K_full);
    auto scaled  = GraphOperations::Scale(scores, 1.0f / sqrtf((float)hidden));
    auto weights = GraphOperations::SOFTMASK(scaled, 1);
    auto attn_out = GraphOperations::BMM(weights, V_full);
    auto output = out.forward(attn_out); output->op_name = "SHA output";
    return output;
}

MultiHeadAttention::MultiHeadAttention(const int embed_dim, const int hidden, const int num_heads, const str& name) : 
    hidden(hidden), embed_dim(embed_dim), num_heads(num_heads), name(name),
    q(embed_dim, hidden, name + " Query"), k(embed_dim, hidden, name + " Key"),
    v(embed_dim, hidden, name + " Value"), o(hidden, embed_dim, name + " Output")

    {   
        if(hidden % num_heads != 0 )
        {printf("Cannot Split MHA because hidden dim: %i is not divisible by num heads: %i", hidden, num_heads);}
        head_dim = hidden / num_heads;
    } 
void MultiHeadAttention::save(std::ofstream& f) const{q.save(f); k.save(f); v.save(f); o.save(f);}
void MultiHeadAttention::load(std::ifstream& f)      {q.load(f); k.load(f); v.load(f); o.load(f);}
graph MultiHeadAttention::forward(const graph& X, const bool mask) 
{
        auto Q = GraphOperations::HeadifytoChannel(q.forward(X),num_heads); 
        auto K = GraphOperations::HeadifytoChannel(k.forward(X),num_heads);
        auto V = GraphOperations::HeadifytoChannel(v.forward(X),num_heads); 
        auto scores = GraphOperations::BMMABT(Q, K);
        auto scaled_scores = GraphOperations::Scale(scores, 1.0f / sqrtf((float)(head_dim))); 
        auto attn_weights = (mask && scaled_scores->dim[2] > 1) ? GraphOperations::SOFTMASK(scaled_scores,1): GraphOperations::SOFTMAX(scaled_scores,1); 
        auto attn_output = GraphOperations::BMM(attn_weights, V);
        auto attn_concat = GraphOperations::DeHeadify(attn_output);
        auto output = o.forward(attn_concat);  
        output->op_name = "Multi Head Attention Output";
        return output;

} 
graph MultiHeadAttention::cross_forward(const graph& X, const graph& Y)
{
        auto Q = GraphOperations::HeadifytoChannel(q.forward(X), num_heads);
        auto K = GraphOperations::HeadifytoChannel(k.forward(Y), num_heads);
        auto V = GraphOperations::HeadifytoChannel(v.forward(Y), num_heads);
        auto scores = GraphOperations::BMMABT(Q, K);
        auto scaled_scores = GraphOperations::Scale(scores,1.0f/sqrtf((float)(embed_dim)));
        auto attn_weights =  (scaled_scores->dim[2] > 1) ? GraphOperations::SOFTMASK(scaled_scores,1): GraphOperations::SOFTMAX(scaled_scores,1); 
        auto attn_output = GraphOperations::BMM(attn_weights, V); 
        auto attn_concat = GraphOperations::DeHeadify(attn_output);
        auto output = o.forward(attn_concat);
        output->op_name = " Cross MHA Block Output";
        return output;
}
graph MultiHeadAttention::cached_forward(const graph& X_new, KVCache&cache, bool mask)
{ 
        auto K_new = k.forward(X_new);   
        auto V_new = v.forward(X_new);

        // Handling Cache Update ======== // Can add MemcpyAsync for SpeedUp;
        
        K_new->forward();
        V_new->forward();

        int offset = cache.current_len * cache.hidden;
        const int tpb = THREADSPERBLOCK;
        const int bpg = (cache.hidden+tpb-1)/tpb;
        Write<<<bpg,tpb>>>(K_new->output, cache.K+offset,cache.hidden);
        Write<<<bpg,tpb>>>(V_new->output, cache.V+offset,cache.hidden);
        cache.current_len += K_new->dim[2];
        CheckError("Write");

        K_new->clear();
        V_new->clear();

        graph K_full = std::make_shared<NodeBackProp>("KV_K",1,1,cache.current_len,cache.hidden,0);
        graph V_full = std::make_shared<NodeBackProp>("KV_V",1,1,cache.current_len,cache.hidden,0);

        K_full->output = cache.K;
        V_full->output = cache.V;

        auto Q_new  = q.forward(X_new);
        auto scores = GraphOperations::BMMABT(GraphOperations::HeadifytoChannel(Q_new, num_heads), GraphOperations::HeadifytoChannel(K_full, num_heads));
        auto scaled  = GraphOperations::Scale(scores, 1.0f / sqrtf((float)head_dim));
        auto weights = (mask && scaled->dim[2] > 1) ? GraphOperations::SOFTMASK(scaled,1): GraphOperations::SOFTMAX(scaled,1); 
        auto attn_out = GraphOperations::BMM(weights, GraphOperations::HeadifytoChannel(V_full, num_heads));
        auto attn_concat = GraphOperations::DeHeadify(attn_out);
        auto output = o.forward(attn_concat);
        output->op_name = "MHA Cached Forward";
        return output;
}
graph MultiHeadAttention::cached_cross_forward(const graph& X_new, KVCache& cache)
{
        auto Q_new = q.forward(X_new);   
        graph K_full = std::make_shared<NodeBackProp>("KV_K",1,1,cache.current_len,cache.hidden,0);
        graph V_full = std::make_shared<NodeBackProp>("KV_V",1,1,cache.current_len,cache.hidden,0);
        K_full->output = cache.K;
        V_full->output = cache.V;
        auto scores = GraphOperations::BMMABT(GraphOperations::HeadifytoChannel(Q_new, num_heads), GraphOperations::HeadifytoChannel(K_full, num_heads));
        auto scaled  = GraphOperations::Scale(scores, 1.0f / sqrtf((float)head_dim));
        auto weights = (scaled->dim[2] > 1) ? GraphOperations::SOFTMASK(scaled,1): GraphOperations::SOFTMAX(scaled,1); 
        auto attn_out = GraphOperations::BMM(weights, GraphOperations::HeadifytoChannel(V_full, num_heads));
        auto attn_concat = GraphOperations::DeHeadify(attn_out);  
        auto output = o.forward(attn_concat); output->op_name = "SHA output";
        return output;
}

LLM::LLM(TextualEmbedding& embed, const Text& Database, int batch, int clen, int hidden_dim, int num_heads) :
    embedder(embed), embed_dim(embed.embed_dim), 
    Dataload(embed, Database, batch, clen), num_heads(num_heads),
    T1(embed_dim, hidden_dim, num_heads),
    T2(embed_dim, hidden_dim, num_heads),
    T3(embed_dim, hidden_dim, num_heads),
    fc1(embed_dim, embed_dim), 
    fc2(embed_dim, embed_dim), proj(embed_dim, embedder.Vocabulary.size(), "Projection")
    {
        if(embedder.Vocabulary.size()  == 0)
        {
            std::cerr << "You may have forgotten to call embedder.updateVocabulary(data) outside the LLM Constructor. \n";
            std::cerr << "Please update the vocabulary with your dataset before initializing the LLM. \n";
            std::exit(1);
        }
        if (embedder.Vocabulary.size() <= clen)
        {
            std::cerr << "Warning: Vocabulary size is less than or equal to context length. This may lead to issues with training. \n";
        }

    }
std::tuple<graph, graph, graph> LLM::build_train(std::shared_ptr<BatchTexts> data)
{
    auto encoder_embeds = Dataload.forward(data, "E");
    auto decoder_embeds = Dataload.forward(data, "D");
    
    auto pos_encodded = GraphOperations::MatrixPositionalEncoding(encoder_embeds); 
    auto Att1 = T1.forward(pos_encodded); 
    auto A1 = fc1.forward(Att1); 

    auto AN1 = GraphOperations::LayerNorm(GraphOperations::Add(A1,Att1));
    auto pos_decoded = GraphOperations::MatrixPositionalEncoding(decoder_embeds);
    auto Att2 = T2.forward(pos_decoded); 
    auto CrossAtt = T3.cross_forward(Att2, AN1);

    auto A2 = fc2.forward(CrossAtt);
    auto AN2 = GraphOperations::LayerNorm(A2 + CrossAtt);

    auto logits = proj.forward(AN2);
    return {encoder_embeds, decoder_embeds, logits};
}
void LLM::generate(const Text& prompt, const int max_len)
{
        graph encoder_embeds = std::make_shared<NodeBackProp>("Encoder Embeddings", 1, 1, prompt.size(), embed_dim, 1);
        graph decoder_embeds = std::make_shared<NodeBackProp>("Decoder Embeddings", 1, 1, 1, embed_dim, 1);

        embedder.encodeText(prompt, "E");
        embedder.forward(encoder_embeds, "E");

        embedder.encodeText({"<START>"}, "D");
        embedder.forward(decoder_embeds, "D");

        // auto encoder_embeds = embedder.forward(prompt, "E");
        // auto decoder_embeds = embedder.forward({"<START>"}, "D");

        // Encoder ================ //
        auto pos_encodded = GraphOperations::MatrixPositionalEncoding(encoder_embeds);
        auto Att1 = T1.forward(pos_encodded);
        auto A1 = fc1.forward(Att1);
        auto AN1 = GraphOperations::LayerNorm(GraphOperations::Add(A1,Att1));
        auto K_enc = T3.k.forward(AN1);
        auto V_enc = T3.v.forward(AN1);
        // ======================= //

        go.nodes = topological_sort(AN1);
        go.forward();

        // Initializing KV_Cache ====//
        K_enc->forward();
        V_enc->forward();
        
        KVCache kv_T2(embedder.MAX_CONTEXT_LEN, T2.hidden);
        KVCache kv_T3(embedder.MAX_CONTEXT_LEN, T3.hidden);
        kv_T3.current_len = V_enc->dim[2];

        Write<<<(K_enc->total+THREADSPERBLOCK-1)/THREADSPERBLOCK,THREADSPERBLOCK>>>(K_enc->output, kv_T3.K, K_enc->total);
        Write<<<(V_enc->total+THREADSPERBLOCK-1)/THREADSPERBLOCK,THREADSPERBLOCK>>>(V_enc->output, kv_T3.V, V_enc->total);
        // ==========================//

        go.clear_graph();
        K_enc->free();
        V_enc->free();
        int start_idx = 0;
        
        std::cout << "Generating text: \n" <<  "\n";
        for (const auto& word : prompt) std::cout << word << "\t";
        while (true)
        {
            auto pos = GraphOperations::MatrixPositionalEncoding(decoder_embeds, start_idx);
            auto Att2 = T2.cached_forward(pos, kv_T2);
            auto CrossAtt = T3.cached_cross_forward(Att2, kv_T3); 
            auto A2 = fc2.forward(CrossAtt);
            auto AN2 = GraphOperations::LayerNorm(GraphOperations::Add(A2, CrossAtt));
            auto logits = proj.forward(AN2);

            go.nodes = topological_sort(logits);
            go.forward();

            int next_token = TopKSampleToCPU(logits, 10);
            const str next_word = embedder.KeySpace[next_token];
            if (next_word == "<end>" || next_word == "<pad>" || kv_T2.current_len >= embedder.MAX_CONTEXT_LEN) break;
            std::cout << next_word << "\t ";
            if (max_len != 0 && kv_T2.current_len >= max_len) break;

            embedder.rencodeText({next_word}, "D");
            embedder.rforward(decoder_embeds, "D");
            go.clear_graph();
            start_idx++;
        }
        std::cout << "\n \n";
        kv_T2.free();
        kv_T3.free();
        decoder_embeds->clear();
}
void LLM::train(const int num_batches, const int percent, const bool model_save, const str mod_path, const bool embed_save, const str embed_path)
{
    auto batch_data = std::make_shared<BatchTexts>(Dataload.batch_size, embedder.embed_dim);        
    auto [enc, dec, logits] = build_train(batch_data);
    auto target = Dataload.forward(batch_data, "T");
    auto probs = GraphOperations::SOFTMAX(logits);
    auto loss = GraphOperations::CrossEntropy(probs, target, true); 
    go.nodes = topological_sort(loss);
    float curr_loss = FLT_MAX;
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
    {
        *batch_data = Dataload.load_data(); 
        go.zero_grad();
        if(!(batch_idx % percent))
        {  
            go.forward(nullptr, true);
            isNan(loss);
            const float new_loss = ReadValueAt(loss, 0); 
            printf("Loss: %f | Batch: %i \n", new_loss, batch_idx);
            
            if(!batch_idx && embed_save) embedder.save(embed_path);
            if(model_save && curr_loss > new_loss) {save(mod_path); embedder.save(embed_path); curr_loss = new_loss;}
        } 
        else go.forward();
        go.backward();
        //if(!(batch_idx % percent)) printHeadGPU(enc,1);
        embedder.EmbeddingUpdate(enc, "E");
        go.ParameterUpdate(nullptr, false, 1e-4f);
    }  
    go.clear_graph();   
}
void LLM::save(const str& filename) const
{
        std::ofstream f(filename, std::ios::binary);
        if (!f) {
            std::cerr << "Error opening file for saving: " << filename << std::endl;
            std::exit(1);
        }
        T1.save(f); T2.save(f); T3.save(f);
        fc1.save(f); fc2.save(f);
        proj.save(f);
        f.close();
        std::cout << "Model saved successfully to " << filename << "\n";
    
}
void LLM::load(const str& filename)
{
        std::ifstream f(filename, std::ios::binary);
        if (!f) {
            std::cerr << "Error opening file for loading: " << filename << std::endl;
            std::exit(1);
        }
        T1.load(f); T2.load(f); T3.load(f); fc1.load(f); fc2.load(f); proj.load(f);
        f.close();
        std::cout << "Model loaded successfully from " << filename << "\n";
}
