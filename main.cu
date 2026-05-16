#pragma once
#include "includes/transformer.h"
#include "includes/diffusion.h"

/*

int main(){
    const int MAX_VOCAB_SIZE = 25000;
    const int MAX_BATCH_SIZE = 16;
    const int MAX_CONTEXT_LEN = 16384;
    const int EMBED_DIM = 128;
    const int context_len = 32;
    const int HIDDEN_DIM = 256;

    GraphOperations go;
    TextualEmbedding embedder(EMBED_DIM,MAX_BATCH_SIZE, MAX_CONTEXT_LEN, MAX_VOCAB_SIZE);
    Text Db = LoadStory("C:/Users/victo/Documents/Coding_Projects/text");
    Text Database = read_words(Db, 0, Db.size());
    embedder.updateVocabulary(Database);
    printf("Total Vocabulary Size: %i | Total Word Count: %i \n", embedder.Vocabulary.size(), Database.size());
    LLM model(go, embedder, Database, MAX_BATCH_SIZE, context_len, HIDDEN_DIM);
    model.train(20000, 100);
    model.generate({"so","sherlock", "holmes", "said"}, 30);
}


int main()
{
    GraphOperations go;
    const int T = 1000, init_depth = 64, t_hidden = 128, img_size = 64, epochs = 10000;
    int t;
    std::random_device rd;
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, T-1);

    auto base = Bi2n("KPOP", 1, img_size, img_size);
    StandardNorm(base);
    auto input  = go.like(base, "Input Image"), target = go.like(base, "Target Image");

    std::cout << "Building U-NET model \n";
    U_NET model(go, input, target, nullptr, 3, 3, init_depth, t_hidden);
    model.load("best_models/huntrix.bin");

    const bool train = false;
    if (train){
    model.build_train();
    std::cout << "Starting Training Loop for " << epochs << " epochs \n";
    for(int epoch = 0; epoch < epochs; ++epoch)
    {
        model.t = t = dist(gen); 
        prepare(base, input, target, t, T, ((uint64_t)rd() << 32) | rd());
        if (epoch % 5 == 0) go.calculate_loss = true;
        model.train();
        if(epoch % 5 == 0)
        {   
            printf("Epoch %i, Loss: %f at t = %i \n", epoch+1, model.loss, t); 
            go.calculate_loss = false;
        }    
        if (epoch % 100 == 0) model.save("gmodel"+std::to_string(epoch)+".bin");
    }
    go.clear_graph();
    }
    
    input->clear();
    target->clear();
    auto test = std::make_shared<NodeBackProp>("Test", 1, base->dim[1], base->dim[2], base->dim[3],1);
    Noise(test);
    Sampling<U_NET> sampler(model, test, T-1,T);
    sampler.loop(T-1, 0);
    //BPrintImage(base, 512, 512);
    sampler.display(512,512);
    return 0;

}