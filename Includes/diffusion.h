#pragma once
#include "image_loader.h"
#include <cerrno>
#include <cstring>  
#include <sys/stat.h>
#include <filesystem>
#if defined(_WIN32)
  #include <direct.h> 
#endif

class VisionAttention 
{
public:
    int batch, channels;
    Convolute2D *Q, *K, *V, *P;
    VisionAttention(const int Channels);
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    graph forward(const graph& X_in);

};

class VisionCrossAttention
{
public: 
    int batch, channels, context_len, embed_dim;
    Convolute2D *Q, *K, *V, *P;
    VisionCrossAttention(const int Channels, const int ContextLen, const int EmbedDim);
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    graph forward(const graph& X_in, const graph& Context);

};

class ResidualBlock
{
public:
    Convolute2D *conv1, *conv2, *skipconv;
    Linear *time_mlp;
    int in, out, hidden, stride;

    ResidualBlock(const int in_channels, const int out_channels, const int t_hidden, const int stride=1); 
    void save(std::ofstream& f) const ;
    void load(std::ifstream& f);
    graph forward(const graph& x, const graph & t_emb);
};

class U_NET
{
public:
    GraphOperations& go;
    const graph &input, &target, &text_embed;
    double* global_scale;
    int t_embed_dim, t_hidden, t, batch;

    ResidualBlock *enc1, *enc2, *enc3, *enc4, *dec1, *dec2, *dec3;
    ResidualBlock *b0, *b1;
    Convolute2D *out;
    Convolute2DT *up1, *up2, *up3;
    TimeMLPBlock *time_mlp;
    float loss;
    
    U_NET(GraphOperations& goref,const graph& input, const graph& target, const graph& text_embed, 
    const int in_channels=3,const int out_channels=3, const int init_depth=64,const int t_node=32);
    void save(const str& filename) const;
    void load(const str& filename);
    graph build_train();
    graph build_inference(const graph& test_input);

    void zero_grad();
    void forward();
    void backward();
    void parameterUpdate();
    void printvals(); 
    void printgrads();
    void printparams();  
    void train();
};

template<typename NET>
class DiffusionTrainer
{
public:
    GraphOperations& go;
    NET& model;
    const graph &base, &input, &target;
    graph prediction;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<int> dist;
    const int T;

    DiffusionTrainer(GraphOperations& go_ref, NET& model_ref, const graph &base_ref, const graph& input_ref, const graph& target_ref, const int T): 
    go(go_ref), model(model_ref), base(base_ref), input(input_ref), target(target_ref), gen(rd()), dist(0, T-1), T(T)
    {   
        std::cout << "Building Training Graph for Diffusion Model \n";
        prediction = model.build_train();
    }
    
    void train(const int epochs, const int show_every = 100, const int save_every = 1000, const str& save_path = "")
    {   
        for(int epoch = 0; epoch < epochs; ++epoch)
        {
            int t = dist(gen); model.t = t;
            const uint64_t seed = ((uint64_t)rd() << 32) | rd();
            prepare(base, input, target, t, T, seed);
            model.train();
            if (epoch % show_every == 0) {printf("Epoch %i/%i, Loss: %f at t = %i \n", epoch+1, epochs, model.loss, model.t);}
            if (epoch % save_every == 0 && !save_path.empty()) model.save(save_path);
        }
        go.clear_graph(prediction, true);
    }


};

template<typename NET>
class Sampling{
public:
    NET &model;
    int T, t;
    const graph &input;
    graph u_theta;
    graph prediction;
    std::random_device rd;

    Sampling(NET& trained_model, const graph &noisy_image, int T_in,  int Big_T_in): 
    model(trained_model), input(noisy_image), t(T_in) ,T(Big_T_in)
    {   
        std::cout << "Building Inference Model for Langevin Sampling \n";
        prediction = model.build_inference(input);
        std::cout << "Final loss before sampling: " << model.loss << "\n";
        u_theta = GraphOperations::like(noisy_image, "U0(X_t,t)");
        if(input->dim[0] != 1) std::cout << "Multi Image inference for Langevin.... \n"; 
    
    }
    
    void forward(double s = 0.008)
    {   
        model.t = t;
        model.zero_grad();
        model.forward();
        const uint64_t seed =  ((uint64_t)rd() << 32) | rd();
        diffuse(input->output, prediction->output, u_theta->output, input->total, t, T, s, seed);--t;

    }

    void display(const str &filename, const int delay_s, const int row=0, const int col=0)
    {   
        StandardDeNorm(input);
        cv::Mat img = n2i(input);
        if (row > 0 && col > 0) cv::resize(img, img, cv::Size(col, row), 0, 0, cv::INTER_AREA);
        cv::imwrite(filename, img);
        cv::imshow("Final Sample", img);
        cv::waitKey(delay_s * 1e3);
    }

    void loop(const int start, const int till = 0, const int show_every = 100)
    {
        for(int i = start; i > till; --i)
        {
            if (i % show_every == 0) printf("Step: %i/%i \n", i, T);
            forward();
        }
    }
};



/*
int main()
{
    GraphOperations go; 
    const int T = 1000, init_depth = 32, t_hidden = 128, img_size = 128, epochs = 30000;
    auto base = Bi2n("KPOP", 1, img_size, img_size); // Loads a batch of images and converts to a node 
    StandardNorm(base); // Normalizes using (X/255 - 0.5) / 0.5
    auto input  = go.like(base, "Input Image"), target = go.like(base,"Target Image");
    U_NET model(go,input,target,nullptr,3,3,init_depth,t_hidden); 
    model.load("best_models/2huntrix.bin");
    DiffusionTrainer<U_NET> trainer(go, model, base, input, target, T);
    const bool train = false;
    if (train)
    {model.build_train();
        for(int epoch = 0; epoch < epochs; ++epoch)
        {
            model.t = t = dist(gen); 
            prepare(base, input, target, t, T, ((uint64_t)rd() << 32) | rd());
            model.train();
            if (epoch % 10 == 0) {printf("Epoch %i, Loss: %f at t = %i \n", epoch+1, model.loss, t);}
        }
        model.save("unet_model.bin");
        go.clear_graph();
    }

    auto test = go.like(base, "Test Image");
    Noise(test);
    Sampling<U_NET> sampler(model, test, T-1,T);
    BPrintImage(test);
    sampler.loop(T-1, 0);
    sampler.display(512,512);
    return 0;
}

*/