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
    GraphOperations &go; 
    int batch, channels;
    Convolute2D *Q, *K, *V, *P;
    VisionAttention(GraphOperations &go_ref, const int Channels);
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    graph forward(const graph& X_in);

};

class VisionCrossAttention
{
public:
    GraphOperations &go; 
    int batch, channels, context_len, embed_dim;
    Convolute2D *Q, *K, *V, *P;
    VisionCrossAttention(GraphOperations &go_ref, const int Channels, const int ContextLen, const int EmbedDim);
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    graph forward(const graph& X_in, const graph& Context);

};

class ResidualBlock
{
public:
    GraphOperations &go;
    Convolute2D *conv1, *conv2, *skipconv;
    Identity *skip;
    Linear *time_mlp;
    int in, out, hidden, stride;

    ResidualBlock(GraphOperations &go_ref, const int in_channels, const int out_channels, const int t_hidden, const int stride=1); 
    void save(std::ofstream& f) const ;
    void load(std::ifstream& f);
    graph forward(const graph& x, const graph & t_emb);
};

class LinearBlock
{
public:
    GraphOperations &go;
    Linear *conv1, *conv2, *skipconv;
    Identity *skip;
    Linear *time_mlp;
    int in, out, hidden, stride;

    LinearBlock(GraphOperations &go_ref, const int in_channels, const int out_channels, const int t_hidden, const int stride=1); 
    void save(std::ofstream& f) const ;
    void load(std::ifstream& f);
    graph forward(const graph& x, const graph & t_emb);
};

class L_NET
{
public:
    GraphOperations &go;
    const graph &input, &target, &text_embed;
    graph prediction;
    double* global_scale;
    int t_embed_dim, t_hidden, t, batch;

    LinearBlock *enc1, *enc2, *enc3, *enc4, *dec1, *dec2, *dec3;
    LinearBlock *b0, *b1;
    Linear *out;
    Linear *up1, *up2, *up3;
    TimeMLPBlock *time_mlp;
    float loss;
    
    L_NET(GraphOperations& goref,const graph& input, const graph& target, const graph& text_embed, 
    const int in_channels=3,const int out_channels=3, const int init_depth=64,const int t_node=32);
    void save(const str& filename) const;
    void load(const str& filename);
    void build_train();
    void build_inference(const graph& test_input);

    void zero_grad();
    void forward();
    void backward();
    void parameterUpdate();
    void printvals(); 
    void printgrads();
    void printparams();  
    void train();
};

class U_NET
{
public:
    GraphOperations &go;
    const graph &input, &target, &text_embed;
    graph prediction;
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
    void build_train();
    void build_inference(const graph& test_input);

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
class Sampling{
public:
    NET &model;
    int T, t;
    const graph &input;
    graph u_theta;
    std::random_device rd;

    Sampling(NET& trained_model, const graph &noisy_image, int T_in,  int Big_T_in): 
    model(trained_model), input(noisy_image), t(T_in) ,T(Big_T_in)
    {   
        std::cout << "Building Inference Model for Langevin Sampling \n";
        model.build_inference(input);
        std::cout << "Final loss before sampling: " << model.loss << "\n";
        model.prediction->op_name = "Model";
        const int a = input->dim[0], b = input->dim[1], c = input->dim[2], d = input->dim[3];
        u_theta = std::make_shared<NodeBackProp>("U0(X_t,t)",a,b,c,d,1);
        if(input->dim[0] != 1) std::cout << "Multi Image inference for Langevin.... \n"; 
    
    }
    
    void forward(double s = 0.008)
    {   
        model.t = t;
        model.zero_grad();
        model.forward();
        const uint64_t seed =  ((uint64_t)rd() << 32) | rd();
        diffuse(input->output, model.prediction->output, u_theta->output, input->total, t, T, s, seed);--t;

    }

    void display(const int row=0, const int col=0)
    {   
        StandardDeNorm(input);
        cv::Mat img = n2i(input);
        if (row > 0 && col > 0) cv::resize(img, img, cv::Size(col, row), 0, 0, cv::INTER_AREA);
        cv::imwrite("sample2.png", img);
        cv::imshow("Final Sample", img);
        cv::waitKey(0);
    }

    void loop(const int start, const int till = 0, const int par = 100)
    {
        for(int i = start; i > till; --i)
        {
            if (i % par == 0) printf("Step: %i/%i \n", i, T);
            forward();
        }}
};
