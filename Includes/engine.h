#pragma once
#include "dataloader.h"
#include "debugging_utils.h"
#include "kernels.h"

struct NodeBackProp;
using graph = std::shared_ptr<NodeBackProp>;
using graph_tree = std::vector<graph>;
using BatchText = std::vector<Text>;
using graphFn = std::function<graph(graph)>;

void isNan(const str name, const float* X, const long long total);

void diffuse(float* input, float* model, float* theta, const long long total, const int t, const int T, const double s, const uint64_t seed);

struct BatchTexts
{
    BatchText encoder;
    BatchText decoder;
    BatchText target;

    BatchTexts(int batch_size,int clen):encoder(batch_size,Text(clen+1)),decoder(batch_size,Text(clen+1)),target(batch_size,Text(clen+1)){}
};

struct NodeBackProp 
{
    str op_name;
    graph_tree inputs;
    float*  output;
    float*  grad;
    int dim[4];
    long long total;
    bool owns_output;
    std::function<void()> forward;
    std::function<void()> backward;
    std::function<void()> free;
    std::function<void()> zero_grad;
    std::function<void()> serious_free;
    std::function<void()> updateParams;
    std::function<void(const double*)> clipnorm;
    std::function<void(double*)> accumulate;
    std::function<void()> printparams;
    virtual void update(const bool W=ADAMW){};
    
    NodeBackProp(str name, int batch, int d1, int d2, int d3, int allocation);
    void clear();
    void reshape(std::vector<int> new_dims);

};

float ReadValueAt(const graph& X, const int& idx, const bool flag = false);

void  WriteValueAt(const graph& X, const float value, const int& idx, const bool flag = false);

struct AdamParameter : public NodeBackProp {
    float *m, *v; int t; 
    float lr, b1, b2, epsilon; 
    long long total_size;
    int batch_size;
    float group_norm;
    float weight_decay;


    AdamParameter(str n, int batch, int out, int in, int row, int col, double norm = NORM);
    
    void save(std::ofstream& f) const;

    void load(std::ifstream& f); // Assumes the class structure is the exact same as when saved, otherwise may cause issues
    
    void update(const bool W = ADAMW) override;

    void accumulate_grad(double* global_scale);

    void gradnorm(const double* global_scale);

};

void Dimension(graph X);
void Dimension(AdamParameter* X);
void Zerograd(AdamParameter* X);
void Zerograd(graph X);
void isNan(graph X, const int type = 0);
void isNan(AdamParameter* X, const int type = 0);
void printGPU(const graph X, const int type = 0);
void printGPU(AdamParameter* X, const int type = 0);
void printHeadGPU(const graph X, const int type = 0);
void printHeadGPU(AdamParameter* X, const int type = 0);
int ArgMaxToCPU(const graph& input, int* X);
int TopKSampleToCPU(const graph& input, int* X, const int k);
void BMinMaxNorm(const graph &X);
void BMaxNorm(const graph & X);
void StandardNorm(const graph &X, const float img_max=255.0f, const float mean=0.5f, const float std=0.5f);
void StandardDeNorm(const graph &X, const float img_max=255.0f, const float mean=0.5f, const float std=0.5f);
void prepare(const graph &base, const graph &input, const graph &target, int t, int T, const uint64_t seed);
graph_tree topological_sort(const graph& root);

class GraphOperations{
public:
    graph_tree nodes;
    double GB = 0;
    bool calculate_loss = false;
    float loss;
    graph like(const graph& X, const str name = "");
    graph Last(const graph& X);
    graph Clamp(const graph& X, const float min, const float max);
    graph Permute(const graph& X, int i0, int i1, int i2, int i3);
    graph Dropout(const graph& X, const float drop_prob, const bool eval = false);
    graph Transpose(const graph& X);
    graph HeadifytoChannel(const graph& X, const int new_channels);
    graph DeHeadify(const graph& X);
    graph PositionalEncoding(const int &t, const int d_model);
    graph MatrixPositionalEncoding(const graph& X, const int start_idx = 0);
    graph Broadcast_Add(const graph& A, const graph& B);
    graph Bias_Add(const graph& A, const graph& B);
    graph Broadcast_Channel(const graph& A, const graph& B);

    graph Scale(const graph& input, const float scale);
    graph Add(const graph& A, const graph& B, const bool last = false);
    graph Multiply(const graph& A, const graph& B, const bool last = false);
    graph Exp(const graph& X);
    graph Log(const graph& X);
    graph Min(const graph& A, const graph& B);
    graph Max(const graph& A, const graph& B);

    graph MeanSquaredError(const graph& prediction, const graph& target, const bool last);
    graph MeanSquaredError(const graph& prediction, const float& target, const int& target_idx, const bool last);
    graph CrossEntropy(const graph& prediction, const graph& target, const bool last);
    graph Entropy(const graph& X);
    graph SoftMaxCrossEntropy(const graph& prediction, const graph& target, const bool last);

    graph BMM(const graph& A, const graph& B); // m x n, n x p = m x p
    graph BMMABT(const graph& A, const graph& B); //  m x n, p x n = m x p
    graph BMMATB(const graph& A, const graph& B); // m x n, m x p = n x p
    graph BMMATBT(const graph& A, const graph& B); // m x n, p x m = n x p

    graph SOFTMAX(const graph& X, const int type = 0); // type 0: row-wise, type 1: column-wise
    graph SOFTMASK(const graph& X, const int type = 0); // type 0: row-wise, type 1: column-wise

    graph GatherAction(const graph& X, const graph& actions);

    graph RELU(const graph& input);
    graph SILU(const graph& input);
    graph TANH(const graph& input);
    graph SIGMOID(const graph& input);
    graph LeakyRELU(const graph& input);

    graph CopyCrop(const graph& input1, const graph& input2);
    graph CopyConcat(const graph& input1, const graph& input2);
    graph VecConcat(const graph_tree& inputs);

    graph LAYERMEAN(const graph& X);

    graph LayerNorm(const graph& X);
    graph BatchNorm(const graph& X);
    graph GroupNorm(const graph& X, const int group=8);
    graph InstanceNorm(const graph & X);

    void clipNorm(double* global_scale);
    void accumulate(double* global_scale); 
    void ParameterUpdate();
    void forward();
    void backward();
    void zero_grad();
    void printNodes(const bool display_grad=false);
    void clear_graph();
    void clean_clear_graph();
};

class Identity{   
private:
    GraphOperations &go;
    str name;
public: 
    Identity(GraphOperations& go_ref, const str name = "");
    graph forward(const graph& X);
};

class Linear
{ 
private: 
    int in, out;
public:
    GraphOperations &go;
    AdamParameter *W1;
    AdamParameter *B1;
    str op_name = "Linear Layer";
    Linear(GraphOperations &go_ref, const int input, const int output, const str name = "");
    graph forward(const graph & X);
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
};

class Convolute2D {
private:
    GraphOperations go;
    graph T_node;
    int inp, out, c, d, pad, stride;
public:
    AdamParameter *weights, *bias;
    str name;
    /**
     * @brief For standard convolution call C2D (go, inp, out);
     * @param name go: GraphOperations reference
     * @param Input: number of input channels
     * @param Output: number of output channels
     * @param C: kernel size row: (default 3)
     * @param D: kernel size col: (default 3)
     * @param stride: stride size (default 1)
     * @param padding: padding size (default 1)
     * @param param: Name of the operation (default "" )
     */
    Convolute2D(GraphOperations&go_ref, int Input, int Output, int C=3, int D=3, int stride = 1, int padding = 1, str param = "");
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    graph forward(const graph& X);
};

class Convolute2DT {
private:
    GraphOperations go;
    graph T_node;
    int inp, out, c, d, pad, stride;
public:
    AdamParameter *weights, *bias;
    str name;
    /**
     * @brief For transposed convolution call C2DT (go, inp, out);
     * @param name go: GraphOperations reference
     * @param Input: number of input channels
     * @param Output: number of output channels
     * @param C: kernel size row: (default 3)
     * @param D: kernel size col: (default 3)
     * @param stride: stride size (default 1)
     * @param padding: padding size (default 1)
     * @param param: Name of the operation (default "" )
     */
    Convolute2DT(GraphOperations&go_ref, int Input, int Output, int C=2, int D=2, int stride=2, int padding=0, str param="");
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    graph forward(const graph& X);
};

class TimeMLPBlock
{

public:
    GraphOperations &go;
    Linear *L0, *L1;
    TimeMLPBlock(GraphOperations &go_ref, const int t_embed_dim, const int t_hidden);
    graph forward(const graph & X);
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
};

void Noise(const graph & input);