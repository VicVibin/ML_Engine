#pragma once
#include "debugging_utils.h"
#include "kernels.h"
#include <iomanip>
#include <functional>
#include <memory>
#include <map>
#include <utility>
#include <random>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <algorithm>

struct NodeBackProp;
using graph = std::shared_ptr<NodeBackProp>;
using graph_tree = std::vector<graph>;
using graphFn = std::function<graph(graph)>;

template<typename numeric>
void isNan(const str name, const numeric* X, const long long total)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (total + tpb-1)/tpb;
    ISNAN<<<bpg,tpb>>>(X, total);
    CheckError(name + "   has nan value");
    return;
}


void diffuse(float* input, float* model, float* theta, const long long total, const int t, const int T, const double s, const uint64_t seed);

struct NodeBackProp 
{
    str op_name;
    graph_tree inputs;
    float*  output;
    float*  grad;
    int dim[4];
    long long total;
    bool forward_called = false;
    std::function<void()> forward;
    std::function<void()> backward;
    std::function<void()> free;
    std::function<void()> zero_grad;
    std::function<void()> serious_free;
    std::function<void(const float lr, const bool adamw)> updateParams;
    std::function<void(const double*)> clipnorm;
    std::function<void(double*)> accumulate;
    std::function<void(const bool full)> printparams;
    virtual void update(const float lr = LEARNING_RATE, const bool W = ADAMW){};
    NodeBackProp(str name, int batch, int d1, int d2, int d3, int allocation);
    void clear();
    void reshape(std::vector<int> new_dims);
    void reshape(int arr[], const int size=4);

};

float ReadValueAt(const graph& X, const int& idx, const bool flag = false);

void  WriteValueAt(const graph& X, const float value, const int& idx, const bool flag = false);

struct AdamParameter : public NodeBackProp {
    float *m, *v; int t; 
    float b1, b2, epsilon; 
    int batch_size;
    float group_norm;
    float weight_decay;


    AdamParameter(str n, int out, int in, int row, int col, double norm = NORM);
    
    void save(std::ofstream& f) const;

    void load(std::ifstream& f); // Assumes the class structure is the exact same as when saved, otherwise may cause issues
    
    void update(const float lr = LEARNING_RATE, const bool W = ADAMW) override;

    void accumulate_grad(double* global_scale);

    void gradnorm(const double* global_scale);

    void operator =(const AdamParameter& other);

};

void Dimension(graph X);
void Dimension(AdamParameter X);
void Zerograd(AdamParameter X);
void Zerograd(graph X);
void isNan(graph X, const int type = 0);
void isNan(AdamParameter X, const int type = 0);
void printGPU(const graph X, const int type = 0);
void printGPU(AdamParameter X, const int type = 0);
void printHeadGPU(const graph X, const int type = 0);
void printHeadGPU(AdamParameter X, const int type = 0);
int ArgMaxToCPU(const graph& input, int* X);
int TopKSampleToCPU(const graph& input, int* X, const int k);
int ArgMaxToCPU(const graph& input);
int TopKSampleToCPU(const graph& input, const int k);
void BMinMaxNorm(const graph &X);
void BMaxNorm(const graph & X);
void StandardNorm(const graph &X, const float img_max=255.0f, const float mean=0.5f, const float std=0.5f);
void StandardDeNorm(const graph &X, const float img_max=255.0f, const float mean=0.5f, const float std=0.5f);
void prepare(const graph &base, const graph &input, const graph &target, int t, int T, const uint64_t seed);
graph_tree topological_sort(const graph& root);

class GraphOperations
{
private:
    static bool calculate_loss;
    void clipNorm(double* global_scale);
    void accumulate(double* global_scale); 

public:
    graph_tree nodes;

    static double GB;
    static float loss;

    void ParameterUpdate(const graph&X = nullptr, const bool show = false, const float lr = LEARNING_RATE, const bool adamw = ADAMW);
    void forward(const graph& X = nullptr, const bool calc_loss = false, const bool show = false, const bool time = false, const bool check_nan = false);
    void backward(const graph&X = nullptr, const bool show = false, const bool time = false, const bool check_nan = false);
    void zero_grad(const graph&X = nullptr, const bool show = false);
    void printParams(const graph&X = nullptr, const bool show = false);
    void printNodes(const graph&X = nullptr, const int show = 0);
    void clear_graph(const graph&X = nullptr, const bool show = false);
    void clean_clear_graph(const graph&X = nullptr, const bool show = false);


    static graph Copy(const graph& X);
    static graph track(const graph_tree& X);
    static graph identity(const graph& X);
    static graph ones_like(const graph& X);
    static graph Constant_like(const graph& X, const float constant);
    // =============== Matrix Operations ============== //
    static graph identity_like(const graph& X);
    static std::pair<graph, graph> LU_factorize(const graph& X);
    static graph like(const graph& X, const str name = "");
    static graph InverseAug(const graph& X);
    static graph Inverse(const graph& X);
    static graph Determinant(const graph& X);
    static std::tuple<graph, graph, graph> Schurr(const graph& X);

    static graph GaussianNoise_like(const graph& X, const float mean, const float std);
    static graph UniformNoise_like(const graph& X, const float mean, const float std);
    static graph NthRow(const graph& X, const int row);
    static graph Clamp(const graph& X, const float min, const float max);
    static graph Permute(const graph& X, int i0, int i1, int i2, int i3);
    static graph Dropout(const graph& X, const float drop_prob, const bool eval = false);
    static graph Transpose(const graph& X);
    static graph HeadifytoChannel(const graph& X, const int new_channels);
    static graph DeHeadify(const graph& X);
    static graph PositionalEncoding(const int &t, const int d_model);
    static graph MatrixPositionalEncoding(const graph& X, const int start_idx = 0);
    static graph Broadcast_Add(const graph& A, const graph& B);
    static graph Bias_Add(const graph& A, const graph& B);
    static graph Broadcast_Channel(const graph& A, const graph& B);

    static graph Scale(const graph& input, const float scale, const bool last = false);
    static graph StandardNorm(const graph& X, const float max = 255.f, const float mean = 0.f, const float std = 1.f);
    static graph RMSNorm(const graph& input, const int type = 0);

    // Matrix functions
    static graph ExpM(const graph& X, const float threshold);

    static graph Add(const graph& A, const graph& B, const bool last = false);
    static graph Subtract(const graph& A, const graph& B, const bool last = false);
    static graph Multiply(const graph& A, const graph& B);
    static graph Exp(const graph& X);
    static graph Log(const graph& X);
    static graph Min(const graph& A, const graph& B);
    static graph Max(const graph& A, const graph& B);

    static graph MeanSquaredError(const graph& prediction, const graph& target, const bool last);
    static graph MeanSquaredError(const graph& prediction, const float* target, const float* target_idx, const bool last);
    static graph NCE(const graph& prediction, const int num_pos, const int num_neg);
    static graph CSInfoNCE(const graph& prediction, const int num_pos, const int num_neg, const float temperature, const bool last);
    static graph CrossEntropy(const graph& prediction, const graph& target, const bool last);
    static graph ContrastLearningTarget(const int batch);

    static graph Entropy(const graph& X, const bool last = false);
    static graph SoftMaxCrossEntropy(const graph& prediction, const graph& target, const bool last);

    static graph BMM(const graph& A, const graph& B); // m x n, n x p = m x p
    static graph BMMABT(const graph& A, const graph& B); //  m x n, p x n = m x p
    static graph BMMATB(const graph& A, const graph& B); // m x n, m x p = n x p
    static graph BMMT(const graph& A, const graph& B); // m x n, p x m = n x p

    static graph SOFTMAX(const graph& X, const int type = 0); // type 0: row-wise, type 1: column-wise
    static graph SOFTMASK(const graph& X, const int type = 0); // type 0: row-wise, type 1: column-wise

    static graph GatherAction(const graph& X, const graph& actions);

    static graph RELU(const graph& input);
    static graph SILU(const graph& input);
    static graph TANH(const graph& input);
    static graph GELU(const graph& input);
    static graph SIGMOID(const graph& input);
    static graph LeakyRELU(const graph& input);

    static graph CopyCrop(const graph& input1, const graph& input2);
    static graph CopyConcat(const graph& input1, const graph& input2);
    static graph VecConcat(const graph_tree& inputs);

    static graph LAYERMEAN(const graph& X);
    static graph BATCHMEAN(const graph& X);

    static graph LayerNorm(const graph& X);
    static graph BatchNorm(const graph& X);
    static graph GroupNorm(const graph& X, const int group=8);
    static graph InstanceNorm(const graph & X);
};

inline graph operator +(const graph &A, const graph &B) {return GraphOperations::Add(A,B);}
inline graph operator -(const graph &A, const graph &B) {return GraphOperations::Subtract(A,B);}
inline graph operator *(const graph &A, const graph &B) {return GraphOperations::Multiply(A,B);}

class Linear
{ 
private: 
    int in, out;
    const bool bias;
public:
    AdamParameter W1;
    AdamParameter B1;
    str op_name = "Linear Layer";
    Linear(const int input, const int output, const str name = "", const bool bias = true);
    graph forward(const graph & X);
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    void operator =(const Linear& other);
};

class Convolute2D {
public:
    int inp, out, c, d, pad, stride;
    str name;
    AdamParameter W1, B1;
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
    Convolute2D(int Input, int Output, int C=3, int D=3, int stride = 1, int padding = 1, str param = "");
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    graph forward(const graph& X);
};

class Convolute2DT {
private:
    graph T_node;
    int inp, out, c, d, pad, stride;
public:
    str name;
    AdamParameter W1, B1;
    
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
    Convolute2DT(int Input, int Output, int C=2, int D=2, int stride=2, int padding=0, str param="");
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    graph forward(const graph& X);
};

class TimeMLPBlock
{
public:
    Linear L0, L1;
    TimeMLPBlock(const int t_embed_dim, const int t_hidden);
    graph forward(const graph & X);
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
};

class Multi_Linear_Residual_Block
{
    /*
    @brief: Required Activation and Normalization layer lambdas with reference capture for activation and normalization.. 
    */
public:
    const int input_dim, output_dim, residuals, hidden_dim, layers;
    std::vector<Linear*> sequence;
    Multi_Linear_Residual_Block(const int input, const int output, const int num_residuals, const int layers, const int hidden_size);
    template<typename ActFn, typename NormFn>
    graph forward(const graph& X, ActFn activation, NormFn norm)
    {
        auto H = sequence[0]->forward(X);
        for (int r = 0; r < residuals; ++r)
        {
            auto A = H; 
            for (int j = 0; j < layers; ++j){
                int idx = r * layers + j + 1; 
                if(idx < residuals * layers) A = activation(sequence[idx]->forward(A));
            } H = norm(H + A);
        }

        return sequence[residuals * layers]->forward(H);
    }

    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
};

void Noise(const graph & input, const float mean = 0.f, const float std = 1.f, const str type = "gaussian");



