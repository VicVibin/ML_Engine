#include "engine.h"
#include <chrono>


template <typename T>
__global__ void ApplyFunction(const float* input, T func, float* output, const long long size)
{
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size){return;}
    output[idx] = func(input[idx]);
}

__global__ void rbf_kernel(const float* A, const float* B, float* output, const float sigma_squared, const float length_squared, const int a_size, const int b_size)
{
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= a_size * b_size){return;}
    const int i = idx / b_size;
    const int j = idx % b_size;
    const float mag = -0.5f * (A[i] - B[j]) * (A[i] - B[j]);
    output[i * b_size + j] = sigma_squared * expf(mag / length_squared);
}

__global__ void acquisition_function_expected_improvement(const float* mean, const float* var, float* output, const int B, const int C, const int H, const bool maximizing)
{
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long size = (long long)B * C * H * (long long)H;
    if(idx >= size){return;}
    const int b = idx / (C * H * H);
    const int c = (idx / (H * H)) % C;
    const int i = (idx / H) % H;
    const int j = idx % H;
    const float curr_best = mean[b*C*H+c*H+i];
    const float curr_var = sqrtf(var[b*C*H*H+c*H*H+i*H+i]);
    const float curr_mean = mean[b*C*H+c*H+j];
    const float Z_score = maximizing ? (curr_mean - curr_best)/curr_var : (curr_best - curr_mean)/curr_var;
    const float norm_cdf = 0.5f * (1.0f + erff(Z_score / 1.41421356237f)); // sqrt(2) = 1.41421356237
    const float norm_pdf = expf(-0.5f * Z_score * Z_score) / 2.50662827463f; // sqrt(2*pi) = 2.50662827463
    // EI(i, j) = (mean[i] - mean[j]) * norm_cdf(Z) + std[i] * norm_pdf(Z); curr_best and mean is swapped for minimzing
    output[idx] = maximizing ? (curr_mean - curr_best) * norm_cdf + curr_var * norm_pdf : (curr_best - curr_mean) * norm_cdf + curr_var * norm_pdf;

}

graph RBF_kernel(const graph& A, const graph& B, const float sigma_squared, const float length_squared)
{
    if (A->dim[2] != 1 || B->dim[2] != 1){printf("RBF kernel only supports 1D inputs"); std::exit(1);}
    const int a_size = A->dim[3];
    const int b_size = B->dim[3];
    auto node = std::make_shared<NodeBackProp>("RBF Kernel of " + A->op_name + " and " + B->op_name, 1,1,a_size,b_size,1);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    node->inputs = {A, B};
    node->forward = [=]()
    {
        rbf_kernel<<<bpg, tpb>>>(A->output, B->output, node->output, sigma_squared, length_squared, a_size, b_size);
        CheckError("RBF Kernel forward");
    };
    node->free = [=](){node->clear();};
    return node;
}

std::pair<graph, graph> GP_Posterior(const graph& X, const graph& candidates, const graph& Y, const float sigma_squared, const float length_squared)
{
    auto x_sx = RBF_kernel(candidates, X, sigma_squared, length_squared); // N x M
    Dimension(x_sx);
    auto xx_inv = GraphOperations::Inverse(RBF_kernel(X, X, sigma_squared, length_squared)); // N X N
    Dimension(xx_inv);
    auto x_ss = RBF_kernel(candidates, candidates, sigma_squared, length_squared); // M x M
    Dimension(x_ss);
    auto x_xs = RBF_kernel(X, candidates, sigma_squared, length_squared);
    Dimension(x_xs);
    auto mean = GraphOperations::BMM(x_sx, GraphOperations::BMM(xx_inv, GraphOperations::Transpose(Y)));
    Dimension(mean);
    auto var = GraphOperations::Subtract(x_ss, GraphOperations::BMM(x_sx, GraphOperations::BMM(xx_inv,x_xs)));
    Dimension(var);
    return {mean,var};
}

std::pair<graph, int*> EI(const graph& mean, const graph& var, const str optimizing_for = "min")
{
    const int B = mean->dim[0], C = mean->dim[1], H = mean->dim[2];
    auto node = std::make_shared<NodeBackProp>("Expected Improvement", B, C, H, H, 1);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    bool optimizing_for_flag = (optimizing_for == "min") ? 0 : 1;
    int* output;
    SafeCudaMalloc("output for EI", output, B*C);
    node->inputs = {mean, var};
    node->forward = [=]()
    {
        acquisition_function_expected_improvement<<<bpg, tpb>>>(mean->output, var->output, node->output, B, C, H, optimizing_for_flag);
        ArgMax<<<(B*C+tpb-1)/tpb, tpb>>>(node->output, output, B*C);
        CheckError("Acquisition function forward");
    };
    node->free = [=]()
    {
        node->clear();
        cudaFree(output);
    };
    return {node, output};

}

struct Function
{
    __host__ __device__ float operator()(const float x) const
    {
        return -4.0f * x * x * x + 3.0f * x * x + 2.0f * x + 4.0f;
    }

};

int main() 
{
    std::cout << "Running Bayesian Optimization Example... \n";
    GraphOperations go;
    auto X = std::make_shared<NodeBackProp>("X", 1,1,1,3,1);
    auto X_star = std::make_shared<NodeBackProp>("X_star", 1,1,1,11,1);
    std::cout << "Applying function to X to get Y... \n";
    WriteValueAt(X, 0.0f, 0); WriteValueAt(X, 0.3f, 1); WriteValueAt(X, 0.72f, 2);
    auto Y = GraphOperations::like(X, "Y");
    Function F;
    Y->forward = [=]()
    {
        ApplyFunction<<<1,3>>>(X->output, F, Y->output, 3);
        CheckError("Applying function to get Y");
    };
    
    std::vector<float> X_star_values = {0.2, 0.6, 0.013, 0.0135, 0.64, 0.11, 0.32, 0.415, 0.53, 0.65, 0.214};
    for (size_t i = 0; i < X_star_values.size(); ++i) {WriteValueAt(X_star, X_star_values[i], i); }
    std::cout << "Calculating GP Posterior... \n";
    auto [mean, var] = GP_Posterior(X, X_star, Y, 1.0f, 0.1f);
    auto [ei, ei_indices] = EI(mean, var, "min");
    Timing A("Posterior and improvement");
    A.start();
    go.forward(ei, false, true, true);
    A.end();
    int idx; cudaMemcpy(&idx, ei_indices, sizeof(int), cudaMemcpyDeviceToHost);
    const int ei_row = idx / X_star->dim[3];
    const int ei_col = idx % X_star->dim[3];
    printf("Best candidate: %f with EI: %f, mean: %f, var: %f\n", ReadValueAt(X_star, idx), 
                        ReadValueAt(ei, idx), ReadValueAt(mean, ei_col), ReadValueAt(var, ei_row*var->dim[3]+ei_row));
    go.clean_clear_graph(ei);
    X->clear(); X_star->clear(); // go.like already has a clear function;


    return 0;

};


