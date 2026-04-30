#include "includes/engine.h"

float ReadValueAt(const graph& X, const int& idx, const bool grad)
{
    float* out_ptr = (float*)malloc(sizeof(float));
    if(!grad) cudaMemcpy(out_ptr, X->output + idx, sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(out_ptr, X->grad + idx, sizeof(float), cudaMemcpyDeviceToHost);
    float out = out_ptr[0];
    free(out_ptr);
    return out;
}

void WriteValueAt(const graph& X, const float value, const int& idx, const bool grad)
{
    // @brief: Writes to a graph output for flag 0, and 1 for grad
    if (!grad) cudaMemcpy(X->output + idx, &value, sizeof(float), cudaMemcpyHostToDevice);
    else cudaMemcpy(X->grad + idx, &value, sizeof(float), cudaMemcpyHostToDevice);
}

void isNan(const str name, const float* X, const long long total)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (total + tpb-1)/tpb;
    ISNAN<<<bpg,tpb>>>(X, total);
    CheckError(name + "   has nan value");
    return;
}

void diffuse(float* input, float* model, float* theta, const long long total, const int t, const int T, const double s, const uint64_t seed)
{
    const double t_scale = ((double)t / T) + s;
    const double t_scale_1 = ((double)(t-1.0) / T) + s;
    const double cost_t = cos((t_scale/(1.0+s))*PIBY2);
    const double cost_t_1 = cos((t_scale_1/(1.0+s))*PIBY2);
    const double cost_t_b = cos((s/(1.0 + s))*PIBY2);
    const double alpha_hat = (cost_t*cost_t)/(cost_t_b*cost_t_b); 
    const double alpha_hat_1 = (cost_t_1*cost_t_1)/(cost_t_b*cost_t_b); 
    const double alpha_t = alpha_hat / alpha_hat_1; //  a_t
    const double beta_t = 1.0 - alpha_t;  // b_t
    const double scale_out = 1.0 / sqrt(alpha_t);
    const double scale_mean = beta_t / sqrt(1.0-alpha_hat);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (tpb + total-1) / tpb;
    const double sqrt_beta = sqrt(beta_t);
    ScaleAdd<<<bpg, tpb>>>(input, model, theta, -scale_mean, total);
    //CheckError("Scale Subtract in Diffuse u_0 = X_t - (B_t / sqrt(1-a_hat))*e_0(X_t,t)");

    ScaleValue(theta, scale_out, total);
    //CheckError(" Scale in Diffuse 1/sqrt(a_t)*u_0");

    if(t > 1) ReplaceNoise<<<bpg, tpb>>>(input, theta, sqrt_beta, total, seed);
    else cudaMemcpy(input, theta, total*sizeof(float), cudaMemcpyDeviceToDevice);
    //CheckError(" X_t = U_0(t) + sqrt(b_t)*N(0,1)");
}

NodeBackProp::NodeBackProp(str name, int batch, int d1, int d2, int d3, int allocation) : op_name(name)
{
        const int tpb = THREADSPERBLOCK;
        dim[0] = batch;
        dim[1] = d1;
        dim[2] = d2;
        dim[3] = d3;
        total = batch * d1 * d2 * d3;
        
        if(allocation == 1)
        {

            const int bpg = (total+tpb-1)/tpb;
            SafeCudaMalloc(op_name, output, total);
            SafeCudaMalloc(op_name, grad, total);
            CheckError("Scale initialization");

            fillKernel<<<bpg,tpb>>>(grad,0.0f, total);
            //CheckError("Fill Kernel for Gradient");
            
        }

        else
        {
            output = nullptr;
            grad = nullptr;
        }
}

void NodeBackProp::clear()
{
    cudaFree(output);
    cudaFree(grad);
    inputs.clear();
    op_name.clear();
}

void NodeBackProp::reshape(std::vector<int> new_dims)
{
    if(new_dims.size() !=4)
    {
        std::cerr << "Reshape only supports 4D tensors\n";
        std::exit(1);
    }
    
    int new_total = new_dims[0]*new_dims[1]*new_dims[2]*new_dims[3];
    
    if(new_total != total)
    {std::cerr << "Reshape cannot change total size of tensor\n";
    std::exit(1);}
    for(int i=0;i<4;++i){dim[i] = new_dims[i];}
}

AdamParameter::AdamParameter(str n, int batch, int out, int in, int row, int col, double norm) : NodeBackProp(n, out, in, row, col,1), lr(LEARNING_RATE), t(1), b1(0.9), b2(0.999), epsilon(1e-8), batch_size(batch), group_norm(norm), weight_decay(0.01f)
{
        std::random_device rd;
        const uint64_t seed =  ((uint64_t)rd() << 32) | rd();
        total_size = out*in*row*col;
        SafeCudaMalloc("M-matrix of AdamParameter",m,total_size);
        SafeCudaMalloc("V-matrix of AdamParameter",v,total_size);
        const int tpb = THREADSPERBLOCK; 
        const int bpg = (total_size+tpb-1) / tpb;
        Standard_Weights<<<bpg,tpb>>>(output, total_size, sqrtf(XAVIER/(in*row*col)), seed); 
        fillKernel<<<bpg,tpb>>>(m,0.0f,total_size);
        fillKernel<<<bpg,tpb>>>(v,0.0f,total_size);
};

void AdamParameter::save(std::ofstream& f) const
{
    uint32_t name_len = (uint32_t)op_name.size();
    f.write(reinterpret_cast<const char*>(&name_len), sizeof(uint32_t));
    f.write(op_name.data(), name_len);
    if (!f)
    {
        std::cerr << "File write failed saving parameter name: " << op_name << "\n";
        std::exit(1);
    }

    f.write(reinterpret_cast<const char*>(&total), sizeof(int));
    if (!f)
    {
        std::cerr << "File write failed saving parameter size: " << op_name << "\n";
        std::exit(1);
    }

    std::vector<float> host(total);

    // Save weights
    cudaMemcpy(host.data(), output, total * sizeof(float), cudaMemcpyDeviceToHost);
    CheckError("cudaMemcpy DeviceToHost weights in save: " + op_name);
    f.write(reinterpret_cast<const char*>(host.data()), total * sizeof(float));
    if (!f)
    {
        std::cerr << "File write failed saving weights: " << op_name << "\n";
        std::exit(1);
    }

    // Save Adam m moment
    cudaMemcpy(host.data(), m, total * sizeof(float), cudaMemcpyDeviceToHost);
    CheckError("cudaMemcpy DeviceToHost m in save: " + op_name);
    f.write(reinterpret_cast<const char*>(host.data()), total * sizeof(float));
    if (!f)
    {
        std::cerr << "File write failed saving m moment: " << op_name << "\n";
        std::exit(1);
    }

    // Save Adam v moment
    cudaMemcpy(host.data(), v, total * sizeof(float), cudaMemcpyDeviceToHost);
    CheckError("cudaMemcpy DeviceToHost v in save: " + op_name);
    f.write(reinterpret_cast<const char*>(host.data()), total * sizeof(float));
    if (!f)
    {
        std::cerr << "File write failed saving v moment: " << op_name << "\n";
        std::exit(1);
    }
}

void AdamParameter::load(std::ifstream& f)
{
    uint32_t name_len;
    f.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
    if (!f)
    {
        std::cerr << "File read failed reading name length for: " << op_name << "\n";
        std::exit(1);
    }

    std::string loaded_name(name_len, '\0');
    f.read(const_cast<char*>(loaded_name.data()), name_len);
    if (!f)
    {
        std::cerr << "File read failed reading name for: " << op_name << "\n";
        std::exit(1);
    }

    if (loaded_name != op_name)
    {
        std::cerr << "Parameter name mismatch: expected '" << op_name
                  << "' but got '" << loaded_name << "'\n";
        std::exit(1);
    }

    // Read and validate size
    int loaded_total;
    f.read(reinterpret_cast<char*>(&loaded_total), sizeof(int));
    if (!f)
    {
        std::cerr << "File read failed reading size for: " << op_name << "\n";
        std::exit(1);
    }

    if (loaded_total != total)
    {
        std::cerr << "Parameter size mismatch for '" << op_name << "': expected " << total << " but loaded " << loaded_total << "\n";
        std::exit(1);
        // Skip past this parameter's 3 buffers (weights, m, v) to keep file cursor aligned
        f.seekg((long long)loaded_total * sizeof(float) * 3, std::ios::cur);
        if (!f)
        {
            std::cerr << "File seek failed after size mismatch for: " << op_name << "\n";
            std::exit(1);
        }
        return;
    }

    std::vector<float> host(total);

    // Load weights
    f.read(reinterpret_cast<char*>(host.data()), total * sizeof(float));
    if (!f)
    {
        std::cerr << "File read failed loading weights: " << op_name << "\n";
        std::exit(1);
    }
    cudaMemcpy(output, host.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    CheckError("cudaMemcpy HostToDevice weights in load: " + op_name);

    // Load Adam m moment
    f.read(reinterpret_cast<char*>(host.data()), total * sizeof(float));
    if (!f)
    {
        std::cerr << "File read failed loading m moment: " << op_name << "\n";
        std::exit(1);
    }
    cudaMemcpy(m, host.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    CheckError("cudaMemcpy HostToDevice m in load: " + op_name);

    // Load Adam v moment
    f.read(reinterpret_cast<char*>(host.data()), total * sizeof(float));
    if (!f)
    {
        std::cerr << "File read failed loading v moment: " << op_name << "\n";
        std::exit(1);
    }
    cudaMemcpy(v, host.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    CheckError("cudaMemcpy HostToDevice v in load: " + op_name);
}

void AdamParameter::update(const bool W)
{   
        const int tpb = THREADSPERBLOCK;
        const int bpg = (total_size + tpb - 1) / tpb;
        //isNan("Gradient of " + op_name, grad, total_size);
        if(W) AdamWUpdate<<<bpg, tpb>>>(output, grad, total_size, t, m, v, b1, b2,epsilon,weight_decay,lr);
        else  AdamUpdate<<<bpg, tpb>>>(output, grad, total_size,t, m, v, b1, b2, epsilon, lr);
        //CheckError("AdamUpdate in AdamParameter update");
        t++;
};

void AdamParameter::accumulate_grad(double* global_scale)
{
        const int tpb = THREADSPERBLOCK;
        const int bpg = (total_size + tpb - 1) / tpb;
        SumSquared<<<bpg, tpb>>>(global_scale, grad, total_size);
        //CheckError("SSWarp in Gradient Norm of " + op_name); 
};

void AdamParameter::gradnorm(const double* global_scale){ScalePtr(grad, global_scale, total_size, 1);};

void Dimension(graph X)
{   std::cout<< "Dimensions for node: " << X->op_name << "\n";
    std::cout << "(";
    for(int i=0;i<4; ++i)
    {
        std::cout << " x " << X->dim[i]<<"\t";
    }
    std::cout<< ") \n";
}

void Dimension(AdamParameter* X)
{   std::cout<< "Dimensions for node: " << X->op_name << "\n";
    std::cout << "(";
    for(int i=0;i<4; ++i)
    {
        std::cout << " x " << X->dim[i]<<"\t";
    }
    std::cout<< ") \n";
}

void Zerograd(AdamParameter* X)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (tpb + X->total_size-1) / tpb;
    fillKernel<<<bpg,tpb>>>(X->grad, 0.0f, X->total_size);

}

void Zerograd(graph X)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (tpb + X->total-1) / tpb;
    fillKernel<<<bpg,tpb>>>(X->grad, 0.0f, X->total);

}

void isNan(graph X, const int type )
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (X->total + tpb-1)/tpb;
    if (type == 0) ISNAN<<<bpg,tpb>>>(X->output, X->total);
    else ISNAN<<<bpg,tpb>>>(X->grad, X->total);

    CheckError(X->op_name + " has nan value");
    return;
}

void isNan(AdamParameter* X, const int type )
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (X->total + tpb-1)/tpb;
    if (type == 0) ISNAN<<<bpg,tpb>>>(X->output, X->total);
    else ISNAN<<<bpg,tpb>>>(X->grad, X->total);

    CheckError(X->op_name + " has nan value");
    return;
}

void printGPU(const graph X, const int type)
{
    int batch = X->dim[0];
    int ch     = X->dim[1];
    int rows   = X->dim[2];
    int cols   = X->dim[3];
    int total  = X->total;

    float *CPU = (float *)malloc(total * sizeof(float));

    if (type == 0) cudaMemcpy(CPU, X->output, total * sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(CPU, X->grad, total * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (type == 0) std::cout << "Printing dimensions for node " <<X->op_name << "->output \n";
    else std::cout << "Printing dimensions for node " <<X->op_name << "->grad \n";
    for(int b = 0; b < batch; ++b)
    {
    std::cout << "_____________________________________ \n";
    std::cout << " -----BATCH " << b << "-----  \n";

        for (int c = 0; c < ch; ++c){
        std::cout << "Channel " << c << ":\n";
        for (int r = 0; r < rows; ++r){
        for (int col = 0; col < cols; ++col){
        int idx = (b*ch*rows*cols) + (c * rows * cols) + (r * cols) + col;
        std::cout << CPU[idx] << "\t";
        }
            std::cout << "\n";
        }
        std::cout << "------------------------------\n";
    }
    std::cout << "_____________________________________ \n \n \n";
    }

    free(CPU);
}

void printGPU(AdamParameter* X, const int type)
{
     int batch = X->dim[0];
    int ch     = X->dim[1];
    int rows   = X->dim[2];
    int cols   = X->dim[3];
    int total  = X->total;

    float *CPU = (float *)malloc(total * sizeof(float));

    if (type == 0) cudaMemcpy(CPU, X->output, total * sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(CPU, X->grad, total * sizeof(float), cudaMemcpyDeviceToHost);

    
    if (type == 0) std::cout << "Printing dimensions for node " <<X->op_name << "->output \n";
    else std::cout << "Printing dimensions for node " <<X->op_name << "->grad \n";
    for(int b = 0; b < batch; ++b)
    {
    std::cout << "_____________________________________ \n";
    std::cout << " -----BATCH " << b << "-----  \n";

        for (int c = 0; c < ch; ++c){
        std::cout << "Channel " << c << ":\n";
        for (int r = 0; r < rows; ++r){
        for (int col = 0; col < cols; ++col){
        int idx = (b*ch*rows*cols) + (c * rows * cols) + (r * cols) + col;
        std::cout << CPU[idx] << "\t";
        }
            std::cout << "\n";
        }
        std::cout << "------------------------------\n";
    }
    std::cout << "_____________________________________ \n \n \n";
    }

    free(CPU);
}

void printHeadGPU(const graph X, const int type)
{
    int ch     = X->dim[1];
    int rows   = X->dim[2];
    int cols   = X->dim[3];
    int total  = X->total;

    float *CPU = (float *)malloc(total * sizeof(float));
    if (type == 0) cudaMemcpy(CPU, X->output, total * sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(CPU, X->grad, total * sizeof(float), cudaMemcpyDeviceToHost);

    if (type == 0) std::cout << "Printing dimensions for node " <<X->op_name << "->output \n";
    else std::cout << "Printing dimensions for node " <<X->op_name << "->grad \n";
    for (int c = 0; c < min(3,ch); ++c)
    {
        std::cout << "Channel " << c << ":\n";
        for (int r = 0; r < min(5,rows); ++r)
        {
            for (int col = 0; col < min(5,cols); ++col)
            {
                int idx = (c * rows * cols) + (r * cols) + col;
                std::cout << CPU[idx] << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "------------------------------\n";
    }

    free(CPU);
}

void printHeadGPU(AdamParameter* X, const int type)
{
    int ch     = X->dim[1];
    int rows   = X->dim[2];
    int cols   = X->dim[3];
    int total  = X->total;

    float *CPU = (float *)malloc(total * sizeof(float));
    if (type == 0) cudaMemcpy(CPU, X->output, total * sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(CPU, X->grad, total * sizeof(float), cudaMemcpyDeviceToHost);

    if (type == 0) std::cout << "Printing dimensions for node " <<X->op_name << "->output \n";
    else std::cout << "Printing dimensions for node " <<X->op_name << "->grad \n";
    for (int c = 0; c < min(3,ch); ++c)
    {
        std::cout << "Channel " << c << ":\n";
        for (int r = 0; r < min(5, rows); ++r)
        {
            for (int col = 0; col < min(5, cols); ++col)
            {
                int idx = (c * rows * cols) + (r * cols) + col;
                std::cout << CPU[idx] << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "------------------------------\n";
    }

    free(CPU);
}

void BMinMaxNorm(const graph &X)
{
    /*
    @brief: Performs Batch Min-Max Normalization on input X for printing purposes. 
    This kernel computes the minimum and maximum values for each batch, normalizes
    the data to the range [0, 255], and prepares it for visualization. 
    It is designed to handle 4D tensors with dimensions (batch_size, channels, height, width) 
    and is optimized for GPU execution.
    */

    const int tpb = THREADSPERBLOCK;
    const int batch_size = X->dim[0];
    const int elements_per_batch = X->total / batch_size;
    
    float* G_max;
    float* G_min;
    SafeCudaMalloc("GPU batch maximum", G_max, batch_size);
    SafeCudaMalloc("GPU batch minimum", G_min, batch_size);

    fillKernel<<<(batch_size + tpb - 1) / tpb, tpb>>>(G_max, -FLT_MAX, batch_size);
    fillKernel<<<(batch_size + tpb - 1) / tpb, tpb>>>(G_min, FLT_MAX, batch_size);

    const int bpg_minmax = (batch_size * X->dim[1] + tpb - 1) / tpb;
    BMax<<<bpg_minmax, tpb>>>(X->output, G_max, batch_size, X->dim[1], elements_per_batch / X->dim[1]);
    BMin<<<bpg_minmax, tpb>>>(X->output, G_min, batch_size, X->dim[1], elements_per_batch / X->dim[1]);
    
    const int bpg_norm = (X->total + tpb - 1) / tpb;
    BatchMinMaxNorm<<<bpg_norm, tpb>>>(X->output, G_max, G_min, batch_size, X->total);
    BatchMinMaxDeNorm<<<bpg_norm, tpb>>>(X->output, 255.0f, 0.0f, batch_size, X->total);
    
    cudaFree(G_max);
    cudaFree(G_min);
    //CheckError("BatchMinMaxNorm Kernel");
}

void BMaxNorm(const graph & X)
{
    float *max, *min;
    const int tpb = THREADSPERBLOCK;
    SafeCudaMalloc("Max", max, X->dim[0]);
    SafeCudaMalloc("Min", min, X->dim[0]);
    fillKernel<<<(X->dim[0]+tpb-1)/tpb,tpb>>>(max,255.0f,X->dim[0]);
    fillKernel<<<(X->dim[0]+tpb-1)/tpb,tpb>>>(min,  0.0f,X->dim[0]);
    BatchMinMaxNorm<<<(X->total+tpb-1)/tpb,tpb>>>(X->output, max,min,X->dim[0], X->total); 
    cudaFree(max);
    cudaFree(min);

}

void StandardNorm(const graph &X, const float img_max, const float mean, const float std)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (X->total + tpb - 1) / tpb;
    StdNorm<<<bpg, tpb>>>(X->output, img_max, mean, std, X->total);
    //CheckError("StandardNorm kernel");
}

void StandardDeNorm(const graph &X, const float img_max, const float mean, const float std)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (X->total + tpb - 1) / tpb;
    StdDeNorm<<<bpg, tpb>>>(X->output, img_max, mean, std, X->total);
    //CheckError("StandardDeNorm kernel");
}

void prepare(const graph &base, const graph &input, const graph &target, int t, int T, const uint64_t seed)
{
    if (input->total != target->total)
    {
        std::cout << "SHAPE MISMATCH IN PREPARATION \n";
        Dimension(input); Dimension(target);
        exit(1);
    }
    const int tpb = THREADSPERBLOCK;
    const int bpg = (input->total+tpb-1)/tpb;

    cudaMemcpy(input->output, base->output, input->total*sizeof(float), cudaMemcpyDeviceToDevice);
    //CheckError("CudaMemcpy");

    GaussianNoise<<<bpg,tpb>>>(target->output, target->total, seed);
    //CheckError("Gaussian Noise in preparation");

    AddNoise<<<bpg, tpb>>>(input->output, target->output,t, T, input->total);
    //CheckError("Addition of Noise in preparation");
}

int ArgMaxToCPU(const graph& input, int* X)
{
    for(int i = 0; i < 3; i++)
    {if (input->dim[i] != 1)
        {
            printf("Dimension does not match kernel"); 
            Dimension(input);
            std::exit(1);
        }
    }
    ArgMax<<<1,1>>>(input->output, X, input->total);
    int max_id;
    cudaMemcpy(&max_id, X, sizeof(int), cudaMemcpyDeviceToHost);
    return max_id;
}

int TopKSampleToCPU(const graph& input, int* X, const int k)
{
    for (int i = 0; i < 3; ++i){
    if (input->dim[i] != 1)
    {
        printf("TopKSample: dimension mismatch at dim[%d]\n", i);
        Dimension(input);
        std::exit(1);
    }}

    if (k > THREADSPERBLOCK || k < 1)
    {
        printf("TopKSample: k must be in [1, 256], got %d\n", k);
        std::exit(1);
    }
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<float> dist(1e-7f, 1.0f - 1e-7f);
    const float rand_val = dist(gen);
    TopKSampleKernel<<<1, 1>>>(input->output, X, input->total, k, rand_val);
    int sampled_id;
    cudaMemcpy(&sampled_id, X, sizeof(int), cudaMemcpyDeviceToHost);
    return sampled_id;
}

graph_tree topological_sort(const graph& root) 
{
    std::unordered_set<NodeBackProp*> visited;
    graph_tree result;
    
    std::function<void(const graph&)> dfs = [&](const graph& node) 
    {
        if (!node || visited.count(node.get())) return;
        
        visited.insert(node.get());
        for (const auto& input : node->inputs) 
        {
            dfs(input);
        }
        result.push_back(node);
    };
    
    dfs(root);

 
    return result;
}

graph GraphOperations::like(const graph& X, const str name)
{
    /*
    @brief Function requires manual clearing of nodes created during graph computation
    */
    auto node = std::make_shared<NodeBackProp>(name, X->dim[0], X->dim[1], X->dim[2], X->dim[3],1);
    node->inputs = {};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    node->forward = [=](){};
    node->backward = [=](){};
    node->zero_grad = [=](){Zerograd(node);};
    node->free = [=](){};
    return node;
}

graph GraphOperations::Dropout(const graph& X, const float p, const bool eval)
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    auto node = std::make_shared<AdamParameter>("Dropout " + X->op_name, a,b,c,d,1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    std::random_device *rd = new std::random_device();
    float* mask = nullptr;
    if(!eval) SafeCudaMalloc("Dropout Mask", mask, node->total);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    node->forward = [=]()
    {
        const uint64_t seed =  ((uint64_t)rd->operator()() << 32) | rd->operator()();
        if (!eval) dropoutKernel<<<bpg,tpb>>>(X->output, mask,node->output, node->total,p, seed);
        else cudaMemcpy(node->output, X->output, node->total*sizeof(float), cudaMemcpyDeviceToDevice);
         //CheckError("Dropout forward");
    };

    node->backward = [=]()
    {
        if (!eval) dropoutKernel<<<bpg,tpb>>>(X->output, mask,node->output, node->total,p,0,1);
        else cudaMemcpy(X->grad, node->grad, node->total*sizeof(float), cudaMemcpyDeviceToDevice);
        //CheckError("Dropout backward");
    };
    node->zero_grad = [=](){Zerograd(node);};
    node->free = [=]()
    {
        delete rd;
        cudaFree(mask);
        node->clear();
    };
    return node;
}

graph GraphOperations::HeadifytoChannel(const graph& X, const int channels)
{
    if (X->dim[1] != 1 || X->dim[3] % channels != 0)
    {
        std::cerr << "Input to headify must have channel dimension of 1 and width must be divisible by number of heads \n";
        Dimension(X);
        std::exit(1);
    }

    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("Headified Channel of " + X->op_name, a, channels, c, d / channels,1);
    const int tpb = THREADSPERBLOCK;
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    
    node->forward = [=]()
    {
        //isNan(X);
        HeadifyColChannel<<<(node->total+tpb-1)/tpb,tpb>>>(X->output, node->output, a, b, c, d, channels);
        //CheckError("Headify forward");
    };
    
    node->backward = [=]()
    {
        //isNan(node,1);
        HeadifyColChannel<<<(node->total+tpb-1)/tpb,tpb>>>(node->grad, X->grad, a, channels, c, d / channels, 1);
        //CheckError("Headify backward");
    };
    
    node->zero_grad = [=](){Zerograd(node);};
    node->free = [=](){node->clear();};
    return node;
}

graph GraphOperations::DeHeadify(const graph& X)
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("Deheadified " + X->op_name, a, 1, c, b*d,1);
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    node->inputs = {X};
    node->forward = [=]()
    {
        //isNan(X);
        HeadifyColChannel<<<(node->total+tpb-1)/tpb,tpb>>>(X->output, node->output, a, b, c, d, 2);
        //CheckError("DeHeadify forward");
    };
    
    node->backward = [=]()
    {
        //isNan(node,1);
        HeadifyColChannel<<<(node->total+tpb-1)/tpb,tpb>>>(node->grad, X->grad, a, b, c, d, 3);
        //CheckError("DeHeadify backward");
    };
    
    node->zero_grad = [=](){Zerograd(node);};
    node->free = [=](){ node->clear();};
    return node;
}

graph GraphOperations::Permute(const graph& X, const int i0, const int i1, const int i2, const int i3)
{
    const int a = X->dim[0]; const int pa = X->dim[i0];
    const int b = X->dim[1]; const int pb = X->dim[i1];
    const int c = X->dim[2]; const int pc = X->dim[i2];
    const int d = X->dim[3]; const int pd = X->dim[i3];
    int inv_perm[4];
    int perm[4] = {i0, i1, i2, i3};
    for(int i = 0; i < 4; i++) {inv_perm[perm[i]] = i;}
    auto node = std::make_shared<NodeBackProp>("Permuted " + X->op_name, pa, pb, pc, pd, 1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1) / tpb;
    node->forward = [=]()
    {
        permute<<<bpg,tpb>>>(X->output, node->output, a, b, c, d, i0, i1, i2, i3);
        //CheckError("Permute forward");
    };
    
    node->backward = [=]()
    {
        permute<<<bpg,tpb>>>(node->grad, X->grad, a, b, c, d, inv_perm[0], inv_perm[1], inv_perm[2], inv_perm[3]);
        //CheckError("Permute backward");
    };
    
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
}

graph GraphOperations::Clamp(const graph& X, const float min, const float max)
{
    const int batch = X->dim[0];
    const int channels = X->dim[1];
    const int a = X->dim[2];
    const int b = X->dim[3];

    auto node = std::make_shared<NodeBackProp>("Clamp", batch, channels, a, b, 1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;

    node->forward = [=]()
    {
        //isNan(X);
        clampKernel<<<bpg,tpb>>>(X->output,nullptr,node->output,min, max, node->total);
        //CheckError("Exp forward");
    };

    node->backward = [=]()
    {
        clampKernel<<<bpg,tpb>>>(X->output, node->grad, X->grad,min, max, node->total,1);
        //CheckError("Exp Backward");
    };

    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};

        
    return node;
}

graph GraphOperations::PositionalEncoding(const int &t, const int d_model)
{

    auto node = std::make_shared<NodeBackProp>("SinoSuodalPosEncoding", 1, 1, 1, d_model, 1);
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1) / tpb;
    node->inputs = {};
    node->forward = [=]()
    {
        PEncoding<<<bpg,tpb>>>(node->output, t, d_model, node->total);
        //CheckError("SinoSuodal Positional Encoding forward");
    };
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
        
}

graph GraphOperations::MatrixPositionalEncoding(const graph& X, const int start_idx)
{
    if (X->dim[1] != 1)
    {
        std::cerr << "MatrixPositionalEncoding expects input with channel dimension of 1\n";
        Dimension(X);
        std::exit(1);
    }

    const int batch = X->dim[0];
    const int rows = X->dim[2];
    const int cols = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("MatrixSinoSuodalPosEncoding", batch, 1, rows, cols, 1);
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1) / tpb;
    node->inputs = {X};
    node->forward = [=]()
    {
        MatPEncoding<<<bpg,tpb>>>(X->output, node->output,batch,rows,cols, node->total, start_idx);
        //CheckError("Matrix SinoSuodal Positional Encoding forward");
    };

    node->backward = [=]()
    {
        Accumulate<<<bpg,tpb>>>(node->grad, X->grad, node->total);
        //CheckError("Matrix SinoSuodal Positional Encoding backward");
    };

    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
        
}

graph GraphOperations::Broadcast_Add(const graph& A, const graph& B)
{
    if(A->dim[1] != B->dim[1])
    {
        std::cout << "Channel mismatch in broadcast add \n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }

    const int batch = A->dim[0];
    const int channels = A->dim[1];
    const int a = A->dim[2];
    const int b = A->dim[3];
    const int c = B->dim[2];
    const int d = B->dim[3];

    if(a % c != 0 || b % d != 0)
    {
        std::cout << "Spatial dimensions not divisible in broadcast add \n";
        std::cout << "A spatial: (" << a << ", " << b << "), B spatial: (" << c << ", " << d << ")\n";
        std::cout << "Requires: A_height % B_height == 0 and A_width % B_width == 0\n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }

    auto node = std::make_shared<NodeBackProp>("Broadcast Add", batch, channels, a, b, 1);
    node->inputs = {A, B};
    const int total_size = batch * channels * a * b;
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    const int B_total = batch * channels * c * d;

    node->forward = [=]()
    {
        //isNan(A); //isNan(B);
        broadcast_add_general<<<bpg,tpb>>>(A->output, B->output, node->output, batch, channels, a, b, c, d);
        //CheckError("Broadcast Add forward");
    };

    node->backward = [=]()
    {
        Accumulate<<<bpg,tpb>>>(node->grad, A->grad, total_size);
        //CheckError("Broadcast Add backward - A grad");

        broadcast_add_backward<<<(B_total+tpb-1)/tpb, tpb>>>(node->grad, B->grad, batch, channels, a, b, c, d);
        //CheckError("Broadcast Add backward - B grad");
    };

    node->free = [=](){node->clear();};

    node->zero_grad = [=](){Zerograd(node);};

    
    return node;
}

graph GraphOperations::Bias_Add(const graph& A, const graph& B)
    {
        if(B->dim[0] != 1 || B->dim[1] != 1 || B->dim[2] != 1 || A->dim[3] != B->dim[3])
        {
            std::cout << "Dimension mismatch in Bias_Add.. Bias must be a 1D vector matching at dim[3] \n";
            Dimension(A);
            Dimension(B);
            std::exit(1);
        }

        const int batch = A->dim[0];
        const int channels = A->dim[1];
        const int a = A->dim[2];
        const int b = A->dim[3];

        auto node = std::make_shared<NodeBackProp>("Add", batch, channels, a, b, 1);
        node->inputs = {A,B};
        GB += (double)(node->total) * sizeof(float) / (pow(2,30));
        const int tpb = THREADSPERBLOCK;
        const int bpg = (node->total+tpb-1)/tpb;

        node->forward = [=]()
        {
            //isNan(A); //isNan(B);
            cudaMemcpy(node->output, A->output, A->total*sizeof(float),cudaMemcpyDeviceToDevice);
            BCumAdd<<<bpg,tpb>>>(node->output, B->output,batch,channels,a,b);
            //CheckError("Bias_Add forward");
        };

        node->backward = [=]()
        {
            Accumulate<<<bpg,tpb>>>(node->grad, A->grad, node->total);
            //CheckError("Add backward - A grad");

            BCompress<<<b, tpb>>>(node->grad, B->grad, batch, channels, a, b);
            //CheckError("Add backward - B grad");
        };

        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};

        
        return node;
    }

graph GraphOperations::Broadcast_Channel(const graph& A, const graph& B)
{
    if(A->dim[1] != B->dim[3] || B->dim[1] != 1 || B->dim[2] != 1)
    {
        std::cout << "Channel mismatch in broadcast channel \n";
        std::cout << "Expected B shape: [1, 1, 1, channels], got ["<<B->dim[0]<<", "<<B->dim[1]<<", "<<B->dim[2]<<", "<<B->dim[3]<<"]\n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }
    
    const int batch = A->dim[0];
    const int channels = A->dim[1];
    const int a = A->dim[2];
    const int b = A->dim[3];

    auto node = std::make_shared<NodeBackProp>("Broadcast Channel", batch, channels, a, b, 1);
    node->inputs = {A, B};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;

    node->forward = [=]()
    {
        //isNan(A); //isNan(B);
        broadcast_add<<<bpg,tpb>>>(A->output, B->output, node->output, batch, channels, a, b);
        //CheckError("Broadcast Add forward");
    };

    node->backward = [=]()
    {
        Accumulate<<<bpg,tpb>>>(node->grad, A->grad, node->total);
        //CheckError("Broadcast Add backward - A grad");

        Channel_Squeeze1D<<<b,tpb>>>(node->grad, B->grad, batch, channels, a, b);
        //CheckError("Broadcast Add backward - B grad");
        
    };

    node->free = [=](){node->clear();};

    node->zero_grad = [=](){Zerograd(node);};

    
    return node;
}

graph GraphOperations::Exp(const graph& X)
{
    const int batch = X->dim[0];
    const int channels = X->dim[1];
    const int a = X->dim[2];
    const int b = X->dim[3];

    auto node = std::make_shared<NodeBackProp>("Exponentiate", batch, channels, a, b, 1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;

    node->forward = [=]()
    {
        //isNan(X);
        exponentiate<<<bpg,tpb>>>(X->output,nullptr,node->output, node->total);
        //CheckError("Exp forward");
    };

    node->backward = [=]()
    {
        exponentiate<<<bpg,tpb>>>(node->output,node->grad, X->grad, node->total,1);
        //CheckError("Exp Backward");
    };

    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};

        
    return node;
}

graph GraphOperations::Log(const graph& X)
{
    const int batch = X->dim[0];
    const int channels = X->dim[1];
    const int a = X->dim[2];
    const int b = X->dim[3];

    auto node = std::make_shared<NodeBackProp>("Exponentiate", batch, channels, a, b, 1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;

    node->forward = [=]()
    {
        //isNan(X);
        natural_logarithm<<<bpg,tpb>>>(X->output,nullptr,node->output, node->total);
        //CheckError("Exp forward");
    };

    node->backward = [=]()
    {
        natural_logarithm<<<bpg,tpb>>>(X->output,node->grad, X->grad, node->total,1);
        //CheckError("Exp Backward");
    };

    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};

        
    return node;
}

graph GraphOperations::Add(const graph& A, const graph& B, const bool last)
    {
        if(A->dim[0] != B->dim[0] || A->dim[1] != B->dim[1] || A->dim[2] != B->dim[2] || A->dim[3] != B->dim[3])
        {
            std::cout << "Dimension mismatch in Add \n";
            Dimension(A);
            Dimension(B);
            std::exit(1);
        }

        const int batch = A->dim[0];
        const int channels = A->dim[1];
        const int a = A->dim[2];
        const int b = A->dim[3];

        auto node = std::make_shared<NodeBackProp>("Add", batch, channels, a, b, 1);
        node->inputs = {A,B};
        GB += (double)(node->total) * sizeof(float) / (pow(2,30));
        const int tpb = THREADSPERBLOCK;
        const int bpg = (node->total+tpb-1)/tpb;

        node->forward = [=]()
        {
            //isNan(A); //isNan(B);
            ScaleAdd<<<bpg,tpb>>>(A->output, B->output, node->output, 1.0, node->total);
            //CheckError("Add forward");
        };

        node->backward = [=]()
        {
        if(last) fillKernel<<<bpg,tpb>>>(node->grad,1.0f,node->total);
            Accumulate<<<bpg,tpb>>>(node->grad, A->grad, node->total);
            //CheckError("Add backward - A grad");

            Accumulate<<<bpg,tpb>>>(node->grad, B->grad, node->total);
            //CheckError("Add backward - B grad");
        };

        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};

        
        return node;
    }

graph GraphOperations::Multiply(const graph& A, const graph& B, const bool last)
    {
        if(A->dim[0] != B->dim[0] || A->dim[1] != B->dim[1] || A->dim[2] != B->dim[2] || A->dim[3] != B->dim[3])
        {
            std::cout << "Dimension mismatch in Add \n";
            Dimension(A);
            Dimension(B);
            std::exit(1);
        }

        const int batch = A->dim[0];
        const int channels = A->dim[1];
        const int a = A->dim[2];
        const int b = A->dim[3];

        auto node = std::make_shared<NodeBackProp>("Add", batch, channels, a, b, 1);
        node->inputs = {A,B};
        GB += (double)(node->total) * sizeof(float) / (pow(2,30));
        const int tpb = THREADSPERBLOCK;
        const int bpg = (node->total+tpb-1)/tpb;

        node->forward = [=]()
        {
            //isNan(A); //isNan(B);
            mulKernel<<<bpg,tpb>>>(A->output, B->output, node->output, node->total);
            //CheckError("Add forward");
        };

        node->backward = [=]()
        {
            if(last) fillKernel<<<bpg,tpb>>>(node->grad,1.0f,node->total); 
            mulKernel<<<bpg,tpb>>>(node->grad, B->grad, A->grad, node->total, true);
            //CheckError("Add backward - A grad");

            mulKernel<<<bpg,tpb>>>(node->grad, A->grad, B->grad, node->total, true);
            //CheckError("Add backward - B grad");
        };

        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};

        
        return node;
    }

graph GraphOperations::Min(const graph& A, const graph& B)
    {
        if(A->dim[0] != B->dim[0] || A->dim[1] != B->dim[1] || A->dim[2] != B->dim[2] || A->dim[3] != B->dim[3])
        {
            std::cout << "Dimension mismatch in Add \n";
            Dimension(A);
            Dimension(B);
            std::exit(1);
        }

        const int batch = A->dim[0];
        const int channels = A->dim[1];
        const int a = A->dim[2];
        const int b = A->dim[3];

        auto node = std::make_shared<NodeBackProp>("Min", batch, channels, a, b, 1);
        node->inputs = {A,B};
        GB += (double)(node->total) * sizeof(float) / (pow(2,30));
        const int tpb = THREADSPERBLOCK;
        const int bpg = (node->total+tpb-1)/tpb;
        float* mask;
        SafeCudaMalloc("MasK", mask, node->total);
        node->forward = [=]()
        {
            //isNan(A); //isNan(B);
            minKernel<<<bpg,tpb>>>(A->output, B->output,mask,nullptr, nullptr,node->output, node->total);
            //CheckError("Min forward");
        };

        node->backward = [=]()
        {
            minKernel<<<bpg,tpb>>>(A->output, B->output,mask, A->grad, B->grad, node->grad, node->total,1);
            //CheckError("Add forward");
        };

        node->free = [=]()
        {
            cudaFree(mask);
            node->clear();
        };
        node->zero_grad = [=](){Zerograd(node);};

        
        return node;
}

graph GraphOperations::Max(const graph& A, const graph& B)
    {
        if(A->dim[0] != B->dim[0] || A->dim[1] != B->dim[1] || A->dim[2] != B->dim[2] || A->dim[3] != B->dim[3])
        {
            std::cout << "Dimension mismatch in Add \n";
            Dimension(A);
            Dimension(B);
            std::exit(1);
        }

        const int batch = A->dim[0];
        const int channels = A->dim[1];
        const int a = A->dim[2];
        const int b = A->dim[3];

        auto node = std::make_shared<NodeBackProp>("Max", batch, channels, a, b, 1);
        node->inputs = {A,B};
        GB += (double)(node->total) * sizeof(float) / (pow(2,30));
        const int tpb = THREADSPERBLOCK;
        const int bpg = (node->total+tpb-1)/tpb;
        float* mask;
        SafeCudaMalloc("MasK", mask, node->total);
        node->forward = [=]()
        {
            //isNan(A); //isNan(B);
            maxKernel<<<bpg,tpb>>>(A->output, B->output,mask,nullptr, nullptr,node->output, node->total);
            //CheckError("Min forward");
        };

        node->backward = [=]()
        {
            maxKernel<<<bpg,tpb>>>(A->output, B->output,mask, A->grad, B->grad, node->grad, node->total,1);
            //CheckError("Add forward");
        };

        node->free = [=]()
        {
            cudaFree(mask);
            node->clear();
        };
        node->zero_grad = [=](){Zerograd(node);};

        
        return node;
}

graph GraphOperations::Transpose(const graph& X)
{
    const int batch = X->dim[0];
    const int channels = X->dim[1];
    const int a = X->dim[2];
    const int b = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("Transposed" + X->op_name, batch, channels, b, a, 1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    node->forward = [=]()
    {
        //isNan(X);
        transpose<<<bpg,tpb>>>(X->output, node->output, batch, channels, a, b);
        //CheckError("GO Transpose");
    };

    node->backward = [=]()
    {
        transpose<<<bpg,tpb>>>(node->grad,X->grad,batch, channels,b,a,1);
        //CheckError("GO Transpose Grad");
    };
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
    }

graph GraphOperations::Last(const graph& X)
{
    if(X->dim[0] != 1 || X->dim[1] != 1)
    {
        std::cout << "Dimension mismatch in Last \n";
        Dimension(X);
        std::exit(1);
    }

    auto node = std::make_shared<NodeBackProp>(X->op_name + " Last", X->dim[0], X->dim[1], 1, X->dim[3], 1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    const int offset = X->total - X->dim[3];
    node->forward = [=]()
    {
        //isNan(X);
        cudaMemcpy(node->output,X->output+offset,node->total*sizeof(float),cudaMemcpyDeviceToDevice);
        //CheckError("Last forward");
    };

    node->backward = [=](){};
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
    }

graph GraphOperations::MeanSquaredError(const graph& prediction, const graph& target, const bool last)
{   
    bool val = true;
    for(int i = 0; i < 4; ++i) {if (prediction->dim[i] != target->dim[i]) {val = false;}}
    if(val == false)
    {
        std::cout << "Dimension mismatch \n MSE dimensions are: \n";
        Dimension(prediction);
        Dimension(target);
        std::exit(1);
    }
        
    const int batch = target->dim[0];
    const int channels = target->dim[1];
    const int c = target->dim[2];
    const int d = target->dim[3];
    auto node = std::make_shared<NodeBackProp>("MSE",1,1,1,1,1);
    node->inputs = {prediction, target};
    GB += (double)(prediction->total + target->total + 1) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (prediction->total+tpb - 1) / tpb;
    node->forward = [=]()
    {   
        if(calculate_loss)
        {
        //isNan(prediction); //isNan(target);
        scalarMSE<<<(prediction->total+tpb-1)/tpb, tpb>>>(prediction->output,target->output,node->output,batch,prediction->total);
        ScaleValue(node->output, (float)target->total,1,1);
        //isNan(node);
        //CheckError("Scalar MSE in MSE forward");
        cudaMemcpy(&loss, node->output, sizeof(float), cudaMemcpyDeviceToHost);
        }
        else loss = 100.0f;
    };
    
    node->backward = [=]()
    {   
        if(last) deriv_MSE<<<bpg,tpb>>>(prediction->output, target->output,nullptr,prediction->grad, batch, c, d, target->total, true);
        else  deriv_MSE<<<bpg,tpb>>>(prediction->output, target->output, node->grad,prediction->grad, batch, c, d, target->total, false);
        //isNan(prediction, 1);
        //CheckError("derivative of MSE in MSE backward");
    };

    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd("Node", node->output, node->total);};
    return node;
}

graph GraphOperations::MeanSquaredError(const graph& prediction, const float& target, const int& target_idx, const bool last)
{   
    if(target_idx < 0 || target_idx >= prediction->dim[3])
    {
        std::cout << "Target index out of bounds in MSE with scalar target \n";
        std::cout << "Received target_idx: " << target_idx << ", prediction dim[3]: " << prediction->dim[3] << "\n";
        std::exit(1);
    }

    auto node = std::make_shared<NodeBackProp>("MSE",1,1,1,1,1);
    node->inputs = {prediction};
    GB += (double)(prediction->total) * sizeof(float) / (pow(2,30));

    node->forward = [=]()
    {   
        float value = ReadValueAt(prediction, target_idx);
        loss = (value - target) * (value - target);
    };   
    node->backward = [=]()
    {   
        float value = ReadValueAt(prediction, target_idx);
        
        cudaMemset(prediction->grad, 0, prediction->total * sizeof(float));
        if(last) WriteValueAt(prediction, 2.0f * (value - target), target_idx);
        else 
        {
            float grad = ReadValueAt(node, target_idx, true);
            WriteValueAt(prediction, 2.0f * (value - target) * grad, target_idx);
        }
    };
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){}; 
    return node;
}

graph GraphOperations::CrossEntropy(const graph& prediction, const graph& target, const bool last)
{   
        bool val = true;
        for(int i = 0; i < 4; ++i) 
        {
            if (prediction->dim[i] != target->dim[i]) {val = false;}
        
        }
        if(val == false)
        {
            std::cout << "Dimension mismatch \n CrossEntropy dimensions are: \n";
            Dimension(prediction);
            Dimension(target);
            std::exit(1);
        }
        
        const int batch = target->dim[0];
        const int channels = target->dim[1];
        const int c = target->dim[2];
        const int d = target->dim[3];
        auto node = std::make_shared<NodeBackProp>("Cross Entropy Loss",1,1,1,1,1);
        node->inputs = {prediction, target};
        GB += (double)(prediction->total + target->total + 1) * sizeof(float) / (pow(2,30));
        const int tpb = THREADSPERBLOCK;
        const int bpg = (prediction->total+tpb-1)/tpb;

        node->forward = [=]()
        {   
            if(calculate_loss)
            {
            scalarCE<<<(prediction->total+tpb-1)/tpb, tpb>>>(prediction->output,target->output,node->output,batch,prediction->total);
            ScaleValue(node->output, (float)batch,1,1);
            //isNan(node);
            //CheckError("Scalar CE in CE forward");
            cudaMemcpy(&loss, node->output, sizeof(float), cudaMemcpyDeviceToHost);
            }
            else loss = 100.0f;

        };

        node->backward = [=]()
        {   
            if(last) deriv_CE<<<bpg,tpb>>>(prediction->output, target->output,nullptr, prediction->grad, batch, c, d, target->total, true);
            else deriv_CE<<<bpg,tpb>>>(prediction->output, target->output, node->grad, prediction->grad, batch, c, d, target->total,false);
            //isNan(prediction, 1);
            //CheckError("derivative of CE in CE backward");
        };

        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};
        return node;
    }

graph GraphOperations::Entropy(const graph& X)
{
    const int batch = X->dim[0];
    const int channels = X->dim[1];
    const int a = X->dim[2];
    const int b = X->dim[3];

    auto node = std::make_shared<NodeBackProp>("Entropy", batch, channels, a, b, 1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;

    node->forward = [=]()
    {
        //isNan(X);
        vectorE<<<bpg,tpb>>>(X->output,nullptr,node->output, node->total);
        //CheckError("Exp forward");
    };

    node->backward = [=]()
    {
        vectorE<<<bpg,tpb>>>(X->output, node->grad, X->grad, node->total,1);
        //CheckError("Exp Backward");
    };

    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
}

graph GraphOperations::SoftMaxCrossEntropy(const graph& prediction, const graph& target, const bool last)
{   
        bool val = true;
        for(int i = 0; i < 4; ++i) {if (prediction->dim[i] != target->dim[i]) {val = false;}}
        if(val == false)
        {
            std::cout << "Dimension mismatch \n CrossEntropy dimensions are: \n";
            Dimension(prediction);
            Dimension(target);
            std::exit(1);
        }
        
        const int batch = target->dim[0];
        const int channels = target->dim[1];
        const int c = target->dim[2];
        const int d = target->dim[3];
        auto node = std::make_shared<NodeBackProp>("SoftMaxCrossEntropy Loss",batch,channels,c,d,1);
        node->inputs = {prediction, target};
        float* softmax_arr, *maxArr, *softmax;
        SafeCudaMalloc("Softmax array", softmax_arr, batch*channels*c);
        SafeCudaMalloc("Max array", maxArr, batch*channels*c);
        SafeCudaMalloc("SoftMax", softmax, node->total);
        GB += (double)(node->total + 1) * sizeof(float) / (pow(2,30));
        const int tpb = THREADSPERBLOCK;
        const int bpg = (prediction->total+tpb-1) / tpb;
        
        node->forward = [=]()
        {   
            if(calculate_loss)
            {
                //isNan(prediction); //isNan(target);
                SoftMax(prediction->output,softmax_arr, softmax, maxArr, batch,channels,c,d,0);
                scalarCE<<<(prediction->total+tpb-1)/tpb,tpb>>>(softmax,target->output,node->output,batch,prediction->total);
                ScaleValue(node->output, (float)batch,1,1);
                isNan(node);
                CheckError("Scalar SCE in SCE forward");
                cudaMemcpy(&loss,node->output,sizeof(float),cudaMemcpyDeviceToHost);
            }
            else loss = 100.0f;
        };

        node->backward = [=]()
        {   
            ScaleAdd<<<bpg,tpb>>>(node->grad, target->output, prediction->grad,-1.0f, target->total);
            if(!last) mulKernel<<<bpg,tpb>>>(prediction->grad, node->grad, prediction->grad, prediction->total);
            ScaleValue(prediction->grad, batch, prediction->total, 1);
            //isNan(prediction, 1);
            //CheckError("derivative of CE in CE backward");
        };

        node->free = [=]()
        {
            node->clear();
            cudaFree(softmax);
            cudaFree(softmax_arr);
            cudaFree(maxArr);
        };
        
        node->zero_grad = [=](){Zerograd("Node",node->output,1);};

        return node;
    }

graph GraphOperations::BMM(const graph& A, const graph& B) // m x n  * n x p = m x p
{
    if(A->dim[0] != B->dim[0] || A->dim[1] != B->dim[1] || A->dim[3] != B->dim[2])
     {
        std::cout << "Dimension mismatch in BMM \n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }

    const int batch = A->dim[0], channels = A->dim[1], m = A->dim[2], n = A->dim[3], p = B->dim[3];
    auto node = std::make_shared<NodeBackProp>("BMM", batch, channels, m, p, 1);
    node->inputs = {A,B};
    GB += (double)(node->total + A->total + B->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((p+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
    dim3 grid_dA((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);   
    dim3 grid_dB((p+BLOCK_SIZE-1)/BLOCK_SIZE,(n+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
    node->forward = [=]()
    {   
        //isNan(A); //isNan(B);
        bcmm<<<grid, block>>>(A->output, B->output, node->output,batch,channels,m,n,p); //Assignment
        //CheckError("BMM... A * B in GraphOperations BMM forward");
    };

    node->backward = [=]()
    {
        bcmmABT<<<grid_dA, block>>>(node->grad, B->output, A->grad, batch, channels, m, p, n,1); // ∂A = ∂Z * B^T
        //CheckError("BMM.. ∂A = ∂Z * B^T in GraphOperations BMM backward");

        bcmmATB<<<grid_dB, block>>>(A->output, node->grad, B->grad,batch, channels, m, n, p,1); // ∂B = A^T * ∂Z            //CheckError("MatMul... X^T*∂Z in GraphOperations BMM backward");
    };
        
        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};
        return node;
    }

graph GraphOperations::BMMABT(const graph& A, const graph& B) //  m x n * p x n = m x p
{
        if(A->dim[0] != B->dim[0] || A->dim[1] != B->dim[1] || A->dim[3] != B->dim[3])
        {
            std::cout << "Dimension mismatch in BMMABT \n";
            Dimension(A);
            Dimension(B);
            std::exit(1);
        }

        const int batch = A->dim[0], channels = A->dim[1], m = A->dim[2], n = A->dim[3], p = B->dim[2];

        auto node = std::make_shared<NodeBackProp>("BMM-ABT", batch, channels, m, p, 1);
        node->inputs = {A,B};

        GB += (double)(node->total + A->total + B->total) * sizeof(float) / (pow(2,30));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((p+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
        dim3 grid_dA((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);  // for m×n output
        dim3 grid_dB((n+BLOCK_SIZE-1)/BLOCK_SIZE,(p+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);  // for p×n output

        node->forward = [=]()
        {   
            //isNan(A); //isNan(B);
            bcmmABT<<<grid, block>>>(A->output, B->output, node->output,batch,channels,m,n,p); 
            //CheckError("BMM... A * B^T in GraphOperations BMMABT forward");
        };

        node->backward = [=]()
        {
            bcmm<<<grid_dA, block>>>(node->grad, B->output, A->grad, batch, channels, m, p, n, 1);
            //CheckError("BMM.. ∂A = ∂C × B in GraphOperations BMMABT backward");

            bcmmATB<<<grid_dB, block>>>(node->grad, A->output, B->grad, batch, channels, m, p, n, 1);
            //CheckError("BMMABT... ∂C^T × A in GraphOperations BMMABT backward");
        };
    
        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};
        return node;
    }

graph GraphOperations::BMMATB(const graph& A, const graph& B) // m x n * m x p = n x p
    {
        if(A->dim[0] != B->dim[0] || A->dim[1] != B->dim[1] || A->dim[2] != B->dim[2])
        {
            std::cout << "Dimension mismatch in BMMATB \n";
            Dimension(A);
            Dimension(B);
            std::exit(1);
        }

        const int batch = A->dim[0], channels = A->dim[1], m = A->dim[2], n = A->dim[3], p = B->dim[3];

        auto node = std::make_shared<NodeBackProp>("BMMATB", batch, channels, n, p, 1);
        node->inputs = {A,B};
        GB += (double)(node->total + A->total + B->total) * sizeof(float) / (pow(2,30));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((p+BLOCK_SIZE-1)/BLOCK_SIZE,(n+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
        dim3 grid_dA((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);  // for m×n output
        dim3 grid_dB((p+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);  // for m×p output

        node->forward = [=]()
        {   
            //isNan(A); //isNan(B);
            bcmmATB<<<grid, block>>>(A->output, B->output, node->output,batch,channels,m,n,p); 
            //CheckError("BMM... A^T * B in GraphOperations BMMATB forward");
        };

        node->backward = [=]()
        {

            bcmmABT<<<grid_dA, block>>>(B->output, node->grad, A->grad, batch, channels, m, p, n, 1);
            //CheckError("BMM.. ∂A = B × ∂C^T in GraphOperations BMMATB backward");

            bcmm<<<grid_dB, block>>>(A->output, node->grad, B->grad, batch, channels, m, n, p, 1);
            //CheckError("BMM... ∂B = A × ∂C in GraphOperations BMMATB backward");
        };

        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};
        return node;
    }

graph GraphOperations::BMMATBT(const graph& A, const graph& B) // m x n * p x m = n x p
{
        if(A->dim[0] != B->dim[0] || A->dim[1] != B->dim[1] || A->dim[2] != B->dim[3])
        {
            std::cout << "Dimension mismatch in BMMATBT \n";
            Dimension(A);
            Dimension(B);
            std::exit(1);
        }
        
        const int batch = A->dim[0], channels = A->dim[1], m = A->dim[2], n = A->dim[3], p = B->dim[2];
        auto node = std::make_shared<NodeBackProp>("BMM-ATBT", batch, channels, n, p, 1);
        node->inputs = {A,B};
        GB += (double)(node->total) * sizeof(float) / (pow(2,30));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((p+BLOCK_SIZE-1)/BLOCK_SIZE,(n+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
        dim3 grid_dA((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);  // for m×n output
        dim3 grid_dB((m+BLOCK_SIZE-1)/BLOCK_SIZE,(p+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);  // for p×m output

        node->forward = [=]()
        {   
            //isNan(A); //isNan(B);
            bcmmATBT<<<grid, block>>>(A->output, B->output, node->output,batch,channels,m,n,p); 
            //CheckError("BMM... A^T * B^T in GraphOperations BMMATBT forward");
        };

        node->backward = [=]()
        {
            bcmmATBT<<<grid_dA, block>>>(B->output, node->grad, A->grad, batch, channels, p, m, n, 1);
            //CheckError("BMM.. ∂A = B^T × ∂C^T in GraphOperations BMMATBT backward");
        
            bcmmATBT<<<grid_dB, block>>>(node->grad, A->output, B->grad, batch, channels, n, p, m, 1);
            //CheckError("BMMATBT... ∂C^T × A^T in GraphOperations BMMATBT backward");
        };
    
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
}
    
graph GraphOperations::SOFTMAX(const graph& X, const int type) // type 0: row-wise, type 1: column-wise
{
        const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
        auto node = std::make_shared<NodeBackProp>("Softmax", a, b,c,d, 1);
        node->inputs = {X};    
        GB += (double)node->total * sizeof(float) / (pow(2,30));
        const int arr_size = (type == 0) ? a*b*c : a*b*d;
        const int max_size = (type == 0) ? a*b*c : a*b*d;
        float *arr, *maxArr;
        SafeCudaMalloc("Softmax array", arr, arr_size);
        SafeCudaMalloc("Max array", maxArr, max_size);

        node->forward = [=]() 
        {   
            //isNan(X);
            SoftMax(X->output, arr, node->output, maxArr, a,b,c, d, type);
            //CheckError("Softmax forward");

        };

        node->backward = [=]() 
        {
            deriv_SoftMax(node->output,node->grad,X->grad,a,b,c,d,type);
            //CheckError("Deriv Softmax in Softmax");
            //isNan(X, 1);
        };

        node->free =  [=]()
        {
            node->clear();
            cudaFree(arr);
            cudaFree(maxArr);
        };
        
        node->zero_grad = [=](){Zerograd(node);};
        return node;

    }

graph GraphOperations::SOFTMASK(const graph& X, const int type) // type 0: row-wise, type 1: column-wise
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("Softmask", a,b,c,d, 1);
    node->inputs = {X};    
    GB += (double)node->total * sizeof(float) / (pow(2,30));
    const int arr_size = (type == 0) ? a*b*c : a*b*d;
    const int max_size = (type == 0) ? a*b*c : a*b*d;
    float* arr, *maxArr;
    SafeCudaMalloc("Softmask array", arr, arr_size);
    SafeCudaMalloc("Max array", maxArr, max_size);

    node->forward = [=]() 
    {   
        //isNan(X);
        SoftMask(X->output, arr, node->output, maxArr, a, b, c, d, type);
         //CheckError("Softmask forward");
    };

    node->backward = [=]() 
    {
        deriv_SoftMax(node->output, node->grad, X->grad,a, b, c, d, type);
        //CheckError("Deriv Softmask in Softmask");
        //isNan(X, 1);
    };

    node->free =  [=]()
    {
        node->clear();
        cudaFree(arr);
        cudaFree(maxArr);
    };
        
    node->zero_grad = [=](){Zerograd(node);};
    return node;

}

graph GraphOperations::GatherAction(const graph& X, const graph& actions) // Actions is a no grad function
{
    if(X->dim[0] != actions->dim[0] || X->dim[1] != 1 || X->dim[1] != actions->dim[1] || X->dim[2] != 1 ||  X->dim[2] != actions->dim[2] || actions->dim[3] != 1)  
    {
        printf("Dimension mismatch, expected (%i x 1 x 1 x col) and (%i x 1 x 1 x 1)... Received: \n", X->dim[0], actions->dim[0]);
        Dimension(X); Dimension(actions);
        std::exit(1);
    } 
        const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
        auto node = std::make_shared<NodeBackProp>("Gathering "+X->op_name+" Actions",a,1,1,1,1);
        node->inputs = {X,actions};    
        GB += (double)node->total * sizeof(float) / (pow(2,30));
        node->forward = [=]() {getactionKernel<<<a,1>>>(X->output, actions->output, node->output,c,a,false);};
        node->backward = [=](){getactionKernel<<<a,1>>>(node->grad, actions->output, X->grad,c,a,true);};
        node->free =  [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};
        return node;


};

graph GraphOperations::Scale(const graph& input, const float scale)
{
    const int a = input->dim[0], b = input->dim[1], c = input->dim[2], d = input->dim[3];
    auto node = std::make_shared<NodeBackProp>("Scale", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (pow(2,30));

    node->forward = [=]() 
    {   
        //isNan(input);
        ScaleGraph(input->output,node->output,scale,node->total);
        //CheckError("Scale Value in Scale forward");

    };

    node->backward = [=]() 
    {
        ScaleGraph(node->grad, input->grad, scale, node->total,1);
        //CheckError("Deriv ReLU in RELU");
        //isNan(input, 1);
    };

    node->free =  [=](){node->clear();};

    node->zero_grad = [=](){Zerograd(node);};

    
    return node; 
    }

graph GraphOperations::RELU(const graph& input)
{
    const int a = input->dim[0], b = input->dim[1], c = input->dim[2], d = input->dim[3];
    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("ReLU", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (pow(2,30));

    node->forward = [=]() 
    {   
        //isNan(input);
        ReLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output,node->output, node->total); // Assignment operation
        //CheckError("RELU in RELU forward");

    };

    node->backward = [=]() 
    {
        deriv_ReLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output, node->grad,input->grad,node->total);
        //CheckError("Deriv ReLU in RELU");
        //isNan(input,1);
    };

    node->free =  [=](){node->clear();};

    node->zero_grad = [=](){Zerograd(node);};

    
    return node; 
    }

graph GraphOperations::SILU(const graph& input)
{
    const int a = input->dim[0], b = input->dim[1], c = input->dim[2], d = input->dim[3];
    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("SiLU", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (pow(2,30));

    node->forward = [=]() 
    {   
        //isNan(input);
        SiLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output,node->output, a*b*c*d); 
        //CheckError("RELU in RELU forward");

    };

    node->backward = [=]() 
    {
        deriv_SiLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output, node->grad,input->grad,node->total);
        //CheckError("Deriv ReLU in RELU");
        //isNan(input, 1);
    };

    node->free =  [=](){node->clear();};

    node->zero_grad = [=](){Zerograd(node);};

    
    return node; 
}

graph GraphOperations::TANH(const graph& input)
{

    const int a = input->dim[0];
    const int b = input->dim[1];
    const int c = input->dim[2];
    const int d = input->dim[3];
    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("Sigmoid", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (pow(2,30));

    node->forward = [=]() 
    {   
        //isNan(input);
        TaNH<<<(node->total+tpb-1)/tpb,tpb>>>(input->output,node->output, node->total); // Assignment operation
        //CheckError("RELU in RELU forward");

    };

    node->backward = [=]() 
    {
        deriv_TaNH<<<(node->total+tpb-1)/tpb,tpb>>>(input->output, node->grad,input->grad,node->total);
        //CheckError("Deriv ReLU in RELU");
        //isNan(input,1);
    };

    node->free =  [=](){node->clear();};

    node->zero_grad = [=](){Zerograd(node);};

    
    return node; 
    }

graph GraphOperations::SIGMOID(const graph& input)
{

    const int a = input->dim[0];
    const int b = input->dim[1];
    const int c = input->dim[2];
    const int d = input->dim[3];
    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("Sigmoid", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (pow(2,30));

    node->forward = [=]() 
    {   
        //isNan(input);
        Sigmoid<<<(node->total+tpb-1)/tpb,tpb>>>(input->output,node->output, node->total); // Assignment operation
        //CheckError("RELU in RELU forward");

    };

    node->backward = [=]() 
    {
        deriv_Sigmoid<<<(node->total+tpb-1)/tpb,tpb>>>(input->output, node->grad,input->grad,node->total);
        //CheckError("Deriv ReLU in RELU");
        //isNan(input,1);
    };

    node->free =  [=](){node->clear();};

    node->zero_grad = [=](){Zerograd(node);};

    
    return node; 
    }

graph GraphOperations::LeakyRELU(const graph& input)
    {
    const int a = input->dim[0];
    const int b = input->dim[1];
    const int c = input->dim[2];
    const int d = input->dim[3];
    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("LeakyReLU", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (pow(2,30));

    node->forward = [=]() 
    {   
        //isNan(input);
        LeakyReLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output,node->output, node->total); // Assignment operation
        //CheckError("LeakyRELU in LeakyRELU forward");

    };

    node->backward = [=]() 
    {
        
        deriv_LeakyReLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output, node->grad,input->grad,node->total);
        //CheckError("Deriv LeakyReLU in LeakyRELU");
        //isNan(input, 1);
    };

    node->free = [=]()
    {
        node->clear(); 
    };

    node->zero_grad = [=](){Zerograd(node);};
    
    
    return node; 
    }

graph GraphOperations::CopyCrop(const graph& input1, const graph& input2) // @Channel wise concatenation with cropping or padding as necessary
{   
        const int batch = input2->dim[0];
        const int depth =  input2->dim[1];
        const int d1  = input1->dim[1];
        const int a1 = input1->dim[2];
        const int b1 = input1->dim[3];
        const int a = input2->dim[2];
        const int b = input2->dim[3];
        const int tpb = THREADSPERBLOCK;
        auto node = std::make_shared<NodeBackProp>("CopyNCrop", batch,input1->dim[1] + depth,a,b,1);
        node->inputs = {input1, input2};
        float *temp, *tGrad; 
        const bool condition = (a1 != a || b1 != b);

        if (condition)
        {
            GB += 3 * (double)(node->total) * sizeof(float) / (pow(2,30));
            SafeCudaMalloc("Temp of CopyCrop", temp, batch * depth * a * b);
            SafeCudaMalloc("TGrad of CopyCrop",tGrad, batch * depth * a * b);
        }
        
        else{GB += (double)(node->total) * sizeof(float) / (pow(2,30));}

        node->forward = [=]()
        {   
        //isNan(input1); //isNan(input2);
        if(condition)
        {
            CopynCrop<<<(tpb+input1->total-1)/tpb, tpb>>>(input1->output, temp, batch, d1,a1,b1,a,b); //Assignment
            Channel_Concat<<<(tpb+node->total-1)/tpb,tpb>>>(temp,input2->output,node->output,batch,d1,depth,a,b); //Assignment
            //CheckError("Concatenation in CopynCrop");
        }
        else{Channel_Concat<<<(tpb+node->total-1)/tpb,tpb>>>(input1->output,input2->output,node->output,batch,d1,depth,a,b);}  

        };

        node->backward= [=]()
        {        
            if(condition)
            { 
            Channel_Split<<<(tpb+node->total-1)/tpb, tpb>>>(tGrad, input2->grad,node->grad,batch,d1,depth,a,b);
            PaddingCrop<<<(tpb+input1->total-1)/tpb, tpb>>>(tGrad, input1->grad,batch,d1,a1,b1,a,b);
            }
            else
            {
                Channel_Split<<<(tpb+node->total-1)/tpb, tpb>>>(input1->grad,input2->grad,node->grad,batch,d1,depth,a,b);
            }
            //CheckError("Backward CopynCrop");
            //isNan(input1, 1); //isNan(input2, 1);
        };  

        node->free = [=]()
        {
            node->clear();
            if(condition){cudaFree(temp);cudaFree(tGrad);}
        };

        node->zero_grad = [=]()
        {
            Zerograd(node);
            if (condition) Zerograd("TGrad", tGrad, batch*depth*a*b);
        };        
        
        return node;
    }

graph GraphOperations::CopyConcat(const graph& input1, const graph& input2) // @Column wise concatenation without cropping
{   
    const int batch = input2->dim[0];
    const int depth =  input2->dim[1];
    const int d1  = input1->dim[1];
    const int a1 = input1->dim[2];
    const int b1 = input1->dim[3];
    const int a = input2->dim[2];
    const int b = input2->dim[3];
    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("CopyConcat", batch,depth,a,b+b1,1);
    node->inputs = {input1, input2};
    if (batch != input1->dim[0] || d1 != depth || a1 != a)
    {
            std::cout << "CopyConcat currently only supports concatenation of tensors with the same channels and rowshape \n";
            Dimension(input1);
            Dimension(input2);
            std::exit(1);
    }
   
        GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    node->forward = [=]()
    {   
            //isNan(input1); //isNan(input2);
            Column_Concat<<<(tpb+node->total-1)/tpb,tpb>>>(input1->output,input2->output,node->output,batch,d1,a,b1,b);
            //CheckError("Concatenation in CopyConcat");
    };

    node->backward= [=]()
    {
        Column_Split<<<(tpb+node->total-1)/tpb, tpb>>>(input1->grad,input2->grad,node->grad,batch,d1,a,b1,b);
        //CheckError("Split in CopyConcat");
        //isNan(input1, 1);//isNan(input2, 1);
    };
    node->zero_grad =  [=]() {Zerograd(node);};
    node->free = [=](){node->clear();};
    return node;
    }

graph GraphOperations::VecConcat(const graph_tree& inputs)
{
    const int batch = inputs[0]->dim[0], depth = inputs[0]->dim[1], a = inputs[0]->dim[2];
    int total_b = 0;
    for (const auto& input : inputs)
    {
        if (input->dim[0] != batch || input->dim[1] != depth || input->dim[2] != a)
        {
            std::cout << "VecConcat currently only supports concatenation of tensors with the same batch, channels and rowshape \n";
            Dimension(input);
            std::exit(1);
        }
        total_b += input->dim[3];
    }

    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("VecConcat", batch, depth, a, total_b, 1);
    node->inputs = inputs;
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));

    node->forward = [=]()
    {    
        for (const auto& input : inputs)
        {
            //isNan(input);
        }
        
        int col_offset = 0;
        
        for (int i=0; i< inputs.size(); i++)
        {
        
        Vector_Concat<<<(inputs[i]->total+tpb-1)/tpb,tpb>>>(inputs[i]->output,node->output,batch,
        depth,a,inputs[i]->dim[3],total_b,col_offset);
        //CheckError("Concatenation in VecConcat");
        col_offset += inputs[i]->dim[3];
        }
    };

    node->backward= [=]()
    {
        int col_offset = 0;
        for(int i = inputs.size()-1; i >= 0; --i)
        {
            
            Vector_Split<<<(inputs[i]->total+tpb-1)/tpb,tpb>>>(node->grad,inputs[i]->grad,batch,
            depth,a,inputs[i]->dim[3],total_b,col_offset);
            col_offset += inputs[i]->dim[3];
        }

        //CheckError("Split in VecConcat");
        /*for (const auto& input : inputs){isNan(input, 1);}*/

    };
    
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    
    return node;
}

graph GraphOperations::LAYERMEAN(const graph& X)
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("LayerMean", a, 1,1,1,1);
    node->inputs = {X};    
    GB += (double)node->total * sizeof(float) / (pow(2,30));

    node->forward = [=]() 
    {   
        //isNan(input);
        LayerMean<<<a,tpb>>>(X->output,node->output,a,b,c,d); // Assignment operation
        //CheckError("RELU in RELU forward");

    };

    node->backward = [=]() 
    {
        LayerMeanGrad<<<(node->total+tpb-1)/tpb,tpb>>>(node->grad, X->grad, a, b, c, d);
        //CheckError("Deriv ReLU in RELU");
        //isNan(input,1);
    };

    node->free =  [=](){node->clear();};

    node->zero_grad = [=](){Zerograd(node);};

    
    return node; 
    }

graph GraphOperations::LayerNorm(const graph& X)
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    const float gamma = 1.0f, beta = 0.0f, epsilon = 1e-5f;
    auto node = std::make_shared<NodeBackProp>("LayerNorm",a,b,c,d,1);
    float *mean, *std, *ggamma_mean, *ggammanode_mean;
    float *ggamma, *ggammanode; 
    const int tpb = THREADSPERBLOCK;

    SafeCudaMalloc("LayerNorm ggamma", ggamma, node->total);
    SafeCudaMalloc("LayerNorm ggammanode", ggammanode, node->total);
    SafeCudaMalloc("LayerNorm mean", mean, a);
    SafeCudaMalloc("LayerNorm  std", std,  a);
    SafeCudaMalloc("LayerNorm ggamma_mean", ggamma_mean, a);
    SafeCudaMalloc("LayerNorm ggammanode_mean", ggammanode_mean,  a);

    node->inputs = {X};

    node->forward = [=]()
    {
        //isNan(X);
        cudaMemcpy(node->output, X->output, node->total*sizeof(float),cudaMemcpyDeviceToDevice);
        //CheckError("CudaMemcpy of LayerNorm");

        LayerMean<<<a, tpb>>>(X->output,mean,a,b,c,d); // Assigment
        //CheckError("LayerMean of LayerNorm");

        LayerStd<<<a, tpb>>>(X->output,mean,std,a,b,c,d); // Assignment
        //CheckError("LayerStd of LayerNorm");

        LNorm<<<(node->total+tpb-1)/tpb,tpb>>>(node->output,mean,std,a,b,c,d,gamma,beta,epsilon); // Assignment
        //CheckError("LNorm of LayerNorm");
    };

    node->backward = [=]()
    {
        //isNan(node, 1);
        cudaMemcpy(ggamma, node->grad, node->total*sizeof(float), cudaMemcpyDeviceToDevice);
        //CheckError("Memset and Memcpy in LNorm Backward");

        ScaleValue(ggamma,gamma,node->total);
        //CheckError("Scale in LNorm Backward");

        mulKernel<<<(node->total+tpb-1)/tpb,tpb>>>(ggamma,node->output,ggammanode,node->total);
        //CheckError("Multiply");

        LayerMean<<<a, tpb>>>(ggamma, ggamma_mean, a,b,c,d, false);
        LayerMean<<<a, tpb>>>(ggammanode, ggammanode_mean, a,b,c,d, false);
        //CheckError("LayerMean of ggammas");

        LayerBackward<<<(node->total+tpb-1)/tpb,tpb>>>(X->grad,node->output,node->grad,ggamma_mean,ggammanode_mean,std,gamma,epsilon,a,b,c,d);
        //CheckError("LayerBackward of LayerNorm");
    };

    node->free = [=]()
    {
        node->clear();
        cudaFree(mean);
        cudaFree(std);
        cudaFree(ggamma);
        cudaFree(ggammanode);
        cudaFree(ggamma_mean);
        cudaFree(ggammanode_mean);      
    };

    node->zero_grad = [=](){Zerograd(node);};

    return node;
}

graph GraphOperations::BatchNorm(const graph& X)
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("BatchNorm",a,b,c,d,1);
    const float gamma = 1.0f, beta = 0.0f, epsilon = 1e-5f;
   float *mean, *std, *ggamma_mean, *ggammanode_mean;
    float *ggamma, *ggammanode;
    const int tpb = THREADSPERBLOCK;
    SafeCudaMalloc("BatchNorm ggamma", ggamma, node->total);
    SafeCudaMalloc("BatchNorm ggammanode", ggammanode, node->total);
    SafeCudaMalloc("BatchNorm mean", mean, b);
    SafeCudaMalloc("BatchNorm  std", std,  b);
    SafeCudaMalloc("BatchNorm ggamma_mean", ggamma_mean, b);
    SafeCudaMalloc("BatchNorm ggammanode_mean", ggammanode_mean,  b);
    node->inputs = {X};

    node->forward = [=]()
    {
        //isNan(X);
        cudaMemcpy(node->output, X->output, node->total*sizeof(float),cudaMemcpyDeviceToDevice);
        //CheckError("CudaMemcpy of BatchNorm");
            
        BatchMean<<<b,tpb>>>(X->output,mean,a,b,c,d); // Assignment
        //CheckError("BatchMean in BatchNorm");

        BatchStd<<<b, tpb>>>(X->output,mean,std,a,b,c,d); // Assignment
        //CheckError("Batch Std in BatchNorm");

        BNorm<<<(node->total+tpb-1)/tpb,tpb>>>(node->output,mean,std,a,b,c,d,gamma,beta, epsilon); // Assignment
        //CheckError("BNorm in BatchNorm");        
    };

    node->backward = [=]()
    {
        //isNan(node,1);
        cudaMemcpy(ggamma, node->grad, node->total*sizeof(float), cudaMemcpyDeviceToDevice);
        //CheckError("Memset and Memcpy in BNorm Backward");

        ScaleValue(ggamma,gamma, node->total);
        //CheckError("Scale in BNorm Backward");

        mulKernel<<<(node->total+tpb-1)/tpb,tpb>>>(ggamma, node->output, ggammanode, node->total);
        //CheckError("Multiply");

        BatchMean<<<b,tpb>>>(ggamma, ggamma_mean, a,b,c,d, false);
        BatchMean<<<b,tpb>>>(ggammanode, ggammanode_mean, a,b,c,d, false);
        //CheckError("BatchMean of ggammas");
        BatchBackward<<<(node->total+tpb-1)/tpb,tpb>>>(X->grad,node->output,node->grad,ggamma_mean,ggammanode_mean,std,gamma,epsilon,a,b,c,d);

    };
        
    node->free = [=]()
    {
        node->clear();
        cudaFree(mean);
        cudaFree(std);
        cudaFree(ggamma_mean);
        cudaFree(ggammanode_mean);
        cudaFree(ggamma);
        cudaFree(ggammanode);
    };
        
    node->zero_grad = [=]()
    {
        Zerograd(node);
    };

    return node;
}

graph GraphOperations::GroupNorm(const graph& X, const int group)
{
    const int a = X->dim[0],b = X->dim[1],c = X->dim[2],d = X->dim[3];
        
    if(b%group != 0)
    {
        std::cout << "Groups cannot be cleanly split \n";
        Dimension(X);std::exit(1);
    }
        
    auto node = std::make_shared<NodeBackProp>("GroupNorm",a,b,c,d,1);
    const float gamma = 1.0f, beta = 0.0f, epsilon = 1e-5f;
    float *mean, *std, *ggamma_mean, *ggammanode_mean, *ggamma, *ggammanode;
    const int tpb = THREADSPERBLOCK;

    SafeCudaMalloc("GroupNorm ggamma", ggamma, node->total);
    SafeCudaMalloc("GroupNorm ggammanode", ggammanode, node->total);
    SafeCudaMalloc("GroupNorm mean", mean, a*group);
    SafeCudaMalloc("GroupNorm  std", std,  a*group);
    SafeCudaMalloc("GroupNorm ggamma_mean", ggamma_mean, a*group);
    SafeCudaMalloc("GroupNorm ggammanode_mean", ggammanode_mean,  a*group);
    
    node->inputs = {X};
        
    node->forward = [=]()
    {
        //isNan(X);
        cudaMemcpy(node->output, X->output, node->total*sizeof(float),cudaMemcpyDeviceToDevice);
        //CheckError("CudaMemcpy of GroupNorm");

        GroupMean<<<a*group,tpb>>>(X->output,mean,a,b,group,c,d);
        //CheckError("GroupMean of GroupNorm");

        GroupStd<<<a*group,tpb>>>(X->output,mean,std,a,b,group,c,d); 
        //CheckError("GroupStd of GroupNorm");

        GNorm<<<(node->total+tpb-1)/tpb,tpb>>>(node->output,mean,std,a,b,group,c,d,gamma, beta, epsilon); // Assignment
        //CheckError("GNorm of GroupNorm");
    };

    node->backward = [=]()
    {   

        //isNan(node, 1);
        cudaMemcpy(ggamma, node->grad, node->total*sizeof(float), cudaMemcpyDeviceToDevice);
        //CheckError("Memset and Memcpy in GroupNorm Backward");

        ScaleValue(ggamma,gamma, node->total);
        //CheckError("Scale in GroupNorm Backward");

        mulKernel<<<(node->total+tpb-1)/tpb,tpb>>>(ggamma, node->output, ggammanode, node->total);
        //CheckError("Multiply");

        GroupMean<<<a*group,tpb>>>(ggamma, ggamma_mean, a,b,group,c,d,false);
        GroupMean<<<a*group,tpb>>>(ggammanode, ggammanode_mean,a,b,group,c,d,false);

        //CheckError("GroupMean of ggammas");
        GroupBackward<<<(node->total+tpb-1)/tpb,tpb>>>(X->grad,node->output,node->grad,ggamma_mean,ggammanode_mean,std,gamma,epsilon,a,b,group,c,d);
    };
        
    node->free = [=]()
    {
        node->clear(); cudaFree(mean); cudaFree(std); cudaFree(ggamma_mean);
        cudaFree(ggammanode_mean); cudaFree(ggamma); cudaFree(ggammanode);
    };

    node->zero_grad = [=](){Zerograd(node);};

    return node;

}

graph GraphOperations::InstanceNorm(const graph & X)  
{
    const int a = X->dim[0];
    const int b = X->dim[1];
    const int c = X->dim[2];
    const int d = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("InstanceNorm",a,b,c,d,1);
    const float gamma = 1.0f;
    const float beta = 0.0f;
    const float epsilon = 1e-5f;
    float *mean, *std, *ggamma_mean, *ggammanode_mean, *ggamma, *ggammanode;
    const int tpb = THREADSPERBLOCK;

    SafeCudaMalloc("InstanceNorm ggamma", ggamma, node->total);
    SafeCudaMalloc("InstanceNorm ggammanode", ggammanode, node->total);
    SafeCudaMalloc("InstanceNorm mean", mean, a*b);
    SafeCudaMalloc("InstanceNorm  std", std,  a*b);
    SafeCudaMalloc("InstanceNorm ggamma_mean", ggamma_mean, a*b);
    SafeCudaMalloc("InstanceNorm ggammanode_mean", ggammanode_mean,  a*b);
    
    node->inputs = {X};      
    node->forward = [=]()
    {
        //isNan(X);
        cudaMemcpy(node->output, X->output, node->total*sizeof(float),cudaMemcpyDeviceToDevice);
        //CheckError("CudaMemcpy of InstanceNorm");

        InstanceMean<<<a*b,tpb>>>(X->output,mean,a,b,c,d);
        //CheckError("InstanceMean of InstanceNorm");

        InstanceStd<<<a*b,tpb>>>(X->output,mean,std,a,b,c,d); 
        //CheckError("InstanceStd of InstanceNorm");

        INorm<<<(node->total+tpb-1)/tpb,tpb>>>(node->output,mean,std,a,b,c,d,gamma, beta, epsilon); //Assignment
        //CheckError("INorm of InstanceNorm");
    };

    node->backward = [=]()
    {
        //isNan(node,1);
            
        cudaMemcpy(ggamma, node->grad, node->total*sizeof(float), cudaMemcpyDeviceToDevice);
        //CheckError("Memset and Memcpy in GroupNorm Backward");

        ScaleValue(ggamma,gamma,node->total);
        //CheckError("Scale in GroupNorm Backward");

        mulKernel<<<(node->total+tpb-1)/tpb,tpb>>>(ggamma, node->output, ggammanode, node->total);
        //CheckError("Multiply");

        InstanceMean<<<a*b,tpb>>>(ggamma, ggamma_mean, a,b,c,d, false);
        InstanceMean<<<a*b,tpb>>>(ggammanode, ggammanode_mean,a,b,c,d, false);

        //CheckError("GroupMean of ggammas");
        InstanceBackward<<<(node->total+tpb-1)/tpb,tpb>>>(X->grad,node->output,node->grad,ggamma_mean,ggammanode_mean,std,gamma,epsilon,a,b,c,d);

    };
        
    node->free = [=]()
    {
        node->clear(); cudaFree(mean); cudaFree(std); cudaFree(ggamma_mean); cudaFree(ggammanode_mean); cudaFree(ggamma); cudaFree(ggammanode);
    };

    node->zero_grad = [=](){Zerograd(node);};
        
    return node;

}
  
void GraphOperations::clipNorm(double* global_scale) {for(auto&node : nodes) if(node->clipnorm) node->clipnorm(global_scale);}
    
void GraphOperations::accumulate(double* global_scale) 
{
        for(auto&node : nodes) if(node->accumulate) node->accumulate(global_scale);
        Sqrt_Scale<<<1,1>>>(global_scale,1.0f,0);
}

void GraphOperations::ParameterUpdate() 
{
    if (nodes.size() == 0)
    {
        printf("Warning... Nodes list is empty and parameter update cannot run.. \n You may have forgotten to topologically sort \n");
        return;
    }

    for(auto&node : nodes) if(node->updateParams) node->updateParams();
}

void GraphOperations::forward() 
{   
    if (nodes.size() == 0)
    {
        printf("Warning... Nodes list is empty and forward cannot run.. \n You may have forgotten to topologically sort \n");
        return;
    }
    for (auto& node : nodes)
    {
        if (node->forward)  
        node->forward();
    }
}

void GraphOperations::backward() 
{
    if (nodes.size() == 0)
    {
        printf("Warning... Nodes list is empty and bacward cannot run.. \n You may have forgotten to topologically sort \n");
        return;
    }
    for(auto it=nodes.rbegin();it!=nodes.rend();++it){if((*it)->backward) (*it)->backward();    } 
}

void GraphOperations::zero_grad() 
{
        if (nodes.size() == 0)
    {
        printf("Warning... Nodes list is empty and Zero grad cannot run.. \n You may have forgotten to topologically sort \n");
        return;
    }
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it){   
    if ((*it)->zero_grad)
    {
        (*it)->zero_grad();
    }}
}

void GraphOperations::printNodes(const bool display_grad) 
{
    for (auto& node : nodes) {
    if (node->zero_grad) 
    {   
    std::cout << "Calling Node: " << node->op_name << "\n"; if (display_grad) printHeadGPU(node,1);
    }}
}

void GraphOperations::clear_graph()
{
    for (auto &node: nodes)
    {
        if(node->free)node->free();
    }
    nodes.clear();
}

void GraphOperations::clean_clear_graph()
{
    for (auto &node: nodes)
    {
        if(node->free)node->free();
        if(node->serious_free) node->serious_free();
    }
}

Identity::Identity(GraphOperations& go_ref, const str name) : go(go_ref), name(name) {}
graph Identity::forward(const graph& X) 
{
        auto node = std::make_shared<NodeBackProp>(name,X->dim[0],X->dim[1],X->dim[2],X->dim[3],1);
        go.GB += (double)(node->total) * sizeof(float) / (pow(2,30));
        node->inputs = {X};
        node->forward  = [=](){cudaMemcpy(node->output, X->output, node->total*sizeof(float), cudaMemcpyDeviceToDevice);};
        node->backward = [=](){cudaMemcpy(X->grad,  node->grad, X->total*sizeof(float), cudaMemcpyDeviceToDevice);};
        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};
        return node;
}

Linear::Linear(GraphOperations &go_ref, const int input, const int output, const str name) : go(go_ref), in(input), out(output) 
{   
    if (name != "") op_name = name;
    W1 =  new AdamParameter(name + " W1",1,1,1,in,out);
    B1 =  new AdamParameter(name + " B1",1,1,1,1,out);     
}
void Linear::save(std::ofstream& f) const{W1->save(f);B1->save(f);}
void Linear::load(std::ifstream& f){W1->load(f);B1->load(f);}
graph Linear::forward(const graph & X)
{   
        if(X->dim[3] != W1->dim[2])
        {
            std::cout << "Shape Mismatch in Linear Layer of " << X->op_name <<": \n";
            std::cout << "Dimensions are input:  (" << X->dim[2] << "," << X->dim[3] << ") and (" << W1->dim[2] << ","<<W1->dim[3] << ") \n";
            std::exit(1); 
        }

        const int batch = X->dim[0];
        const int channels = X->dim[1];
        const int row = X->dim[2];
        const int col = X->dim[3];
        const int tpb = THREADSPERBLOCK;
        auto node = std::make_shared<NodeBackProp>(op_name, batch, channels, row, out, 1);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((out+BLOCK_SIZE-1)/BLOCK_SIZE,(row+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
        dim3 grid_dA((col+BLOCK_SIZE-1)/BLOCK_SIZE,(row+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);   
        dim3 grid_dB((out+BLOCK_SIZE-1)/BLOCK_SIZE,(col+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);

        node->inputs = {X};

        node->forward = [=]()
        {   
            bcmm<<<grid, block>>>(X->output, W1->output, node->output,batch,channels, row, col, out, 0, 3, 0, 3);
            //CheckError("MatMul... X*W1 in Linear Layer forward");

            BCumAdd<<<(tpb+batch*channels*row-1)/tpb, tpb>>>(node->output, B1->output,batch,channels, row, out); 
            //CheckError("Add... X*W1+B1 in Linear Layer forward");

        };

        node->backward= [=]()
        {
            bmmABT<<<grid_dA, block>>>(node->grad, W1->output, X->grad, batch, row, out, col,1,3,0,3);
            //CheckError("MatMul... ∂Z*W^T in Linear Layer backward");

            bmmATB<<<grid_dB, block>>>(X->output, node->grad, W1->grad, batch, row, col, out,1,3,3,0);
            //CheckError("MatMul... X^T*∂Z in Linear Layer backward");

            BCompress<<<out, tpb>>>(node->grad, B1->grad, batch, channels, row, out);
            //CheckError("Compress... Squeeze(∂Z)->∂b in Lineary Layer backward");
            
            
        };
        
        node->free = [=](){node->clear();};

        node->serious_free = [=]()
        {
            W1->clear();
            B1->clear();
        };

        node->zero_grad = [=]()
        {
            Zerograd(node);
            Zerograd(W1);
            Zerograd(B1);
        };
        
        node->accumulate = [=](double* global_scale)
        {
            W1->accumulate_grad(global_scale);
            B1->accumulate_grad(global_scale);
        };

        node->clipnorm = [=](const double* global_scale)
        {
            W1->gradnorm(global_scale);
            B1->gradnorm(global_scale);
        };

        node->updateParams = [=]()
        {
            W1->update();
            B1->update();
        };

        node->printparams = [=]()
        {
        printHeadGPU(W1);
        printHeadGPU(B1);
        };

        return node;

}

Convolute2D::Convolute2D(GraphOperations&go_ref, int Input, int Output, int C, int D, int stride, int padding, str param) 
: go(go_ref), out(Output), inp(Input), c(C), d(D), pad(padding), stride(stride), name(param)
{   
    
        weights = new AdamParameter(name+ " Weight ", 1, out, inp, c, d);
        bias    = new AdamParameter(name+ " Bias ", 1, 1, out, 1, 1);
}
void Convolute2D::save(std::ofstream& f) const{weights->save(f); bias->save(f);}
void Convolute2D::load(std::ifstream& f){weights->load(f); bias->load(f);}
graph Convolute2D::forward(const graph& X)
{   
    if(X->dim[1] != inp)
    {
        std::cout << "Dimension Mismatch! in "<< name <<": \n"; Dimension(X);
        std::cout << "Actual input (Batch x depth): (" << X->dim[0] << "x" << X->dim[1] <<  ") \n";
        std::cout << "Expected input (Batch x depth): (" << X->dim[0] << "x" << inp <<  ") \n"; std::exit(1);
    }
    
    const int batch = X->dim[0];
    const int a = X->dim[2];  
    const int b = X->dim[3];  
    const int outR = (2 * pad + a - c) / stride + 1;
    const int outC = (2 * pad + b - d) / stride + 1;    

    auto node = std::make_shared<NodeBackProp>(name, batch, out, outR, outC, 1);
    node->inputs = {X};
    go.GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;

    dim3 block_fwd(8, 8, 16);
    dim3 block_wgt(16,16,4);

    dim3 grid_forward((outC + block_fwd.x - 1) / block_fwd.x, (outR + block_fwd.y - 1) / block_fwd.y, (batch * out + block_fwd.z - 1) / block_fwd.z);
    dim3 grid_weight_grad((out + block_wgt.x - 1) / block_wgt.x, (inp + block_wgt.y - 1) / block_wgt.y, (c * d + block_wgt.z - 1) / block_wgt.z);
    dim3 grid_input_grad((b + block_fwd.x - 1) / block_fwd.x, (a + block_fwd.y - 1) / block_fwd.y,(batch * inp + block_fwd.z - 1) / block_fwd.z);

    
    node->forward = [=]()
    {
        //isNan(X);
        CV2D<<<grid_forward, block_fwd>>>(X->output,weights->output,bias->output,node->output,batch,out,inp,a,b,c,d,pad,stride);
        //CheckError("Forward Convolution in " + name);
    };
    
    node->backward = [=]()
    {
        //isNan(node, 1);
        GV2D<<<grid_weight_grad, block_wgt>>>(X->output,node->grad,weights->grad,batch,out,inp,a,b,c,d,pad,stride);
        Channel_Squeeze1D<<<out,tpb>>>(node->grad,bias->grad,batch,out,outR,outC);
        CV2D_GradInput<<<grid_input_grad, block_fwd>>>(node->grad,weights->output,X->grad,batch,out,inp,a,b,c,d,outR, outC,pad, stride);
        //CheckError("Weight + Bias + Input Gradient in " + name);

    };

    node->zero_grad = [=]()
    {
        Zerograd(node);
        Zerograd(weights);
        Zerograd(bias);
    };
    
    node->free = [=](){node->clear();};
    
    node->serious_free = [=]()
    {
        weights->clear();
        bias->clear();
    };

    node->accumulate = [=](double* global_scale)
    {
        weights->accumulate_grad(global_scale);
        bias->accumulate_grad(global_scale);
    };
    
    node->clipnorm = [=](const double* global_scale)
    {
        weights->gradnorm(global_scale);
        bias->gradnorm(global_scale);
    };
    
    node->updateParams = [=]()
    {
        weights->update();
        bias->update();
    };
    
    node->printparams = [=]()
    {
        printHeadGPU(weights);
        printHeadGPU(bias);
    };

    return node;
}

Convolute2DT::Convolute2DT(GraphOperations& go_ref, int Input, int Output, int C, int D, int stride, int padding, str param) 
    : go(go_ref), out(Output), inp(Input), c(C), d(D), pad(padding), stride(stride), name(param)
{   
    weights = new AdamParameter(name + " Weight ", 1, out, inp, c, d);
    bias    = new AdamParameter(name + " Bias ", 1, 1, out, 1, 1);
}
void Convolute2DT::save(std::ofstream& f) const{weights->save(f); bias->save(f);}
void Convolute2DT::load(std::ifstream& f) {weights->load(f); bias->load(f);}
graph Convolute2DT::forward(const graph& X)
{   
    if(X->dim[1] != inp)
    {
        std::cout << "Dimension Mismatch! in " << name << ": \n";
        Dimension(X);
        std::cout << "Actual input (Batch x depth): (" << X->dim[0] << "x" << X->dim[1] <<  ") \n";
        std::cout << "Expected input (Batch x depth): (" << X->dim[0] << "x" << inp <<  ") \n";
        std::exit(1);
    }
    
    const int batch = X->dim[0];
    const int inp_h = X->dim[2];
    const int inp_w = X->dim[3];
    const int out_h = (inp_h - 1) * stride - 2 * pad + c;
    const int out_w = (inp_w - 1) * stride - 2 * pad + d;
    
    auto node = std::make_shared<NodeBackProp>(name, batch, out, out_h, out_w, 1);
    node->inputs = {X};
    go.GB += (double)(node->total) * sizeof(float) / (pow(2, 30));
    
    const int tpb = THREADSPERBLOCK;
    dim3 block(8, 8, 16);
    dim3 grid_fwd((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y,(batch * out + block.z - 1) / block.z);
    dim3 grid_igrad((inp_w + block.x - 1)/block.x,(inp_h + block.y - 1) / block.y, (batch * inp + block.z - 1) / block.z);

    node->forward = [=]()
    {

        //isNan(X);
        CVT2D<<<grid_fwd, block>>>(X->output, weights->output, bias->output, node->output, batch, out, inp, inp_h, inp_w, c, d, pad, stride);
        //CheckError("ConvTranspose2D Forward Kernel");
        //isNan(node);
    };

    node->backward = [=]()
    {
        isNan(node, 1);

        GVT2D<<<out,tpb>>>(X->output, node->grad, weights->grad, batch, out, inp, inp_h, inp_w, c, d, pad, stride);
        Channel_Squeeze1D<<<out,tpb>>>(node->grad, bias->grad, batch, out, out_h, out_w);
        //CheckError("Gradient w.r.t W and Bias for ConvTranspose2D");

        CVT2D_GradInput<<<grid_igrad, block>>>(node->grad, weights->output, X->grad, batch, out, inp, inp_h, inp_w, c, d, pad, stride);
        //CheckError("Gradient w.r.t input for ConvTranspose2D");
    };

    node->zero_grad = [=]()
    {
        Zerograd(node);
        Zerograd(weights); 
        Zerograd(bias);
    };

    node->free = [=]()
    {
        node->clear();
    };

    node->serious_free = [=]()
    {
        weights->clear();
        bias->clear();
    };

    node->accumulate = [=](double* global_scale)
    {
        weights->accumulate_grad(global_scale);
        bias->accumulate_grad(global_scale);
    };

    node->clipnorm = [=](const double* global_scale)
    {
        weights->gradnorm(global_scale);
        bias->gradnorm(global_scale);
    };

    node->updateParams = [=]()
    {
        weights->update();
        bias->update();
    };

    node->printparams = [=]()
    {
        printHeadGPU(weights);
        printHeadGPU(bias);
    };

    return node;
}

TimeMLPBlock::TimeMLPBlock(GraphOperations &go_ref, const int t_embed_dim, const int t_hidden): go(go_ref)
{
    L0 = new Linear(go, t_embed_dim, t_hidden, "Time MLP L0");
    L1 = new Linear(go, t_hidden, t_hidden, "Time MLP L1");
}
graph TimeMLPBlock::forward(const graph & X)
{
        auto first = L0->forward(X);
        auto activate = go.SILU(first); 
        auto node = L1->forward(activate);
        return node;
}
void TimeMLPBlock::save(std::ofstream& f) const
{
    L0->save(f);
    L1->save(f);
}
void TimeMLPBlock::load(std::ifstream& f)
{
    L0->load(f);
    L1->load(f);
}

void Noise(const graph & input)
{
    std::random_device rd;
    const uint64_t seed =  ((uint64_t)rd() << 32) | rd();
    GaussianNoise<<<(input->total+THREADSPERBLOCK-1)/THREADSPERBLOCK,THREADSPERBLOCK>>>(input->output, input->total, seed);
    // CheckError("Addition of Gaussian noise in noise kernel");
}


