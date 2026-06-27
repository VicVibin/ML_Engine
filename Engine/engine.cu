#include "engine.h"

float ReadValueAt(const graph& X, const int& idx, const bool grad)
{
    float out;
    if(!grad) cudaMemcpy(&out, X->output + idx, sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(&out, X->grad + idx, sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}

void WriteValueAt(const graph& X, const float value, const int& idx, const bool grad)
{
    // @brief: Writes to a graph output for flag 0, and 1 for grad
    if (!grad) cudaMemcpy(X->output + idx, &value, sizeof(float), cudaMemcpyHostToDevice);
    else cudaMemcpy(X->grad + idx, &value, sizeof(float), cudaMemcpyHostToDevice);
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
    else  Write<<<bpg, tpb>>>(theta, input, total);
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
    {
        std::cerr << "Reshape cannot change total size of tensor\n";
        std::cout<< "Dimensions for node: " << op_name << "\n";
        std::cout << "(";for(const auto& d : dim) std::cout << " x " << d <<"\t";std::cout<< ") \n";
        std::cout << "Attempting to reshape to " << "("; for(const auto& d : new_dims) std::cout << " x " << d <<"\t";std::cout<< ") \n";
        std::exit(1);
    }
    
    for(int i=0;i<4;++i){dim[i] = new_dims[i];}
}

void NodeBackProp::reshape(int arr[], const int size)
{
    if(size !=4)
    {
        std::cerr << "Reshape only supports 4D tensors\n";
        std::exit(1);
    }
    
    int new_total = arr[0]*arr[1]*arr[2]*arr[3];
    
    if(new_total != total)
    {
        std::cerr << "Reshape cannot change total size of tensor\n";
        std::cout<< "Dimensions for node: " << op_name << "\n";
        std::cout << "(";for(const auto & d : dim) std::cout << " x " << d <<"\t"; std::cout<< ") \n";
        std::cout << "Attempting to reshape to " << "("; for(int i=0; i<size; ++i) std::cout << " x " << arr[i] <<"\t";std::cout<< ") \n";
        std::exit(1);
    }
    
    for(int i=0;i<4;++i){dim[i] = arr[i];}
}

AdamParameter::AdamParameter(str n, int out, int in, int row, int col, double norm) : NodeBackProp(n, out, in, row, col,1), t(1), b1(0.9), b2(0.999), epsilon(1e-8), group_norm(norm), weight_decay(0.01f)
{
        std::random_device rd;
        const uint64_t seed =  ((uint64_t)rd() << 32) | rd();
        SafeCudaMalloc("M-matrix of AdamParameter",m,total);
        SafeCudaMalloc("V-matrix of AdamParameter",v,total);
        const int tpb = THREADSPERBLOCK; 
        const int bpg = (total+tpb-1) / tpb;
        Standard_Weights<<<bpg,tpb>>>(output, total, sqrtf(XAVIER/(in*row*col)), seed); 
        fillKernel<<<bpg,tpb>>>(m,0.0f,total);
        fillKernel<<<bpg,tpb>>>(v,0.0f,total);
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

void AdamParameter::load(std::ifstream& f){
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

void AdamParameter::operator=(const AdamParameter& other)
{
    if(dim[0] != other.dim[0] || dim[1] != other.dim[1] || dim[2] != other.dim[2] ||  dim[3] != other.dim[3])
    {
        std::cout << "Shape Mismatch in equator operator of AdamParameter " << this->op_name<< " and " << other.op_name;
        for(int i=0;i<4; ++i){ std::cout << " x " << this->dim[i]<<"\t";} std::cout << ") " << this->op_name << "shape \n";
        for(int i=0;i<4; ++i){ std::cout << " x " << other.dim[i]<<"\t";} std::cout << ") " << other.op_name<< "shape \n";
        std::exit(1);
    }
    Write<<<(total+THREADSPERBLOCK-1)/THREADSPERBLOCK,THREADSPERBLOCK>>>(other.output, output, other.total);
    CheckError(this->op_name + " = " + other.op_name);
};

void AdamParameter::update(const float lr, const bool W)
{   
        const int tpb = THREADSPERBLOCK;
        const int bpg = (total + tpb - 1) / tpb;
        //isNan("Gradient of " + op_name, grad, total);
        if(W) AdamWUpdate<<<bpg, tpb>>>(output, grad, total, t, m, v, b1, b2,epsilon,weight_decay,lr);
        else  AdamUpdate<<<bpg, tpb>>>(output, grad, total,t, m, v, b1, b2, epsilon, lr);
        //CheckError("AdamUpdate in AdamParameter update");
        t++;
};

void AdamParameter::accumulate_grad(double* global_scale)
{
        const int tpb = THREADSPERBLOCK;
        const int bpg = (total + tpb - 1) / tpb;
        SumSquared<<<bpg, tpb>>>(global_scale, grad, total);
        //CheckError("SSWarp in Gradient Norm of " + op_name); 
};

void AdamParameter::gradnorm(const double* global_scale){ScalePtr(grad, global_scale, total, 1);};

void Dimension(graph X)
{   std::cout<< "Dimensions for node: " << X->op_name << " "; printf("(%i x %i x %i x %i) \n", X->dim[0],X->dim[1],X->dim[2],X->dim[3]);
}

void Dimension(AdamParameter X)
{  std::cout<< "Dimensions for node: " << X.op_name << " " ; printf("(%i x %i x %i x %i) \n", X.dim[0],X.dim[1],X.dim[2],X.dim[3]);
}

void Zerograd(AdamParameter X)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (tpb + X.total-1) / tpb;
    fillKernel<<<bpg,tpb>>>(X.grad, 0.0f, X.total);

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
    const str val = (type == 0) ? "output" : "grad";
    CheckError(X->op_name + val + " has nan value");
    return;
}

void isNan(AdamParameter X, const int type )
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (X.total + tpb-1)/tpb;
    if (type == 0) ISNAN<<<bpg,tpb>>>(X.output, X.total);
    else ISNAN<<<bpg,tpb>>>(X.grad, X.total);
    const str val = (type == 0) ? "output" : "grad";
    CheckError(X.op_name + val + " has nan value");
    return;
}

void printGPU(const graph X, const int type)
{
    int batch = X->dim[0];
    int ch     = X->dim[1];
    int rows   = X->dim[2];
    int cols   = X->dim[3];
    int total  = X->total;
    std::vector<float> CPU(total);
    if (type == 0) cudaMemcpy(CPU.data(), X->output, total * sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(CPU.data(), X->grad, total * sizeof(float), cudaMemcpyDeviceToHost);
    
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

}

void printGPU(AdamParameter X, const int type)
{
     int batch = X.dim[0];
    int ch     = X.dim[1];
    int rows   = X.dim[2];
    int cols   = X.dim[3];
    int total  = X.total;

    std::vector<float> CPU(total);

    if (type == 0) cudaMemcpy(CPU.data(), X.output, total * sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(CPU.data(), X.grad, total * sizeof(float), cudaMemcpyDeviceToHost);

    
    if (type == 0) std::cout << "Printing dimensions for node " <<X.op_name << "->output \n";
    else std::cout << "Printing dimensions for node " <<X.op_name << "->grad \n";
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
}

void printHeadGPU(const graph X, const int type)
{
    int ch     = X->dim[1];
    int rows   = X->dim[2];
    int cols   = X->dim[3];
    int total  = X->total;

    std::vector<float> CPU(total);
    if (type == 0) cudaMemcpy(CPU.data(), X->output, total * sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(CPU.data(), X->grad, total * sizeof(float), cudaMemcpyDeviceToHost);

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
}

void printHeadGPU(AdamParameter X, const int type)
{
    int ch     = X.dim[1];
    int rows   = X.dim[2];
    int cols   = X.dim[3];
    int total  = X.total;

    std::vector<float> CPU(total);
    if (type == 0) cudaMemcpy(CPU.data(), X.output, total * sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(CPU.data(), X.grad, total * sizeof(float), cudaMemcpyDeviceToHost);

    if (type == 0) std::cout << "Printing dimensions for node " <<X.op_name << "->output \n";
    else std::cout << "Printing dimensions for node " <<X.op_name << "->grad \n";
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

    Write<<<bpg,tpb>>>(base->output, input->output, input->total);
    //CheckError("CudaMemcpy");

    GaussianNoise<<<bpg,tpb>>>(target->output, 0.f, 1.f, target->total, seed);
    //CheckError("Gaussian Noise in preparation");

    AddNoise<<<bpg, tpb>>>(input->output, target->output,t, T, input->total);
    //CheckError("Addition of Noise in preparation");
}

int ArgMaxToCPU(const graph& input, int* X)
{
    for(int i = 0; i < 3; i++)
    {if (input->dim[i] != 1)
        {
            printf("Dimension does not match kernel's required (1 x 1 x 1 x D)"); 
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

int ArgMaxToCPU(const graph& input)
{
    int* X;
    SafeCudaMalloc("X", X, 1);
    for(int i = 0; i < 3; i++)
    {if (input->dim[i] != 1)
        {
            printf("Dimension does not match kernel's required (1 x 1 x 1 x D)"); 
            Dimension(input);
            std::exit(1);
        }
    }
    
    ArgMax<<<1,1>>>(input->output, X, input->total);
    int max_id;
    cudaMemcpy(&max_id, X, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(X);
    return max_id;
}

int TopKSampleToCPU(const graph& input, const int k)
{
    int* X;
    SafeCudaMalloc("X", X, 1);
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
    cudaFree(X);
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

graph GraphOperations::Copy(const graph& X)
{
    auto node = std::make_shared<NodeBackProp>("Copy of " + X->op_name, X->dim[0], X->dim[1], X->dim[2], X->dim[3],1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    node->forward = [=](){Write<<<bpg,tpb>>>(X->output, node->output, node->total);};
    node->backward= [=](){Accumulate<<<bpg,tpb>>>(node->grad, X->grad, node->total);};
    node->zero_grad = [=](){Zerograd(node);};
    node->free = [=](){node->clear();};
    return node;
}

graph GraphOperations::like(const graph& X, const str name) // Does not track X ==== Made to track X last night... Big Change
{
    auto node = std::make_shared<NodeBackProp>(name, X->dim[0], X->dim[1], X->dim[2], X->dim[3],1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    node->forward = [=](){return;};
    node->backward= [=](){return;};
    node->zero_grad = [=](){Zerograd(node);};
    node->free = [=](){node->clear();};
    return node;
}

graph GraphOperations::identity(const graph& X) // Does not track X;
{
    auto node = std::make_shared<NodeBackProp>(X->op_name, X->dim[0], X->dim[1], X->dim[2], X->dim[3],1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float)/(1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total + tpb-1)/tpb;
    node->forward = [=](){Write<<<bpg,tpb>>>(X->output, node->output, node->total);};
    node->backward= [=](){Write<<<bpg,tpb>>>(node->grad, X->grad, node->total);};
    node->zero_grad = [=](){Zerograd(node);};
    node->free = [=](){node->clear();};
    return node;
}

graph GraphOperations::ones_like(const graph& X)
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];    
    auto node = std::make_shared<NodeBackProp>("Ones of " + X->op_name, a, b, c, d, 1);
    GB += (double)(node->total) / (1ULL << 30);
    fillKernel<<<(node->total+THREADSPERBLOCK-1)/THREADSPERBLOCK,THREADSPERBLOCK>>>(node->output, 1.0f, node->total);
    CheckError("Ones Like Kernel");
    node->inputs = {X};
    node->zero_grad = [=](){Zerograd(node);};
    node->free = [=](){node->clear();};
    return node;
}

// ==================== Linear Algebra Ops =================== //

graph GraphOperations::identity_like(const graph& X)
{

    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    if(c != d)
    {
        std::cout << "Identity_like matrices can only be constructed as squares, input" << X->op_name << " is not a square \n";
        std::exit(1); 
    }
    auto node = std::make_shared<NodeBackProp>("Identity Matrix",a,b,c,d,1);
    GB += (double)(node->total) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    identityKernel<<<bpg,tpb>>>(node->output,a,b,c);
    CheckError("IdentityKernel");
    node->inputs = {X};
    node->zero_grad = [=](){Zerograd(node);};
    node->free = [=](){node->clear();};
    return node;
}

std::pair<graph, graph> GraphOperations::LU_factorize(const graph& X) // Returns LU Matrices;
{
    const int B = X->dim[0], C = X->dim[1], H = X->dim[2], W =  X->dim[3];
    // If H > W : (H x W) | (W x W)  for L and U
    // if H < W : (H x H) | (W x H)  for L and U;
    const int l_w = (H > W) ? W : H;
    const int u_h = (H > W) ? W : H;
    auto L = std::make_shared<NodeBackProp>("L of " + X->op_name, B,C,H,l_w,1);
    auto U = std::make_shared<NodeBackProp>("U of " + X->op_name, B,C,u_h,W,1);
    L->inputs = {X}; U->inputs = {X};
    GB += (double)(2 * X->total) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (X->total+tpb-1)/tpb;
    const int lpg = (L->total+tpb-1)/tpb;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((W+block.x-1)/block.x,(H+block.y-1)/block.y, B*C);

    L->forward = [=]()
    {
        if(L->forward_called) return; L->forward_called = true; U->forward_called = true;
        Write<<<bpg,tpb>>>(X->output, U->output, X->total);
        identityKernel<<<lpg,tpb>>>(L->output, B,C,H);
        lu_factorization<<<bpg,tpb>>>(L->output, U->output, B,C,H,W);
    };

    U->forward = [=]()
    {
        if(U->forward_called) return; 
        U->forward_called = true; L->forward_called = true;
        Write<<<bpg,tpb>>>(X->output, U->output, X->total);
        identityKernel<<<lpg,tpb>>>(L->output, B,C,H);
        lu_factorization<<<bpg,tpb>>>(L->output, U->output, B,C,H,W);
    };

    U->backward = [=]()
    {
        bcmm<<<grid,block>>>(L->output, U->grad, X->grad,B,C,H,u_h,W,1);
        //CheckError("LU U portion Gradient to X");
    };

    L->backward = [=]()
    {
        bcmm<<<grid,block>>>(L->grad, U->output,X->grad,B,C,H,u_h,W,1);
        //isNan(node,1);
    };

    L->free = [=](){L->clear();}; U->free = [=](){U->clear();};
    L->zero_grad = [=]() {Zerograd(L); U->forward_called = false; L->forward_called = false;};
    U->zero_grad = [=](){Zerograd(U); U->forward_called = false; L->forward_called = false;};
    return {L,U};
}

graph GraphOperations::Inverse(const graph& X)
{
    if(X->dim[2] != X->dim[3])
    {
        std::cout << "Inverse can only be applied to square matrices, input " << X->op_name << " is not square \n";
        Dimension(X);
        std::exit(1);
    }
    const int B = X->dim[0], C = X->dim[1], H = X->dim[2];
    auto node = std::make_shared<NodeBackProp>("Inverse of " + X->op_name, B, C, H, H, 1);
    auto Y = like(X, "Intermediate for Inverse");
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((H + block.x - 1) / block.x, (H + block.y - 1) / block.y, B*C);
    float* U;
    SafeCudaMalloc("U", U, X->total);
    node->forward = [=]()
    {
        //isNan(X);
        Write<<<bpg,tpb>>>(X->output, U, X->total);
        identityKernel<<<bpg,tpb>>>(Y->output, B,C,H);
        lu_factorization<<<bpg,tpb>>>(Y->output, U, B,C,H,H);
        solveLX<<<bpg,tpb>>>(Y->output, Y->grad, B, C, H);
        solveUy<<<bpg,tpb>>>(U, Y->grad, node->output, B, C, H);    
        //CheckError("Inverse forward");
    };

    node->backward = [=]()
    {
        //isNan(node,1);
        // Derivation: Z = X^-1, XZ = I, dX*Z + X*dZ = 0, dX*Z=-X*dZ, dX = -X*dZ*X. dX = -X*dZ*X
        bcmm<<<grid, block>>>(X->output, node->grad, Y->grad, B, C, H,H,H); // buff->grad = X*dZ
        bcmm<<<grid, block>>>(Y->grad, X->output, X->grad,B, C, H,H,H,1,3,3,3,-1.0f); // X->grad = -X*dZ*X
        CheckError("Inverse Backward using bcmm kernels");
    };

    node->free = [=](){node->clear(); Y->clear(); cudaFree(U);};
    //node->zero_grad = [=](){Zerograd(node);};  Already defined by like     
    return node;
}

graph GraphOperations::InverseAug(const graph& X)
{
    if(X->dim[2] != X->dim[3])
    {
        std::cout << "Inverse can only be applied to square matrices, input " << X->op_name << " is not square \n";
        Dimension(X);
        std::exit(1);
    }

    const int B = X->dim[0], C = X->dim[1], H = X->dim[2];
    auto node = like(X, "Inverse of " + X->op_name);
    auto    Y = like(X, "Intermediate for Inverse");
    node->inputs = {X}; Y->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((H + block.x - 1) / block.x, (H + block.y - 1) / block.y, B*C);
    node->forward = [=]()
    {
        //isNan(X);
        identityKernel<<<bpg,tpb>>>(node->output, B,C,H);
        Write<<<bpg,tpb>>>(X->output, Y->output, X->total);
        augmentedRowReduction<<<(B*C+tpb-1)/tpb,tpb>>>(Y->output, node->output, B, C, H);
        //CheckError("Inverse forward");
    };

    node->backward = [=]()
    {
        //isNan(node,1);
        // Derivation: Z = X^-1, XZ = I, dX*Z + X*dZ = 0, dX*Z=-X*dZ, dX = -X*dZ*X. dX = -X*dZ*X
        bcmm<<<grid, block>>>(X->output, node->grad, Y->grad, B, C, H,H,H); // buff->grad = X*dZ
        bcmm<<<grid, block>>>(Y->grad, X->output, X->grad,B, C, H,H,H,1,3,3,3,-1.0f); // X->grad = -X*dZ*X
        CheckError("Inverse Backward using bcmm kernels");
    };

    node->free = [=](){node->clear(); Y->clear();};
    //node->zero_grad = [=](){Zerograd(node);};   Already defined by like   
    return node;
}

graph GraphOperations::Determinant(const graph& X)
{    
    if(X->dim[2] != X->dim[3])
    {
        std::cout << "Determinant can only be applied to square matrices, input " << X->op_name << " is not square \n";
        Dimension(X);
        std::exit(1);
    }
    
    const int B = X->dim[0], C = X->dim[1], H = X->dim[2];
    auto node = std::make_shared<NodeBackProp>("Determinant of " + X->op_name, B, C, 1, 1, 1);
    auto L = like(X, "L");
    auto U = like(X, "U");
    auto Y = like(X, "Intermediate for Determinant and Inverse");
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (X->total+tpb-1)/tpb;

    node->forward = [=]()
    {
        //isNan(X);
        identityKernel<<<bpg, tpb>>>(L->output, B, C, H);
        Write<<<bpg,tpb>>>(X->output,U->output, X->total);
        lu_factorization<<<bpg,tpb>>>(L->output, U->output, B, C, H, H);
        scalar_diagonal_product<<<bpg,tpb>>>(U->output, node->output, B, C, H);
        //CheckError("Determinant forward");    
    };

    node->backward = [=]()
    {
        //isNan(node,1);
        solveLX<<<bpg,tpb>>>(L->output, Y->output, B, C, H);
        solveUy<<<bpg,tpb>>>(U->output, Y->output, U->grad, B, C, H); // (X^-1 stored in U->grad)
        transpose<<<bpg,tpb>>>(U->grad, L->grad, B, C, H, H); // U->grad = (X^-1) | L->grad = (X^-1)^T
        AccumulateBC<<<bpg,tpb>>>(L->grad, node->output, X->grad, B, C, H, H); // X->grad += (X^-1)^T * det(X)
        //CheckError("Determinant Backward");
    };

    node->free = [=](){node->clear(); L->clear(); U->clear(); Y->clear();};
    node->zero_grad = [=](){Zerograd(node);};      
    return node;
}

std::tuple<graph,graph,graph> GraphOperations::Schurr(const graph& X)
{

    if(X->dim[2] != X->dim[3])
    {
        std::cout << "Eigenvalue decomposition can only be applied to square matrices, input " << X->op_name << " is not square \n";
        Dimension(X);
        std::exit(1);
    }
    
    const int B = X->dim[0], C = X->dim[1], H = X->dim[2];
    const int UPPER_BOUND  = 2 * int(log(H)) + 5;
    auto P = GraphOperations::like(X, "P of " + X->op_name);
    auto D = GraphOperations::like(X, "D of " + X->op_name);
    auto P_inv = GraphOperations::like(X, "P inverse of " + X->op_name);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (X->total+tpb-1)/tpb;
    float *Q, *R, *L, *U, *Y; 
    SafeCudaMalloc("Q", Q, X->total);
    SafeCudaMalloc("R", R, X->total);
    SafeCudaMalloc("L", L, X->total);
    SafeCudaMalloc("U", U, X->total);
    SafeCudaMalloc("Y", Y, X->total);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((H+block.x-1)/block.x, (H+block.y-1)/block.y, B * C);
    GB += (double)(X->total) / (1 << 27); // 8*X / 2^30

    // P->inputs = {X};  D->inputs = {X}; P_inv->inputs = {X};  Already tracked by like
    
    P->forward = [=]()
    {
        if(P->forward_called) return;

        // ======================  Calculate P and D =============================== //
        Write<<<bpg,tpb>>>(X->output, Q, X->total); // X to Q;
        gram_schmidt_cols<<<bpg,tpb>>>(Q, R, B, C, H); // Q, R
        Write<<<bpg, tpb>>>(Q, U, X->total); // U = Q_0
        bcmm<<<grid,block>>>(R, Q, D->output, B, C, H,H,H); // D = RQ
        for(int i = 0; i < UPPER_BOUND; ++i)
        {
            Write<<<bpg,tpb>>>(D->output, Q, X->total);               // A_k^T to Q
            gram_schmidt_cols<<<bpg,tpb>>>(Q, R, B, C, H);                  // Q^T, R;
            bcmm<<<grid,block>>>(U, Q, P->output, B, C, H, H, H);   // P =  U_i-1 * Q_i;
            bcmm<<<grid,block>>>(R, Q, D->output, B, C, H,H,H);     // D = RQ
            Write<<<bpg,tpb>>>(P->output, U, X->total);                // L = P
        }// At the end U and P both store P;

        // ======================= Calculate P^-1 =========================== //
        identityKernel<<<bpg,tpb>>>(L, B, C, H);
        lu_factorization<<<bpg,tpb>>>(L, U, B, C, H, H);
        solveLX<<<bpg,tpb>>>(L, Y, B, C, H);
        solveUy<<<bpg,tpb>>>(U, Y, P_inv->output,B, C,H);
        P->forward_called = true; D->forward_called = true; P_inv->forward_called = true;
    
    };
    
    D->forward = [=]()
    {   
        if(D->forward_called) return;

        // ======================  Calculate P and D =============================== //
        Write<<<bpg,tpb>>>(X->output, Q, X->total); // X to Q;
        gram_schmidt_cols<<<bpg,tpb>>>(Q, R, B, C, H); // Q, R
        Write<<<bpg, tpb>>>(Q, U, X->total); // U = Q_0
        bcmm<<<grid,block>>>(R, Q, D->output, B, C, H,H,H); // D = RQ
        for(int i = 0; i < UPPER_BOUND; ++i)
        {
            Write<<<bpg,tpb>>>(D->output, Q, X->total);               // A_k^T to Q
            gram_schmidt_cols<<<bpg,tpb>>>(Q, R, B, C, H);                  // Q^T, R;
            bcmm<<<grid,block>>>(U, Q, P->output, B, C, H, H, H);   // P =  U_i-1 * Q_i;
            bcmm<<<grid,block>>>(R, Q, D->output, B, C, H,H,H);     // D = RQ
            Write<<<bpg,tpb>>>(P->output, U, X->total);                // L = P
        }// At the end U and P both store P;

        // ======================= Calculate P^-1 =========================== //
        identityKernel<<<bpg,tpb>>>(L, B, C, H);
        lu_factorization<<<bpg,tpb>>>(L, U, B, C, H, H);
        solveLX<<<bpg,tpb>>>(L, Y, B, C, H);
        solveUy<<<bpg,tpb>>>(U, Y, P_inv->output,B, C,H);
        P->forward_called = true; D->forward_called = true; P_inv->forward_called = true;
    
    };

    P_inv->forward = [=]()
    { 
        if(P_inv->forward_called) return;

        // ======================  Calculate P and D =============================== //
        Write<<<bpg,tpb>>>(X->output, Q, X->total); // X to Q;
        gram_schmidt_cols<<<bpg,tpb>>>(Q, R, B, C, H); // Q, R
        Write<<<bpg, tpb>>>(Q, U, X->total); // U = Q_0
        bcmm<<<grid,block>>>(R, Q, D->output, B, C, H,H,H); // D = RQ
        for(int i = 0; i < UPPER_BOUND; ++i)
        {
            Write<<<bpg,tpb>>>(D->output, Q, X->total);               // A_k^T to Q
            gram_schmidt_cols<<<bpg,tpb>>>(Q, R, B, C, H);                  // Q^T, R;
            bcmm<<<grid,block>>>(U, Q, P->output, B, C, H, H, H);   // P =  U_i-1 * Q_i;
            bcmm<<<grid,block>>>(R, Q, D->output, B, C, H,H,H);     // D = RQ
            Write<<<bpg,tpb>>>(P->output, U, X->total);                // L = P
        }// At the end U and P both store P;

        // ======================= Calculate P^-1 =========================== //
        identityKernel<<<bpg,tpb>>>(L, B, C, H);
        lu_factorization<<<bpg,tpb>>>(L, U, B, C, H, H);
        solveLX<<<bpg,tpb>>>(L, Y, B, C, H);
        solveUy<<<bpg,tpb>>>(U, Y, P_inv->output,B, C,H);
        P->forward_called = true; D->forward_called = true; P_inv->forward_called = true;
    
    };

    P->backward = [=]()
    {   
        // A = PDP^-1  dA = (dP *DP^-1) + (...) + (...)
        bcmm<<<grid,block>>>(D->output, P_inv->output, Y, B, C, H, H, H);  // Y stores D * P^-1
        bcmm<<<grid,block>>>(P->grad, Y, X->grad, B, C, H, H, H, 1);       // dX += dP * Y
    };

    D->backward = [=]()
    {   
        // A = PDP^-1  dA = (...) + (P*dD*P^-1) + (...)
        bcmm<<<grid,block>>>(D->grad, P_inv->output, U, B, C, H, H, H);    // U stores dD*P^-1
        bcmm<<<grid,block>>>(P->output,  U, X->grad, B, C, H, H, H, 1);    // dX += P * U
    };

    P_inv->backward = [=]()
    {   
        // A = PDP^-1  dA = (...) + (...) + (P*D*dP^-1)
        bcmm<<<grid,block>>>(D->output, P_inv->grad, U, B, C, H, H, H);    // L stores D*dP^-1
        bcmm<<<grid,block>>>(P->output,  L, X->grad, B, C, H, H, H, 1);    // dX += P * L
    };

    P->zero_grad   =  [=](){Zerograd(P); P->forward_called = false; D->forward_called = false;  P_inv->forward_called = false;};
    D->zero_grad   =  [=](){Zerograd(D); P->forward_called = false; D->forward_called = false;  P_inv->forward_called = false;};
    P_inv->zero_grad= [=](){Zerograd(P_inv); P->forward_called = false; D->forward_called = false;  P_inv->forward_called = false;};
    
    P->free = [=]()
    {
        P->clear();  cudaFree(Q);
        cudaFree(R); cudaFree(L);
        cudaFree(U); cudaFree(Y);
    };
    // Free already defined for D and P_inv by like
    return {P,D,P_inv};
}

// ==================== Tensor Ops =================== //

graph GraphOperations::StandardNorm(const graph& X, const float max, const float mean, const float std)
{
    const int B = X->dim[0];
    const int C = X->dim[1];
    const int H = X->dim[2];
    const int W = X->dim[3];

    auto node = like(X, "Standard Normalization");
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;

    node->forward = [=]()
    {
        Write<<<bpg, tpb>>>(X->output, node->output, node->total);
        StdNorm<<<bpg,tpb>>>(node->output, max, mean, std, node->total);
    };

    node->backward = [=]()
    {
       Accumulate<<<bpg,tpb>>>(node->grad, X->grad, node->total, max * std);
    };

    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};    
    return node;
}

graph GraphOperations::Dropout(const graph& X, const float p, const bool eval)
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("Dropout " + X->op_name, a,b,c,d,1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    std::random_device *rd = new std::random_device();
    float* mask = nullptr;
    if(!eval) SafeCudaMalloc("Dropout Mask", mask, node->total);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    node->forward = [=]()
    {
        const uint64_t seed =  ((uint64_t)rd->operator()() << 32) | rd->operator()();
        if (!eval) dropoutKernel<<<bpg,tpb>>>(X->output, mask,node->output, node->total,p, seed);
        else Write<<<bpg,tpb>>>(X->output, node->output, node->total);
         //CheckError("Dropout forward");
    };

    node->backward = [=]()
    {
        if (!eval) dropoutKernel<<<bpg,tpb>>>(X->output, mask,node->output, node->total,p,0,1);
        else Write<<<bpg,tpb>>>(node->grad, X->grad, node->total);
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

graph GraphOperations::HeadifytoChannel(const graph& X, const int heads)
{
    if (X->dim[1] != 1 || X->dim[3] % heads != 0)
    {
        std::cerr << "Headify: channel must be 1 and embed must be divisible by heads\n";
        Dimension(X); std::exit(1);
    }
    const int b = X->dim[0], seq = X->dim[2], embed = X->dim[3], head_dim = embed / heads;
    auto node = std::make_shared<NodeBackProp>("Headified " + X->op_name, b, heads, seq, head_dim, 1);
    const int tpb = THREADSPERBLOCK;
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    node->forward  = [=]() mutable
    {
        HeadifyColChannel<<<(node->total+tpb-1)/tpb, tpb>>>(X->output,    node->output, b, heads, seq, head_dim, 0);
    };
    node->backward = [=](){ HeadifyColChannel<<<(node->total+tpb-1)/tpb, tpb>>>(node->grad,   X->grad,      b, heads, seq, head_dim, 1); };
    node->zero_grad = [=](){ Zerograd(node); };
    node->free      = [=](){ node->clear(); };
    return node;
}

graph GraphOperations::DeHeadify(const graph& X)
{
    const int b = X->dim[0], heads = X->dim[1], seq = X->dim[2], head_dim = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("Deheadified " + X->op_name, b, 1, seq, heads * head_dim, 1);
    const int tpb = THREADSPERBLOCK;
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    node->forward  = [=](){ HeadifyColChannel<<<(node->total+tpb-1)/tpb, tpb>>>(X->output,  node->output, b, heads, seq, head_dim, 2); };
    node->backward = [=](){ HeadifyColChannel<<<(node->total+tpb-1)/tpb, tpb>>>(node->grad, X->grad,      b, heads, seq, head_dim, 3); };
    node->zero_grad = [=](){ Zerograd(node); };
    node->free      = [=](){ node->clear(); };
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
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
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
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
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
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
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
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
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
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
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
        Accumulate<<<bpg,tpb>>>(node->grad, A->grad, node->total);
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

        auto node = std::make_shared<NodeBackProp>("Bias Add", batch, channels, a, b, 1);
        node->inputs = {A,B};
        GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
        const int tpb = THREADSPERBLOCK;
        const int bpg = (node->total+tpb-1)/tpb;

        node->forward = [=]()
        {
            //isNan(A); //isNan(B);
            Write<<<bpg,tpb>>>(A->output, node->output, node->total);
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
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
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

graph GraphOperations::ExpM(const graph& X, const float threshold)
{
    if(X->dim[2] != X->dim[3])
    {
        std::cout << "Exponentiate Matrix can only be applied to square matrices, input" << X->op_name << " is not a square \n";
        Dimension(X);
        std::exit(1); 
    }

    const int batch = X->dim[0];
    const int channels = X->dim[1];
    const int a = X->dim[2];
    auto node = std::make_shared<NodeBackProp>("Exponentiate", batch, channels, a, a, 1);
    auto buff = std::make_shared<NodeBackProp>("ExpM Buffer ", batch, channels, a, a, 1);

    auto gnode = std::make_shared<NodeBackProp>("Freschet grad", batch, channels, 2*a, 2*a, 1);
    auto gbuff = std::make_shared<NodeBackProp>("Freschet buff ",batch, channels, 2*a, 2*a, 1);

    node->inputs = {X};
    double* scale;
    double  scalar;
    int s, thresh_loop; float val;
    SafeCudaMalloc("Scale", scale, 1);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    const int nbpg =(gnode->total+tpb-1)/tpb;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((a+BLOCK_SIZE-1)/BLOCK_SIZE,(a+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels); 
    dim3 ggrid((2*a+BLOCK_SIZE-1)/BLOCK_SIZE,(2*a+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels); 
    
    node->forward = [=]() mutable
    {   
        isNan(X);
        SumSquared<<<bpg,tpb>>>(scale, X->output, X->total);
        isNan<double>("Scale", scale, 1);
        cudaMemcpy(&scalar, scale, sizeof(double), cudaMemcpyDeviceToHost);
        s = max(0.0, ceil(log2(sqrt(scalar))));
        val = .5f;
        thresh_loop = 2;
        while(val > threshold)
        {
            val /= (float)(thresh_loop);
            thresh_loop += 1;
        }
        scalar = 1.0 / (double)(1 << (int)s);  // FIX: was 2 << s
        identityKernel<<<bpg,tpb>>>(node->output, batch, channels, a); // I
        Scale_Write<<<bpg,tpb>>>(X->output, buff->output, node->total, scalar); // X / 2^s
        Accumulate<<<bpg,tpb>>>(buff->output, node->output, node->total); // 1 + X / 2^s
        for(int i = 2; i <= thresh_loop; i++)
        {
            // buff->grad = (X/2^s) * buff->output = (X/2^s)^i
            bcmm<<<grid,block>>>(X->output, buff->output, buff->grad, batch, channels, a, a, a, 0, 3, 3, 3, scalar);

            // result += (X/2^s)^i / i!
            Accumulate<<<bpg,tpb>>>(buff->grad, node->output, node->total, 1.0/std::tgamma(i+1));

            // buff = (X/2^s)^i  for next iteration
            Write<<<bpg,tpb>>>(buff->grad, buff->output, node->total);
        }

        for(int i = 0; i < s; i++)  // result = result ^(2^s)
        {
            bcmm<<<grid,block>>>(node->output, node->output, buff->output, batch, channels, a, a, a);
            Write<<<bpg,tpb>>>(buff->output, node->output, node->total);
        }
    
    };

    node->backward = [=]() mutable
    {
        /*
        Replacements
        X->output = gnode->output;
        node->output = gnode->grad;
        buff->output = gbuff->output;
        gbuff->grad  = gbuff->grad;
        */
                                        //A_00 & A_03 = X^T, A_01 = node->grad, A_02 = 0.0f;
        frechet_preparation<<<nbpg, tpb>>>(X->output, node->grad, gnode->output, batch, channels, a);
        SumSquared<<<nbpg,tpb>>>(scale, gnode->output, gnode->total);
        cudaMemcpy(&scalar, scale, sizeof(double), cudaMemcpyDeviceToHost);
        s = max(0.0, ceil(log2(sqrt(scalar))));
        val = .5f;
        thresh_loop = 2;
        while(val > threshold) {val /= (float)(thresh_loop); thresh_loop += 1;}
        scalar = 1.0 / (double)(1 << (int)s);  // FIX: was 2 << s

        identityKernel<<<nbpg,tpb>>>(gnode->grad, batch, channels, 2*a); // I
        Scale_Write<<<nbpg,tpb>>>(gnode->output, gbuff->output, gnode->total, scalar); // X / 2^s
        Accumulate<<<nbpg,tpb>>>(gbuff->output, gnode->grad,    gnode->total); // 1 + X / 2^s
        for(int i = 2; i <= thresh_loop; i++)
        {
            // buff->grad = (X/2^s) * buff->output = (X/2^s)^i
            bcmm<<<ggrid,block>>>(gnode->output, gbuff->output, gbuff->grad, batch, channels, 2*a, 2*a, 2*a, 0, 3, 3, 3, scalar);

            // result += (X/2^s)^i / i!
            Accumulate<<<nbpg,tpb>>>(gbuff->grad, gnode->grad, gnode->total, 1.0/std::tgamma(i+1));

            // buff = (X/2^s)^i  for next iteration
            Write<<<nbpg,tpb>>>(gbuff->grad, gbuff->output, gnode->total);
        }

        for(int i = 0; i < s; i++)  // result = result ^(2^s)
        {
            bcmm<<<ggrid,block>>>(gnode->grad,  gnode->grad, gbuff->output, batch, channels, 2*a, 2*a, 2*a);
            Write<<<nbpg,  tpb>>>(gbuff->output,gnode->grad, gnode->total);
        }
        
        collect_frechet<<<bpg,tpb>>>(gnode->grad, X->grad, batch, channels, a); // Accumulation of top right of gnode->grad to X->grad;
    };
    
    node->free = [=](){cudaFree(scale); node->clear(); buff->clear(); gnode->clear(); gbuff->clear();};
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
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
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
        mulKernel<<<bpg,tpb>>>(node->output, node->grad, X->grad, node->total,true);
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

    auto node = std::make_shared<NodeBackProp>("Logarithm", batch, channels, a, b, 1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
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
        GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
        const int tpb = THREADSPERBLOCK;
        const int bpg = (node->total+tpb-1)/tpb;

        node->forward = [=]()
        {
            //isNan(A); //isNan(B);
            ScaleAdd<<<bpg,tpb>>>(A->output, B->output, node->output, 1.0, node->total);
            //CheckError("Add forward");
            if(last) cudaMemcpy(&loss, node->output, sizeof(float), cudaMemcpyDeviceToHost);
        };

        node->backward = [=]()
        {
            if(last) fillKernel<<<bpg,tpb>>>(node->grad, 1.0f, node->total);
            Accumulate<<<bpg,tpb>>>(node->grad, A->grad, node->total);
            //CheckError("Add backward - A grad");

            Accumulate<<<bpg,tpb>>>(node->grad, B->grad, node->total);
            //CheckError("Add backward - B grad");
        };

        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};     
        return node;
    }

graph GraphOperations::Subtract(const graph& A, const graph& B, const bool last)
    {
        if(A->dim[0] != B->dim[0] || A->dim[1] != B->dim[1] || A->dim[2] != B->dim[2] || A->dim[3] != B->dim[3])
        {
            std::cout << "Dimension mismatch in subtract \n";
            Dimension(A);
            Dimension(B);
            std::exit(1);
        }

        const int batch = A->dim[0];
        const int channels = A->dim[1];
        const int a = A->dim[2];
        const int b = A->dim[3];

        auto node = std::make_shared<NodeBackProp>(" Subtract", batch, channels, a, b, 1);
        node->inputs = {A,B};
        GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
        const int tpb = THREADSPERBLOCK;
        const int bpg = (node->total+tpb-1)/tpb;

        node->forward = [=]()
        {
            //isNan(A); isNan(B);
            ScaleAdd<<<bpg,tpb>>>(A->output, B->output, node->output, -1.f, node->total);
            //CheckError("Subtract forward");
            if(last) cudaMemcpy(&loss, node->output, sizeof(float), cudaMemcpyDeviceToHost);
        };

        node->backward = [=]()
        {
            if(last) fillKernel<<<bpg,tpb>>>(node->grad, 1.0f, node->total);
            Accumulate<<<bpg,tpb>>>(node->grad, A->grad, node->total,  1.f);
            //CheckError("Subtract backward - A grad");

            Accumulate<<<bpg,tpb>>>(node->grad, B->grad, node->total, -1.f);
            //CheckError("Subtract backward - B grad");
        };

        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};

        
        return node;
    }

graph GraphOperations::Multiply(const graph& A, const graph& B)
    {
        if(A->dim[0] != B->dim[0] || A->dim[1] != B->dim[1] || A->dim[2] != B->dim[2] || A->dim[3] != B->dim[3])
        {
            std::cout << "Dimension mismatch in Multiply  \n";
            Dimension(A);
            Dimension(B);
            std::exit(1);
        }

        const int batch = A->dim[0];
        const int channels = A->dim[1];
        const int a = A->dim[2];
        const int b = A->dim[3];

        auto node = std::make_shared<NodeBackProp>("Multiply", batch, channels, a, b, 1);
        node->inputs = {A,B};
        GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
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
            mulKernel<<<bpg,tpb>>>(node->grad, B->output, A->grad, node->total, true);
            //CheckError("Add backward - A grad");

            mulKernel<<<bpg,tpb>>>(node->grad, A->output, B->grad, node->total, true);
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
        GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
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
        GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
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
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
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

graph GraphOperations::NthRow(const graph& X, const int row) // 0 based indexing
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d= X->dim[3];
    auto node = std::make_shared<NodeBackProp>(X->op_name + " Last", a,b,1,d, 1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    node->forward = [=](){nth_row_kernel<<<bpg,tpb>>>(X->output, node->output, row, a,b,c,d,false);};
    node->backward = [=](){nth_row_kernel<<<bpg,tpb>>>(node->grad, X->grad, row, a,b,c,d, true);};
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
    GB += (double)(prediction->total + target->total + 1) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (prediction->total+tpb-1) / tpb;
    
    node->forward = [=]()
    {   
        if(calculate_loss || !last)
        {
            //isNan(prediction); //isNan(target);
           
            WriteValueAt(node,0.0f,0);
            scalar_mse_kernel<<<bpg, tpb>>>(prediction->output, target->output, node->output, prediction->total);
            //isNan(node);

            if(last)
            {
                CheckError("Scalar MSE in MSE forward");
                cudaMemcpy(&loss, node->output, sizeof(float), cudaMemcpyDeviceToHost);
            } 
        }    
    };
    
    
    node->backward = [=]()
    {   
        deriv_mse_kernel<<<bpg,tpb>>>(prediction->output, target->output, node->grad, prediction->grad, target->total, last, false);
    };

    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd("Node", node->output, node->total);};
    return node;
}

graph GraphOperations::Entropy(const graph& X, const bool last)
{
    const int batch = X->dim[0];
    const int channels = X->dim[1];
    const int a = X->dim[2];
    const int b = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("Entropy " + X->op_name,batch,channels,a,b,1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (X->total+tpb-1)/tpb;

    node->forward = [=]()
    {
        e_kernel<<<bpg,tpb>>>(X->output,nullptr,node->output,X->total,last,false);
        //CheckError("Entropy forward");
    };

    node->backward = [=]()
    {
        e_kernel<<<bpg,tpb>>>(X->output, node->grad,node->output,X->total,last,true);
        //CheckError("Entropy Backward");
    };

    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
}

graph GraphOperations::MeanSquaredError(const graph& prediction, const float* target, const float* target_idx, const bool last)
{   
    if(prediction->dim[1] != 1 or prediction->dim[2] != 1)
    {
        std::cout << "Function defined to work for [B x 1 x 1 x W] \n";
        Dimension(prediction);
    }
    const int batch = prediction->dim[0];
    const int width = prediction->dim[3];
    auto node = std::make_shared<NodeBackProp>("MSE",1,1,1,1,1);
    node->inputs = {prediction};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    const int bpg = (batch + THREADSPERBLOCK - 1 )/ THREADSPERBLOCK;
    constexpr int tpb = THREADSPERBLOCK;
    
    node->forward = [=]()
    {   
        fillKernel<<<1,1>>>(node->output,0.0f, 1);
        idx_mse_kernel<<<bpg,tpb>>>(prediction->output, target, target_idx, node->output, batch,width);  
    };   
    
    node->backward = [=]()
    {   
        idx_mse_backward<<<bpg,tpb>>>(prediction->output, target, target_idx, nullptr, prediction->grad, batch,width,last,false);
    };
    
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);}; 
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
        GB += (double)(node->total + target->total) * sizeof(float) / (1ULL << 30);
        const int tpb = THREADSPERBLOCK;
        const int bpg = (prediction->total+tpb-1)/tpb;

        node->forward = [=]()
        {   
            if(calculate_loss || !last)
            {
            WriteValueAt(node,0.0f,0); //22544
            scalar_ce_kernel<<<(prediction->total+tpb-1)/tpb, tpb>>>(prediction->output, target->output, node->output, prediction->total);
            ScaleValue(node->output, (float)prediction->dim[3], 1);
            //isNan(node);
            //CheckError("Scalar CE in CE forward");
            if(last) cudaMemcpy(&loss, node->output, sizeof(float), cudaMemcpyDeviceToHost);
            else loss = 100.0f;
            }
           
        };

        node->backward = [=]()
        {   
            deriv_ce_kernel<<<bpg,tpb>>>(prediction->output,target->output,node->grad,prediction->grad,target->total,last,false);
            //isNan(prediction, 1);
            //CheckError("derivative of CE in CE backward");
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
        auto node = std::make_shared<NodeBackProp>("SoftMaxCrossEntropy Loss",1,1,1,1,1);
        node->inputs = {prediction, target};
        float* softmax_arr, *maxArr, *softmax;
        SafeCudaMalloc("Softmax array", softmax_arr, batch*channels*c);
        SafeCudaMalloc("Max array", maxArr, batch*channels*c);
        SafeCudaMalloc("SoftMax", softmax, node->total);
        GB += (double)(target->total + 1) * sizeof(float) / (1ULL << 30);
        const int tpb = THREADSPERBLOCK;
        const int bpg = (prediction->total+tpb-1) / tpb;
        
        node->forward = [=]()
        {   
            if(calculate_loss || !last)
            {
                //isNan(prediction); //isNan(target);
                SoftMax(prediction->output, softmax_arr, softmax, maxArr, batch,channels,c,d,0);
                WriteValueAt(node,0.0f,0);
                scalar_ce_kernel<<<bpg,tpb>>>(softmax,target->output,node->output,prediction->total);
                if(last) cudaMemcpy(&loss,node->output,sizeof(float),cudaMemcpyDeviceToHost);
                //CheckError("SoftMaxCrossEntropy")
            }
            
        };

        node->backward = [=]()
        {   
            ScaleAdd<<<bpg,tpb>>>(softmax, target->output, prediction->grad,-1.0f, target->total);
            if(!last) ScalePtr(prediction->grad, node->grad, prediction->total);
            ScaleValue(prediction->grad, (float)batch, prediction->total, 1);
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

graph GraphOperations::NCE(const graph& X, const int num_pos, const int num_neg) // []
{
    const int b = X->dim[0],  w = X->dim[3];

    if(X->dim[1] != 1 || X->dim[2] != 1)
    {
        std::cout << "Function only works on shape [B x 1 x 1 x K] \n ";
        Dimension(X);
    }

    if(1 + num_pos + num_neg != w)
    {
        printf("Layout issue in the last dimension... Expected 1 + num_pos: %i + num_neg: %i = %i | Last dim: %i",
        num_pos, num_neg, 1+num_pos+num_neg, w);
        std::exit(1);
    }
    
    auto node = std::make_shared<NodeBackProp>("NCE", b, 1, 1, num_pos, 1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpgf = (node->total+tpb-1)/tpb;
    const int bpgb = (b*w +tpb-1)/tpb;

    node->forward = [=]()
    {
        //isNan(X);
        nceKernel<<<bpgf,tpb>>>(X->output, node->output, num_pos, num_neg, b);
        //CheckError("NCE Forward");
    };

    node->backward = [=]()
    {
        nceDerivKernel<<<bpgb,tpb>>>(X->output, node->grad, X->grad, num_pos, num_neg, b);
        //CheckError("NCE Backward");
    };
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
    }

graph GraphOperations::CSInfoNCE(const graph& prediction, const int num_pos, const int num_neg, const float temperature, const bool last) 
{        
 /**
 * @brief Cosine Information Noise Contrastive Estimation:
 *
 * @details
 * The function Loss assumes the following layout of prediction
 * [B x (1 + P + N) x D], The first row is the positive anchor, The next P rows are the  positive anchors and the last
 * N rows are the negative anchors with information dimension D
 * 
 * Provide the following information to the Loss function
 * @param: Predicted [B x (1 + P + N) x D datapoints]. Assume K = (1 + P + N)
 * @param: The number of positive anchors
 * @param: The number of negative anchors
 * @param: Temperature scalar value
 * @param: Boolean to determine if This is the last layer of the network for node->grad initialization rather than accumulation;
 */
    // Use only the first row for the cosine similarity   
    auto z_norm  = RMSNorm(prediction, 0);                            // [B x K x W]
    auto z_first = NthRow(z_norm, 0);                                // [B x 1 x K]
    auto similarity = Scale(BMMABT(z_first, z_norm), temperature);   // [B X 1 X K]
    auto infonce = NCE(similarity, num_pos, num_neg);
    return BATCHMEAN(infonce);
}

graph GraphOperations::ContrastLearningTarget(const int batch) // Output matrix: [[1,1,-1],[1,1,-1],[-1,-1,1]]
{   
    auto node = std::make_shared<NodeBackProp>("Contrastive Learning Target", batch,1,3,3,1);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total + tpb-1) / tpb;
    cl_target<<<bpg,tpb>>>(node->output, batch, node->total); 
    CheckError("Initializing Target");
    node->forward = [=](){return;};
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
}

graph GraphOperations::BMM(const graph& A, const graph& B)     // m x n * n x p = m x p
{
    if(A->dim[3] != B->dim[2])
    {
        std::cout << "Dimension mismatch in BMM \n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }
    
    const bool batch_compatible = (A->dim[0] == B->dim[0]) || (A->dim[0] == 1) || (B->dim[0] == 1);
    const bool channel_compatible = (A->dim[1] == B->dim[1]) || (A->dim[1] == 1) || (B->dim[1] == 1);
    if (!(batch_compatible && channel_compatible))
    {
        std::cout << "Batch and Channel mismatch in BMM...\n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }
    const int A_case = (A->dim[0] > 1) + 2 * (A->dim[1] > 1);
    const int B_case = (B->dim[0] > 1) + 2 * (B->dim[1] > 1);
    const int C_case = (A_case > B_case) ? A_case : B_case; 


    const int batch = A->dim[0], channels = A->dim[1], m = A->dim[2], n = A->dim[3], p = B->dim[3];
    auto node = std::make_shared<NodeBackProp>("BMM", batch, channels, m, p, 1);
    node->inputs = {A,B};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((p+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
    dim3 grid_dA((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);   
    dim3 grid_dB((p+BLOCK_SIZE-1)/BLOCK_SIZE,(n+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
    node->forward = [=]()
    {   
        //isNan(A); //isNan(B);
        bcmm<<<grid, block>>>(A->output, B->output, node->output,   batch, channels, m, n, p, 0, A_case, B_case, C_case); //Assignment
        //CheckError("BMM... A * B in GraphOperations BMM forward");
    };

    node->backward = [=]()
    {
        bcmmABT<<<grid_dA, block>>>(node->grad, B->output, A->grad, batch, channels, m, p, n, 1, C_case, B_case, A_case); // ∂A = ∂Z * B^T
        //CheckError("BMM.. ∂A = ∂Z * B^T in GraphOperations BMM backward");

        bcmmATB<<<grid_dB, block>>>(A->output, node->grad, B->grad,batch, channels,  n, m, p, 1, A_case, C_case, B_case); // ∂B = A^T * ∂Z            
        //CheckError("MatMul... X^T*∂Z in GraphOperations BMM backward");
    };
        
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
}

graph GraphOperations::BMMABT(const graph& A, const graph& B)  // m x n * p x n = m x p
{
    if(A->dim[3] != B->dim[3])
    {
        std::cout << "Dimension mismatch in BMMABT \n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }

    const bool batch_compatible = (A->dim[0] == B->dim[0]) || (A->dim[0] == 1) || (B->dim[0] == 1);
    const bool channel_compatible = (A->dim[1] == B->dim[1]) || (A->dim[1] == 1) || (B->dim[1] == 1);
    if (!(batch_compatible && channel_compatible))
    {
        std::cout << "Batch and Channel mismatch in BMMABT...\n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }

    const int A_case = (A->dim[0] > 1) + 2 * (A->dim[1] > 1);
    const int B_case = (B->dim[0] > 1) + 2 * (B->dim[1] > 1);
    const int C_case = (A_case > B_case) ? A_case : B_case; 


    const int batch = A->dim[0], channels = A->dim[1], m = A->dim[2], n = A->dim[3], p = B->dim[2];
    auto node = std::make_shared<NodeBackProp>("BMMABT", batch, channels, m, p, 1);
    node->inputs = {A,B};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((p+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
    dim3 grid_dA((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);   
    dim3 grid_dB((p+BLOCK_SIZE-1)/BLOCK_SIZE,(n+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
    node->forward = [=]()
    {   
        //isNan(A); //isNan(B);
        bcmmABT<<<grid, block>>>(A->output, B->output, node->output, batch,channels,m,n,p,0, A_case, B_case, C_case); //Assignment
        //CheckError("BMMABT... A * B in GraphOperations BMMABT forward");
    };

    node->backward = [=]()
    {
        bcmm<<<grid_dA, block>>>(node->grad,B->output,A->grad,batch,channels,m,p,n,1,C_case,B_case,A_case); // ∂A = ∂Z * B^T
        //CheckError("BMMABT.. ∂A = ∂Z * B^T in GraphOperations BMMABT backward");

        bcmmT<<<grid_dB, block>>>(A->output,node->grad,B->grad,batch,channels,n,m,p,1,A_case,C_case,B_case); // ∂B = A^T * ∂Z            
        //CheckError("BMMABT... X^T*∂Z in GraphOperations BMMABT backward");
    };
        
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
}

graph GraphOperations::BMMATB(const graph& A, const graph& B)  // n x m * n x p = m x p
{
    if(A->dim[2] != B->dim[2])
    {
        std::cout << "Dimension mismatch in BMM \n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }
    const bool batch_compatible = (A->dim[0] == B->dim[0]) || (A->dim[0] == 1) || (B->dim[0] == 1);
    const bool channel_compatible = (A->dim[1] == B->dim[1]) || (A->dim[1] == 1) || (B->dim[1] == 1);
    if (!(batch_compatible && channel_compatible))
    {
        std::cout << "Batch and Channel mismatch in BMMATB...\n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }

    const int A_case = (A->dim[0] > 1) + 2 * (A->dim[1] > 1);
    const int B_case = (B->dim[0] > 1) + 2 * (B->dim[1] > 1);
    const int C_case = (A_case > B_case) ? A_case : B_case; 

    const int batch = A->dim[0], channels = A->dim[1], m = A->dim[3], n = A->dim[2], p = B->dim[3];
    auto node = std::make_shared<NodeBackProp>("BMMATB", batch, channels, m, p, 1);
    node->inputs = {A,B};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((p+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
    dim3 grid_dA((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);   
    dim3 grid_dB((p+BLOCK_SIZE-1)/BLOCK_SIZE,(n+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
    node->forward = [=]()
    {   
        //isNan(A); //isNan(B);
        bcmmATB<<<grid, block>>>(A->output, B->output, node->output,batch,channels,m,n,p,0, A_case, B_case, C_case); //Assignment
        //CheckError("BMMATB... A * B in GraphOperations BMMATB forward");
    };

    node->backward = [=]()
    {
        bcmmT<<<grid_dA, block>>>(node->grad, B->output, A->grad, batch, channels, m, p, n, 1, C_case, B_case, A_case); // ∂A = ∂Z * B^T
        //CheckError("BMMT.. ∂A = ∂Z * B^T in GraphOperations BMMATB backward");

        bcmm<<<grid_dB, block>>>(A->output, node->grad, B->grad,batch, channels,   n, m, p, 1, A_case, C_case, B_case); // ∂B = A^T * ∂Z            
        //CheckError("BMM... X^T*∂Z in GraphOperations BMMATB backward");
    };
        
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
}
 
graph GraphOperations::BMMT(const graph& A, const graph& B) // n x m * p x n = m x p
{
    if(A->dim[2] != B->dim[3])
    {
        std::cout << "Dimension mismatch in BMMT \n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }
    
    const bool batch_compatible = (A->dim[0] == B->dim[0]) || (A->dim[0] == 1) || (B->dim[0] == 1);
    const bool channel_compatible = (A->dim[1] == B->dim[1]) || (A->dim[1] == 1) || (B->dim[1] == 1);
    
    if (!(batch_compatible && channel_compatible))
    {
        std::cout << "Batch and Channel mismatch in BMMT...\n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }
    
    const int A_case = (A->dim[0] > 1) + 2 * (A->dim[1] > 1);
    const int B_case = (B->dim[0] > 1) + 2 * (B->dim[1] > 1);
    const int C_case = (A_case > B_case) ? A_case : B_case; 


    const int batch = A->dim[0], channels = A->dim[1], m = A->dim[3], n = A->dim[2], p = B->dim[2];
    auto node = std::make_shared<NodeBackProp>("BMMT", batch, channels, m, p, 1);
    
    node->inputs = {A,B};
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((p+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
    dim3 grid_dA((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);   
    dim3 grid_dB((p+BLOCK_SIZE-1)/BLOCK_SIZE,(n+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
    
    node->forward = [=]()
    {   
        //isNan(A); //isNan(B);
        bcmmT<<<grid, block>>>(A->output, B->output, node->output,batch,channels,m,n,p,0, A_case, B_case, C_case); //Assignment
        //CheckError("BMMT... A * B in GraphOperations BMMT forward");
    };

    node->backward = [=]()
    {
        bcmmATB<<<grid_dA, block>>>(node->grad, B->output, A->grad, batch, channels, m, p, n, 1, C_case, B_case, A_case); // ∂A = ∂Z * B^T
        //CheckError("BMMATB.. ∂A = ∂Z * B^T in GraphOperations BMMT backward");

        bcmmABT<<<grid_dB, block>>>(A->output, node->grad, B->grad, batch, channels, n, m, p, 1, A_case, C_case, B_case); // ∂B = A^T * ∂Z            
        //CheckError("BMMABT... X^T*∂Z in GraphOperations BMMT backward");
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
        const int arr_size = (type == 0) ? a*b*c : a*b*d;
        const int max_size = (type == 0) ? a*b*c : a*b*d;
        float *arr, *maxArr;
        GB += (double)(node->total + arr_size + max_size)* sizeof(float) / (1ULL << 30);
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
    const int arr_size = (type == 0) ? a*b*c : a*b*d;
    const int max_size = (type == 0) ? a*b*c : a*b*d;
    float* arr, *maxArr;
    SafeCudaMalloc("Softmask array", arr, arr_size);
    SafeCudaMalloc("Max array", maxArr, max_size);
    GB += (double)(node->total + arr_size + max_size) * sizeof(float) / (1ULL << 30);

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
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("Gathering "+X->op_name+" Actions",a,b,c,1,1);
    node->inputs = {X,actions};    
    GB += (double)node->total * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (a+tpb-1)/tpb;
    const int total_by_last = X->total / d;
    node->forward = [=]() {getactionKernel<<<bpg,tpb>>>(X->output, actions->output, node->output, d, total_by_last,false);};
    node->backward = [=](){getactionKernel<<<bpg,tpb>>>(node->grad, actions->output, X->grad,d, total_by_last,true);};
    node->free =  [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
};

graph GraphOperations::Scale(const graph& input, const float scale, const bool last)
{
    const int a = input->dim[0], b = input->dim[1], c = input->dim[2], d = input->dim[3];
    auto node = std::make_shared<NodeBackProp>("Scale", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;

    node->forward = [=]() 
    {   
        //isNan(input);
        Scale_Write<<<bpg,tpb>>>(input->output, node->output, node->total, scale);
        //CheckError("Scale Value in Scale forward");

    };

    node->backward = [=]() 
    {
        if(last) fillKernel<<<bpg,tpb>>>(node->grad, 1.0f, node->total);
        Accumulate<<<bpg,tpb>>>(node->grad, input->grad, node->total, scale);
        //CheckError("Deriv Scale in Scale");
        //isNan(input, 1);
    };

    node->free =  [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node; 
    }

graph GraphOperations::Constant_like(const graph& X, const float constant)
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("Noise", a,b,c,d,1);
    node->inputs = {X};    
    GB += (double)node->total * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    std::mt19937 rng{std::random_device{}()};
    node->forward = [=](){fillKernel<<<bpg,tpb>>>(node->output, constant, X->total);};
    node->backward = [=](){return;};
    node->free =  [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node; 
}



graph GraphOperations::GaussianNoise_like(const graph& X, const float mean, const float std)
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("Noise", a,b,c,d,1);
    node->inputs = {X};    
    GB += (double)node->total * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    std::mt19937 rng{std::random_device{}()};
    node->forward = [=]() mutable
    {   
        //isNan(X);
        GaussianNoise<<<bpg,tpb>>>(node->output,mean,std,node->total, (uint64_t)rng());
        //CheckError("Scale Value in Scale forward");
    };
    node->backward = [=](){return;};
    node->free =  [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node; 
}

graph GraphOperations::UniformNoise_like(const graph& X, const float min, const float max)
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("Noise", a,b,c,d,1);
    node->inputs = {X};    
    GB += (double)node->total * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    std::mt19937 rng{std::random_device{}()};
    node->forward = [=]() mutable
    {   
        //isNan(X);
        UnifNoise<<<bpg,tpb>>>(node->output, min, max,node->total, (uint64_t)rng());
        //CheckError("Scale Value in Scale forward");
    };
    node->backward = [=](){return;};
    node->free =  [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node; 
}

graph GraphOperations::RMSNorm(const graph& X, const int type) // type 0: row-wise, type 1: column-wise
{
        const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
        auto node = std::make_shared<NodeBackProp>("RMSNorm",a,b,c,d,1);
        node->inputs = {X};    
        const int arr_size = (type == 0) ? a*b*c : a*b*d;
        float *arr;
        GB += (double)(node->total + arr_size)* sizeof(float) / (1ULL << 30);
        SafeCudaMalloc("Softmax array", arr, arr_size);
        const int tpb = THREADSPERBLOCK;
        const int bpg = (node->total + tpb-1)/tpb;
        
        node->forward = [=]() 
        {   
            Write<<<bpg,tpb>>>(X->output, node->output, node->total);
            if(type == 1) SumSquaredSqrtRows<<<(arr_size + tpb-1)/tpb,tpb>>>(node->output,arr,a,b,c,d);
            else  SumSquaredSqrtCols<<<(arr_size + tpb-1)/tpb,tpb>>>(node->output,arr,a,b,c,d);
            Scale_arr<<<bpg,tpb>>>(node->output,arr,a,b,c,d,1,type);
        };

        node->backward = [=]()
        {
            Accumulate_rmsnorm_kernel<<<(node->total + tpb-1)/tpb, tpb>>>(node->grad, node->output, arr, X->grad,a, b, c, d, type);
            //CheckError("RMSNorm backward");
        };

        node->free =  [=]()
        {
            node->clear();
            cudaFree(arr);
        };
        
        node->zero_grad = [=](){Zerograd(node);};
        return node;

    }

graph GraphOperations::RELU(const graph& input)
{
    const int a = input->dim[0], b = input->dim[1], c = input->dim[2], d = input->dim[3];
    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("ReLU", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (1ULL << 30);

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
    GB += (double)node->total * sizeof(float) / (1ULL << 30);

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
    auto node = std::make_shared<NodeBackProp>("TANH", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (1ULL << 30);

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

graph GraphOperations::GELU(const graph& input)
{

    const int a = input->dim[0];
    const int b = input->dim[1];
    const int c = input->dim[2];
    const int d = input->dim[3];
    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("GELU", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (1ULL << 30);

    node->forward = [=]() 
    {   
        //isNan(input);
        GeLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output,node->output, node->total); // Assignment operation
        //CheckError("RELU in RELU forward");

    };

    node->backward = [=]() 
    {
        deriv_GeLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output, node->grad,input->grad,node->total);
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
    GB += (double)node->total * sizeof(float) / (1ULL << 30);

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
    GB += (double)node->total * sizeof(float) / (1ULL << 30);

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
            GB += 3 * (double)(node->total) * sizeof(float) / (1ULL << 30);
            SafeCudaMalloc("Temp of CopyCrop", temp, batch * depth * a * b);
            SafeCudaMalloc("TGrad of CopyCrop",tGrad, batch * depth * a * b);
        }
        
        else{GB += (double)(node->total) * sizeof(float) / (1ULL << 30);}

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
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
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
    GB += (double)(node->total) * sizeof(float) / (1ULL << 30);

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
    auto node = std::make_shared<NodeBackProp>("LayerMean", a,1,1,1,1);
    node->inputs = {X};    
    GB += (double)node->total * sizeof(float) / (1ULL << 30);

    node->forward = [=]() 
    {   
        //isNan(X);
        LayerMean<<<a,tpb>>>(X->output,node->output,a,b,c,d); // Assignment operation
        //CheckError("LayerMean forward");

    };

    node->backward = [=]() 
    {
        LayerMeanGrad<<<(node->total+tpb-1)/tpb,tpb>>>(node->grad,X->grad,a,b,c,d);
        //CheckError("LayerMean backward");
        //isNan(X,1);
    };

    node->free =  [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node; 
}

graph GraphOperations::BATCHMEAN(const graph& X)
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("BatchMean", 1,b,1,1,1);
    node->inputs = {X};    
    GB += (double)node->total * sizeof(float) / (1ULL << 30);

    node->forward = [=]() 
    {   
        //isNan(X);
        BatchMean<<<b,tpb>>>(X->output,node->output,a,b,c,d); // Assignment operation
        //CheckError("LayerMean forward");

    };

    node->backward = [=]() 
    {
        BatchMeanGrad<<<(X->total+tpb-1)/tpb,tpb>>>(node->grad, X->grad, a, b, c, d);
        //CheckError("LayerMean backward");
        //isNan(X,1);
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
    const int bpg = (node->total + tpb -1)/tpb;

    SafeCudaMalloc("LayerNorm ggamma", ggamma, node->total);
    SafeCudaMalloc("LayerNorm ggammanode", ggammanode, node->total);
    SafeCudaMalloc("LayerNorm mean", mean, a);
    SafeCudaMalloc("LayerNorm  std", std,  a);
    SafeCudaMalloc("LayerNorm ggamma_mean", ggamma_mean, a);
    SafeCudaMalloc("LayerNorm ggammanode_mean", ggammanode_mean,  a);
    GB += (double)(3 * node->total + 4 * a) * sizeof(float) / (1ULL << 30); 

    node->inputs = {X};

    node->forward = [=]()
    {
        //isNan(X);
        Write<<<bpg,tpb>>>(X->output, node->output, node->total);
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
        Scale_Write<<<bpg,tpb>>>(node->grad, ggamma, node->total, gamma);

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
    const int bpg = (node->total+tpb-1)/tpb;
    SafeCudaMalloc("BatchNorm ggamma", ggamma, node->total);
    SafeCudaMalloc("BatchNorm ggammanode", ggammanode, node->total);
    SafeCudaMalloc("BatchNorm mean", mean, b);
    SafeCudaMalloc("BatchNorm  std", std,  b);
    SafeCudaMalloc("BatchNorm ggamma_mean", ggamma_mean, b);
    SafeCudaMalloc("BatchNorm ggammanode_mean", ggammanode_mean,  b);
    GB += (double)(3 * node->total + 4 * b) * sizeof(float) / (1ULL << 30); 
    node->inputs = {X};

    node->forward = [=]()
    {
        //isNan(X);
        Write<<<bpg,tpb>>>(X->output, node->output, node->total);
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
        Scale_Write<<<bpg,tpb>>>(node->grad, ggamma, node->total, gamma);
        mulKernel<<<bpg,tpb>>>(ggamma, node->output, ggammanode, node->total);
        //CheckError("Multiply");

        BatchMean<<<b,tpb>>>(ggamma, ggamma_mean, a,b,c,d, false);
        BatchMean<<<b,tpb>>>(ggammanode, ggammanode_mean, a,b,c,d, false);
        //CheckError("BatchMean of ggammas");
        BatchBackward<<<bpg,tpb>>>(X->grad,node->output,node->grad,ggamma_mean,ggammanode_mean,std,gamma,epsilon,a,b,c,d);

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
        
    node->zero_grad = [=](){Zerograd(node);};

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
    const int bpg = (node->total+tpb-1)/tpb;

    SafeCudaMalloc("GroupNorm ggamma", ggamma, node->total);
    SafeCudaMalloc("GroupNorm ggammanode", ggammanode, node->total);
    SafeCudaMalloc("GroupNorm mean", mean, a*group);
    SafeCudaMalloc("GroupNorm  std", std,  a*group);
    SafeCudaMalloc("GroupNorm ggamma_mean", ggamma_mean, a*group);
    SafeCudaMalloc("GroupNorm ggammanode_mean", ggammanode_mean,  a*group);
    GB += (double)(3 * node->total + 4 * a*group) * sizeof(float) / (1ULL << 30); 
    
    node->inputs = {X};
        
    node->forward = [=]()
    {
        //isNan(X);
        const int bpg = (node->total+tpb-1)/tpb;
        Write<<<bpg,tpb>>>(X->output, node->output, node->total);
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
        Scale_Write<<<bpg,tpb>>>(node->grad, ggamma, node->total, gamma);
        //CheckError("Scale in GroupNorm Backward");

        mulKernel<<<bpg,tpb>>>(ggamma, node->output, ggammanode, node->total);
        //CheckError("Multiply");

        GroupMean<<<a*group,tpb>>>(ggamma, ggamma_mean, a,b,group,c,d,false);
        GroupMean<<<a*group,tpb>>>(ggammanode, ggammanode_mean,a,b,group,c,d,false);

        //CheckError("GroupMean of ggammas");
        GroupBackward<<<bpg, tpb>>>(X->grad,node->output,node->grad,ggamma_mean,ggammanode_mean,std,gamma,epsilon,a,b,group,c,d);
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
    const int bpg = (node->total + tpb-1)/tpb;

    SafeCudaMalloc("InstanceNorm ggamma", ggamma, node->total);
    SafeCudaMalloc("InstanceNorm ggammanode", ggammanode, node->total);
    SafeCudaMalloc("InstanceNorm mean", mean, a*b);
    SafeCudaMalloc("InstanceNorm  std", std,  a*b);
    SafeCudaMalloc("InstanceNorm ggamma_mean", ggamma_mean, a*b);
    SafeCudaMalloc("InstanceNorm ggammanode_mean", ggammanode_mean,  a*b);
    GB += (double)(3 * node->total + 4 * a*b) * sizeof(float) / (1ULL << 30); 
    
    node->inputs = {X};      
    node->forward = [=]()
    {
        //isNan(X);
        Write<<<bpg,tpb>>>(X->output, node->output, node->total);
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
        Scale_Write<<<bpg, tpb>>>(node->grad, ggamma, node->total, gamma);
        mulKernel<<<(node->total+tpb-1)/tpb,tpb>>>(ggamma, node->output, ggammanode, node->total);
        //CheckError("Multiply");

        InstanceMean<<<a*b,tpb>>>(ggamma, ggamma_mean, a,b,c,d, false);
        InstanceMean<<<a*b,tpb>>>(ggammanode, ggammanode_mean,a,b,c,d, false);

        //CheckError("Instance Means of ggammas");
        InstanceBackward<<<(node->total+tpb-1)/tpb,tpb>>>(X->grad,node->output,node->grad,ggamma_mean,ggammanode_mean,std,gamma,epsilon,a,b,c,d);

    };
        
    node->free = [=]()
    {
        node->clear(); cudaFree(mean); cudaFree(std); cudaFree(ggamma_mean); cudaFree(ggammanode_mean); cudaFree(ggamma); cudaFree(ggammanode);
    };

    node->zero_grad = [=](){Zerograd(node);};
        
    return node;

}
  
graph GraphOperations::track(const graph_tree& X)  
{
    str name = "Tracking: [";
    for(const auto &p : X) name += (p->op_name + " ");
    name += "]";
    auto node = std::make_shared<NodeBackProp>(name,1,1,1,1,0);
    node->inputs = X;      
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};   
    return node;

}
  
void GraphOperations::clipNorm(double* global_scale) {for(auto&node : nodes) if(node->clipnorm) node->clipnorm(global_scale);}
    
void GraphOperations::accumulate(double* global_scale) 
{
        for(auto&node : nodes) if(node->accumulate) node->accumulate(global_scale);
        Sqrt_Scale<<<1,1>>>(global_scale,1.0f,0);
}

void GraphOperations::ParameterUpdate(const graph&X, const bool show, const float lr, const bool adamw) 
{
    if(X) nodes = topological_sort(X);
    if (nodes.size() == 0)
    {
        printf("Warning... Nodes list is empty and parameter update cannot run.. \n You may have forgotten to topologically sort \n");
        return;
    }

    for(auto&node : nodes) if(node->updateParams)
    {
        node->updateParams(lr, adamw);
        if(show) printHeadGPU(node, 1);
    } 
}

void GraphOperations::forward(const graph&X, const bool calc_loss, const bool show, const bool time, const bool check_nan) 
{   
    if(X)nodes = topological_sort(X);
    if(calc_loss) calculate_loss = true;
    if (nodes.size() == 0)
    {
        printf("Warning... Nodes list is empty and forward cannot run.. \n You may have forgotten to topologically sort \n");
        return;
    }
    
    for (auto& node : nodes)
    {
        Timing timer(node->op_name + " forward");
        if(time){ CheckError("Before " + node->op_name + " forward"); timer.start();}
        if (node->forward) 
        {  
            node->forward();
            if(check_nan) {isNan(node); CheckError("IsNan of " + X->op_name);}
            if(show) {Dimension(node); printHeadGPU(node);}          
        }

        if(time){ CheckError("After " + node->op_name + " forward"); timer.end();}
    }
    calculate_loss = false;
}

void GraphOperations::backward(const graph&X,const bool show, const bool time, const bool check_nan) 
{
    if(X) nodes = topological_sort(X);
    if (nodes.size() == 0)
    {
        printf("Warning... Nodes list is empty and bacward cannot run.. \n You may have forgotten to topologically sort \n");
        return;
    }
    for(auto it=nodes.rbegin();it!=nodes.rend();++it)
    {   
        Timing timer((*it)->op_name + " backward");
        if(time){ CheckError("Previous"); timer.start();}
        if((*it)->backward) 
        {
            (*it)->backward(); 
            if(check_nan) {isNan((*it)); CheckError("IsNan of " + (*it)->op_name);}
            if(show) printHeadGPU((*it),1); 
        }
        if(time){ CheckError("After " + (*it)->op_name + " backward"); timer.end();}
    } 
}

void GraphOperations::zero_grad(const graph&X, const bool show) 
{
    if(X) nodes = topological_sort(X);
    if (nodes.size() == 0)
    {
        printf("Warning... Nodes list is empty and Zero grad cannot run.. \n You may have forgotten to topologically sort \n");
        return;
    }
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it){   
    if ((*it)->zero_grad)
    {
        //std::cout << "Zeroing gradient of" + (*it)->op_name + "\n";
        (*it)->zero_grad();
        //CheckError((*it)->op_name);
    }}
}

void GraphOperations::printNodes(const graph&X, const int show) 
{
    for (auto& node : nodes) {
    if (node->zero_grad) 
    {   
    std::cout << "Calling Node: " << node->op_name << "\n"; 
    Dimension(node);
    if (show == 1) printHeadGPU(node);
    if (show == 2) printHeadGPU(node,1);
    }}
}

void GraphOperations::printParams(const graph&X, const bool show) 
{
    if(X) nodes = topological_sort(X);
    for (auto& node : nodes) {
    if (node->printparams) node->printparams(show);}
}

void GraphOperations::clear_graph(const graph&X, const bool show)
{
    if(X) nodes = topological_sort(X);
    for (auto &node: nodes)
    {
        if(node->free)
        {   
            if(show) std::cout << "Freeing Node: " << node->op_name << "\n";
            node->free();
        }
    }
    nodes.clear();
}

void GraphOperations::clean_clear_graph(const graph&X, const bool show)
{
    if(X) nodes = topological_sort(X);
    for (auto &node: nodes)
    {
        if(node->free)
        {
            if(show) std::cout << "Freeing Node: " << node->op_name << "\n";
            node->free();
        }

        if(node->serious_free)
        { 
            if(show) std::cout << "Deleting Parameters for Node: " << node->op_name << "\n";
            node->serious_free();
        }
    }
}

double GraphOperations::GB = 0.0; bool GraphOperations::calculate_loss = false; float GraphOperations::loss = 0.0f;

Linear::Linear(const int input, const int output, const str name, const bool bias) : in(input), out(output), bias(bias),
W1(name + " W1",1,1,in,out), B1(name + " B1",1,1,1,out) 
{   
    if (name != "") op_name = name;   
}
void Linear::save(std::ofstream& f) const{W1.save(f); if(bias) B1.save(f);}
void Linear::load(std::ifstream& f){W1.load(f); if(bias) B1.load(f);}
void Linear::operator=(const Linear& other) {W1 = other.W1; if(bias) B1 = other.B1;}
graph Linear::forward(const graph & X)
{   
        if(X->dim[3] != W1.dim[2])
        {
            std::cout << "Shape Mismatch in Linear Layer of " << X->op_name <<": \n";
            std::cout << "Dimensions are input " << X->op_name << "and " << op_name << ":  (" << X->dim[2] << "," << X->dim[3] << ") and (" << W1.dim[2] << ","<<W1.dim[3] << ") \n";
            std::exit(1); 
        }

        const int batch = X->dim[0];
        const int channels = X->dim[1];
        const int m = X->dim[2];
        const int n = X->dim[3];
        const int p = out;
        const int tpb = THREADSPERBLOCK;
        auto node = std::make_shared<NodeBackProp>(op_name, batch, channels, m, p, 1);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((p+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
        dim3 grid_dA((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);   
        dim3 grid_dB((p+BLOCK_SIZE-1)/BLOCK_SIZE,(n+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);

        node->inputs = {X};

        node->forward = [=]()
        {   
            bcmm<<<grid, block>>>(X->output, W1.output, node->output,batch,channels, m, n, p, 0, 3, 0, 3);
            //CheckError("MatMul... X*W1 in Linear Layer forward");

            if(bias) 
            {
                BCumAdd<<<(tpb+batch*channels*m-1)/tpb, tpb>>>(node->output, B1.output,batch,channels, m, p);
            //CheckError("Add... X*W1+B1 in Linear Layer forward");
            }
            

        };

        node->backward= [=]()
        {
            bcmmABT<<<grid_dA, block>>>(node->grad, W1.output, X->grad, batch,channels, m, p, n,1,3,0,3);
            CheckError("MatMul... ∂Z*W^T in " + op_name + " backward");

            bcmmATB<<<grid_dB, block>>>(X->output, node->grad, W1.grad, batch,channels, n, m, p,1,3,3,0);
            CheckError("MatMul... X^T*∂Z in " + op_name + " backward");

            if(bias)
            {
                BCompress<<<out, tpb>>>(node->grad, B1.grad, batch, channels, m, p);
                //CheckError("Compress... Squeeze(∂Z)->∂b in Lineary Layer backward");
            }
        };
        
        node->free = [=](){node->clear();};

        node->serious_free = [=]()
        {
            W1.clear();
            B1.clear();
        };

        node->zero_grad = [=]()
        {
            Zerograd(node);
            Zerograd(W1);
            if(bias) Zerograd(B1);
        };
        
        node->accumulate = [=](double* global_scale)
        {
            W1.accumulate_grad(global_scale);
            if(bias) B1.accumulate_grad(global_scale);
        };

        node->clipnorm = [=](const double* global_scale)
        {
            W1.gradnorm(global_scale);
            if(bias) B1.gradnorm(global_scale);
        };

        node->updateParams = [=](const float lr, const bool adamw)
        {
            W1.update(lr, adamw);
            if(bias) B1.update(lr, adamw);
        };

        node->printparams = [=](const bool full)
        {
        if(!full){printHeadGPU(W1);  if (bias) { printHeadGPU(B1); };}
        else {printGPU(W1); if (bias) { printGPU(B1); };};
        };

        return node;

}

class LeftLinear  // W * x + b;
{ 
private: 
    int in, out;
    const bool bias;
public:
    AdamParameter W1;
    AdamParameter B1;
    str op_name = "Left Linear Layer";
    LeftLinear(const int out_, const int in_, const int X3, const str name, const bool bias = true) : in(in_), out(out_), bias(bias),
    W1(name + " W1",1,1, out_, in_), B1(name + " B1",1,1,1, X3){}
    graph forward(const graph & X)
    {   

        if(W1.dim[3] != X->dim[2])
        {
            std::cout << "Shape Mismatch in Linear Layer of " << X->op_name <<": \n";
            Dimension(W1);  Dimension(X);
            std::exit(1); 
        }

        if(bias && B1.dim[3] != X->dim[3])
        {
            std::cout << "Shape Mismatch in Linear Layer of " << X->op_name <<": \n";
            Dimension(X); Dimension(B1); std::exit(1); 
        }
        
        const int batch = X->dim[0];
        const int channels = X->dim[1];
        const int m = W1.dim[2];
        const int n = W1.dim[3];
        const int p = X->dim[3];
        const int tpb = THREADSPERBLOCK;
        auto node = std::make_shared<NodeBackProp>(op_name, batch, channels, m, p, 1);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid   ((p+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
        dim3 grid_dA((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);   
        dim3 grid_dB((p+BLOCK_SIZE-1)/BLOCK_SIZE,(n+BLOCK_SIZE-1)/BLOCK_SIZE,batch*channels);
        node->inputs = {X};
        node->forward = [=]()
        {   
            isNan(X);
            bcmm<<<grid, block>>>(W1.output, X->output, node->output,batch,channels, m, n, p, 0, 0, 3, 3);
            //CheckError("MatMul... X*W1 in Linear Layer forward");

            if(bias) 
            {
                BCumAdd<<<(tpb+batch*channels*m-1)/tpb, tpb>>>(node->output, B1.output,batch,channels, m, p);
                //CheckError("Add... W1 * X +B1 in Linear Layer forward");
            }
            
            isNan(node);
        };

        node->backward= [=]()
        {
            bcmmABT<<<grid_dA, block>>>(node->grad, X->output, W1.grad, batch,channels, m, p, n,1,3,3,0);
            CheckError("MatMul... ∂Z*X^T in " + op_name + " backward");

            bcmmATB<<<grid_dB, block>>>(W1.output, node->grad, X->grad, batch,channels, n, m, p,1,0,3,3);
            CheckError("MatMul... W^T*∂Z in " + op_name + " backward");

            if(bias)
            {
                BCompress<<<out, tpb>>>(node->grad, B1.grad, batch, channels, m, p);
                //CheckError("Compress... Squeeze(∂Z)->∂b in Lineary Layer backward");
            }
        };
        
        node->free = [=](){node->clear();};

        node->serious_free = [=]()
        {
            W1.clear();
            B1.clear();
        };

        node->zero_grad = [=]()
        {
            Zerograd(node);
            Zerograd(W1);
            if(bias) Zerograd(B1);
        };
        
        node->accumulate = [=](double* global_scale)
        {
            W1.accumulate_grad(global_scale);
            if(bias) B1.accumulate_grad(global_scale);
        };

        node->clipnorm = [=](const double* global_scale)
        {
            W1.gradnorm(global_scale);
            if(bias) B1.gradnorm(global_scale);
        };

        node->updateParams = [=](const float lr, const bool adamw)
        {
            W1.update(lr, adamw);
            if(bias) B1.update(lr, adamw);
        };

        node->printparams = [=](const bool full)
        {
        if(!full){printHeadGPU(W1);  if (bias) { printHeadGPU(B1); };}
        else {printGPU(W1); if (bias) { printGPU(B1); };};
        };

        return node;

    }
    void save(std::ofstream& f) const{W1.save(f); if(bias) B1.save(f);}
    void load(std::ifstream& f){W1.load(f); if(bias) B1.load(f);}
};

Convolute2D::Convolute2D(int Input, int Output, int C, int D, int stride, int padding, str param) : out(Output), inp(Input), c(C), d(D), pad(padding), 
    stride(stride), name(param),
    W1(name + " Weight ", out, inp, c, d),
    B1(name + " Bias ",     1, out, 1, 1)
{   

}
void Convolute2D::save(std::ofstream& f) const{W1.save(f); B1.save(f);}
void Convolute2D::load(std::ifstream& f){W1.load(f); B1.load(f);}
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
    GraphOperations::GB += (double)(node->total) * sizeof(float) / (1ULL << 30);
    const int tpb = THREADSPERBLOCK;

    dim3 block_fwd(8, 8, 16);
    dim3 block_wgt(16,16,4);

    dim3 grid_forward((outC + block_fwd.x - 1) / block_fwd.x, (outR + block_fwd.y - 1) / block_fwd.y, (batch * out + block_fwd.z - 1) / block_fwd.z);
    dim3 grid_weight_grad((out + block_wgt.x - 1) / block_wgt.x, (inp + block_wgt.y - 1) / block_wgt.y, (c * d + block_wgt.z - 1) / block_wgt.z);
    dim3 grid_input_grad((b + block_fwd.x - 1) / block_fwd.x, (a + block_fwd.y - 1) / block_fwd.y,(batch * inp + block_fwd.z - 1) / block_fwd.z);    
    node->forward = [=]()
    {
        //isNan(X);
        CV2D<<<grid_forward, block_fwd>>>(X->output,W1.output,B1.output,node->output,batch,out,inp,a,b,c,d,pad,stride);
        //CheckError("Forward Convolution in " + name);
    };
    
    node->backward = [=]()
    {
        //isNan(node, 1);
        GV2D<<<grid_weight_grad, block_wgt>>>(X->output,node->grad,W1.grad,batch,out,inp,a,b,c,d,pad,stride);
        Channel_Squeeze1D<<<out,tpb>>>(node->grad,B1.grad,batch,out,outR,outC);
        CV2D_GradInput<<<grid_input_grad, block_fwd>>>(node->grad,W1.output,X->grad,batch,out,inp,a,b,c,d,outR, outC,pad, stride);
        //CheckError("Weight + Bias + Input Gradient in " + name);

    };

    node->zero_grad = [=]()
    {
        Zerograd(node);
        Zerograd(W1);
        Zerograd(B1);
    };
    
    node->free = [=](){node->clear();};
    
    node->serious_free = [=]()
    {
        W1.clear();
        B1.clear();
    };

    node->accumulate = [=](double* global_scale)
    {
        W1.accumulate_grad(global_scale);
        B1.accumulate_grad(global_scale);
    };
    
    node->clipnorm = [=](const double* global_scale)
    {
        W1.gradnorm(global_scale);
        B1.gradnorm(global_scale);
    };
    
    node->updateParams = [=](const float lr, const bool adamw)
    {
        W1.update(lr, adamw);
        B1.update(lr, adamw);
    };
    
    node->printparams = [=](const bool full)
    {
        if(!full){printHeadGPU(W1); printHeadGPU(B1);}
        else {printGPU(W1); printGPU(B1);};
        
        
    };

    return node;
}

Convolute2DT::Convolute2DT(int Input, int Output, int C, int D, int stride, int padding, str param) 
    :out(Output), inp(Input), c(C), d(D), pad(padding), stride(stride), name(param),
    W1(name + " Weight ",  out, inp, c, d), B1(name + " Bias ", 1, out, 1, 1)

{   

}
void Convolute2DT::save(std::ofstream& f) const{W1.save(f); B1.save(f);}
void Convolute2DT::load(std::ifstream& f) {W1.load(f); B1.load(f);}
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

    GraphOperations::GB += (double)(node->total + W1.total + B1.total) * sizeof(float) / (1ULL << 30);
    
    const int tpb = THREADSPERBLOCK;
    dim3 block(8, 8, 16);
    dim3 grid_fwd((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y,(batch * out + block.z - 1) / block.z);
    dim3 grid_igrad((inp_w + block.x - 1)/block.x,(inp_h + block.y - 1) / block.y, (batch * inp + block.z - 1) / block.z);

    node->forward = [=]()
    {

        //isNan(X);
        CVT2D<<<grid_fwd, block>>>(X->output, W1.output, B1.output, node->output, batch, out, inp, inp_h, inp_w, c, d, pad, stride);
        //CheckError("ConvTranspose2D Forward Kernel");
        //isNan(node);
    };

    node->backward = [=]()
    {
        isNan(node, 1);

        GVT2D<<<out,tpb>>>(X->output, node->grad, W1.grad, batch, out, inp, inp_h, inp_w, c, d, pad, stride);
        Channel_Squeeze1D<<<out,tpb>>>(node->grad, B1.grad, batch, out, out_h, out_w);
        //CheckError("Gradient w.r.t W and B1 for ConvTranspose2D");

        CVT2D_GradInput<<<grid_igrad, block>>>(node->grad, W1.output, X->grad, batch, out, inp, inp_h, inp_w, c, d, pad, stride);
        //CheckError("Gradient w.r.t input for ConvTranspose2D");
    };

    node->zero_grad = [=]()
    {
        Zerograd(node);
        Zerograd(W1); 
        Zerograd(B1);
    };

    node->free = [=](){node->clear();};

    node->serious_free = [=]() {W1.clear();B1.clear();};

    node->accumulate = [=](double* global_scale)
    {   W1.accumulate_grad(global_scale);
        B1.accumulate_grad(global_scale);
    };

    node->clipnorm = [=](const double* global_scale)
    {
        W1.gradnorm(global_scale);
        B1.gradnorm(global_scale);
    };

    node->updateParams = [=](const float lr, const bool adamw)
    {
        W1.update(lr, adamw);
        B1.update(lr, adamw);
    };

    node->printparams = [=](const bool full)
    {
        if(full){printGPU(W1); printGPU(B1);}
        else {printHeadGPU(W1); printHeadGPU(B1);}

    };

    return node;
}

TimeMLPBlock::TimeMLPBlock(const int t_embed_dim, const int t_hidden) : L0(t_embed_dim, t_hidden, "Time MLP L0"), L1(t_hidden, t_hidden, "Time MLP L1") {}

graph TimeMLPBlock::forward(const graph & X)
{
    auto first = L0.forward(X);
    auto activate = GraphOperations::SILU(first); 
    auto node = L1.forward(activate);
    return node;
}
void TimeMLPBlock::save(std::ofstream& f) const{L0.save(f);L1.save(f);}
void TimeMLPBlock::load(std::ifstream& f)      {L0.load(f); L1.load(f);}

Multi_Linear_Residual_Block::Multi_Linear_Residual_Block(const int input, const int output, const int num_residuals, const int layers, const int hidden_size): 
input_dim(input), output_dim(output), residuals(num_residuals), hidden_dim(hidden_size), layers(layers)
    {
        sequence.push_back(new Linear(input, hidden_size));
        for (int i = 1; i < num_residuals * layers; ++i)sequence.push_back(new Linear(hidden_size, hidden_size));
        sequence.push_back(new Linear(hidden_size, output));
    }
void Multi_Linear_Residual_Block::save(std::ofstream& f) const{for(const auto &p : sequence) p->save(f);}
void Multi_Linear_Residual_Block::load(std::ifstream& f){for(auto&p : sequence) p->load(f);}

void Noise(const graph & input, const float mean, const float std, const str type)
{
    std::random_device rd;
    const uint64_t seed =  ((uint64_t)rd() << 32) | rd();
    const unsigned int tpb = THREADSPERBLOCK;
    const unsigned int bpg = (input->total + tpb-1)/tpb;
    if(type == "gaussian") GaussianNoise<<<bpg,tpb>>>(input->output, mean, std, input->total, seed);
    if(type == "uniform")      UnifNoise<<<bpg,tpb>>>(input->output, mean, std, input->total, seed);
}

