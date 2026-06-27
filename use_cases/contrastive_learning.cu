#include "diffusion.h"

struct MNISTLoader
{
    int   batch_size;
    graph data;  
    std::vector<float> Label = {0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f};
    std::unordered_map<int, std::vector<int>> label_map;

    int*  d_row_idx  = nullptr; 
    int*  d_cl_rows  = nullptr;
    int   n_rows     = 0;
    int   cl_slots   = 0; 
    MNISTLoader(const int MAX_BATCH, const str& filepath) : batch_size(MAX_BATCH)
    {
        data = ReadCSV(filepath);
        data->reshape({(int)data->total / data->dim[3], 1, 1, data->dim[3]});
        Dimension(data);
        const int stride   = (data->total) / data->dim[0];
        n_rows             = data->dim[0];
        for (int i = 0; i < n_rows; i++)
        {
            int lbl = static_cast<int>(ReadValueAt(data, i * stride));
            label_map[lbl].push_back(i);
            CheckError("Label Creation at idx " + std::to_string(i));
        }
        CheckError("Label Creation");
        SafeCudaMalloc("D_row_idx", d_row_idx, MAX_BATCH);
    }

    ~MNISTLoader(){
        if (d_row_idx) cudaFree(d_row_idx);
        if (d_cl_rows) cudaFree(d_cl_rows);
        data->clear();
    }

    void destroy()
    {   
        Label.clear();
        label_map.clear();
        data->clear();
        if (d_row_idx) { cudaFree(d_row_idx); d_row_idx = nullptr; }
        if (d_cl_rows) { cudaFree(d_cl_rows); d_cl_rows = nullptr; }
    }

    void ensure_cl_rows(int needed)
    {
        if (needed > cl_slots)
        {
            if (d_cl_rows) cudaFree(d_cl_rows);
            cudaMalloc(&d_cl_rows, needed * sizeof(int));
            cl_slots = needed;
        }
    }

    graph ReadCSV(const str& filepath){
        std::ifstream file(filepath);
        if (!file.is_open()) { std::cerr << "Could not open file\n"; std::exit(1); }

        std::string line;
        std::vector<float> row;
        int row_s = 0, col_s = 0;

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string cell;
            int sub_col = 0;
            bool valid_row = true;
            std::vector<float> current_row_data;

            while (std::getline(ss, cell, ','))
            {
                try   { ++sub_col; current_row_data.push_back(std::stof(cell)); }
                catch (const std::invalid_argument&) { valid_row = false; break; }
            }
            if (valid_row && sub_col > 0)
            {
                row.insert(row.end(), current_row_data.begin(), current_row_data.end());
                col_s = sub_col;
                ++row_s;
            }
        }

        auto node = std::make_shared<NodeBackProp>("CSV", 1, 1, row_s, col_s, 1);
        cudaMemcpy(node->output, row.data(), node->total * sizeof(float), cudaMemcpyHostToDevice);
        file.close();
        return node;
    }

    graph contrastive_load(const int batch, const int num_pos, const int num_neg) // outputs a [batch x (1 + num_pos + num_neg) x 28 x 28] tensor where each anchor image is followed by its num_pos positive examples and num_neg negative examples
    {
        const int total_per  = 1 + num_pos + num_neg;      
        const int total_imgs = total_per * batch;

        ensure_cl_rows(total_imgs);

        auto node = std::make_shared<NodeBackProp>("CL Training MNIST", total_imgs, 1, 28, 28, 1);
        node->inputs = {};

        const int stride   = data->dim[3];
        const int img_size = stride - 1;

        node->forward = [=]()
        {
            static std::mt19937 rng{std::random_device{}()};

            std::vector<int> h_rows(total_imgs);
            int write = 0;

            for (int t = 0; t < batch; t++)
            {
                int anc_lbl = static_cast<int>(rng() % Label.size());
                const auto& anc_pool = label_map.at(anc_lbl);
                int ai = rng() % anc_pool.size();
                h_rows[write++] = anc_pool[ai];

                for (int p = 0; p < num_pos; p++)
                {
                    int pi = rng() % anc_pool.size();
                    if (anc_pool.size() > 1) while (pi == ai) pi = rng() % anc_pool.size();
                    h_rows[write++] = anc_pool[pi];
                }

                for (int n = 0; n < num_neg; n++)
                {
                    int neg_lbl = static_cast<int>(rng() % Label.size());
                    while (neg_lbl == anc_lbl) neg_lbl = static_cast<int>(rng() % Label.size());
                    const auto& neg_pool = label_map.at(neg_lbl);
                    int ni = rng() % neg_pool.size();
                    h_rows[write++] = neg_pool[ni];
                }
            }

            cudaMemcpy(d_cl_rows, h_rows.data(), h_rows.size() * sizeof(int), cudaMemcpyHostToDevice);
            dim3 grid(total_imgs, (img_size + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
            cl_scatter_kernel<<<grid, THREADSPERBLOCK>>>(data->output, node->output, d_cl_rows, stride, img_size);
            CheckError("CL scatter kernel");
        };

        node->free      = [=](){ node->clear(); };
        node->zero_grad = [=](){ Zerograd(node); };
        return node;
    }

    std::pair<graph, graph> load(const int batch_) // outputs a [batch x 1 x 28 x 28] tensor and a [batch x 1 x 1 x 10] one-hot labels tensor
    {
        auto node   = std::make_shared<NodeBackProp>("MNIST Data",   batch_,     1, 28, 28, 1);
        auto target = std::make_shared<NodeBackProp>("MNIST Labels", batch_,     1,  1, 10, 1);
        node->inputs   = {};
        target->inputs = {};

        const int stride   = data->dim[3];
        const int img_size = stride - 1;
        const int TPB = THREADSPERBLOCK;
        const int BPG = (target->total + TPB-1)/TPB;

        dim3 grid_A(batch_, (img_size + TPB - 1) / TPB);
        int grid_B = (batch_ + TPB - 1) / TPB;
        std::vector<int> h_idx(n_rows);
        std::iota(h_idx.begin(), h_idx.end(), 0);
        std::mt19937 rng{std::random_device{}()};
        std::shuffle(h_idx.begin(), h_idx.end(), rng);
        std::uniform_int_distribution dist(0, n_rows - batch_ - 1);

        node->forward = [=]() mutable
        {   
            cudaMemcpy(d_row_idx, h_idx.data() + dist(rng), batch_ * sizeof(int), cudaMemcpyHostToDevice);
            scatter_images_kernel<<<grid_A, TPB>>>(data->output, node->output,d_row_idx, stride, img_size);
            CheckError("scatter_images_kernel");
            fillKernel<<<BPG,TPB>>>(target->output, 0.f, target->total);
            onehot_kernel<<<grid_B, TPB>>>(target->output, d_row_idx, data->output, batch_, stride);
            CheckError("onehot_kernel");
        };

        node->free = [=]()
        {
            node->clear();
            target->clear();
        };
        
        node->zero_grad = [=](){ Zerograd(node); Zerograd(target); };
        return {node, target};
    }

    std::pair<graph, graph> load_different(const int batch, const int num_classes) // outputs a [batch x num_classes x 28 x 28] tensor and a [batch x num_classes x 10] one-hot labels tensor, where each image in the batch belongs to a different class
    {
    auto node   = std::make_shared<NodeBackProp>("MNIST Data",   batch, num_classes, 28, 28, 1);
    auto target = std::make_shared<NodeBackProp>("MNIST Labels", batch, num_classes,  1, 10, 1);
    node->inputs   = {};
    target->inputs = {};

    const int stride   = data->dim[3];
    const int img_size = stride - 1;
    const int TPB      = THREADSPERBLOCK;
    const int total    = batch * num_classes;
    dim3 grid_A(total, (img_size + TPB - 1) / TPB);
    int  grid_B        = (total + TPB - 1) / TPB;

    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> dist(0, n_rows - 1);

    ensure_cl_rows(total);

    std::vector<int> h_idx(total);

    node->forward = [=]() mutable
    {
        for (int i = 0; i < batch; i++)
        {
            std::unordered_set<int> seen_labels;
            seen_labels.reserve(num_classes);
            for (int j = 0; j < num_classes; j++)
            {
                int idx, lbl;
                do {
                    idx = dist(rng);
                    lbl = static_cast<int>(ReadValueAt(data, idx * stride));
                } while (seen_labels.count(lbl) > 0);

                seen_labels.insert(lbl);
                h_idx[i * num_classes + j] = idx;
            }
        }

        cudaMemcpy(d_cl_rows, h_idx.data(), total * sizeof(int), cudaMemcpyHostToDevice);
        scatter_images_kernel<<<grid_A, TPB>>>(data->output, node->output, d_cl_rows, stride, img_size);
        fillKernel<<<grid_B, TPB>>>(target->output, 0.f, batch*num_classes);
        onehot_kernel<<<grid_B, TPB>>>(target->output, d_cl_rows, data->output, batch, stride);
        CheckError("Load Different Kernel");
    };

    node->free      = [=](){ node->clear(); target->clear(); };
    node->zero_grad = [=](){ Zerograd(node); Zerograd(target); };
    return {node, target};
}
};

class SwiGLUFFN
{
private:
    Linear gate_proj, up_proj, down_proj;
public:
    SwiGLUFFN(const int dim):
    gate_proj(dim, dim * 2, "SwiGLU Gate Projection", false), 
    up_proj(dim, dim * 2,   "SwiGLU Up   Projection", false), 
    down_proj(dim * 2, dim, "SwiGLU Down Projection", false){}

    graph forward(const graph& x)
    {      
        auto gate = GraphOperations::SILU(gate_proj.forward(x));
        auto up   = up_proj.forward(x);
        return down_proj.forward(gate * up);
    }

    void save(std::ofstream& f) const
    {
        gate_proj.save(f);
        up_proj.save(f);
        down_proj.save(f);
    }
    void load(std::ifstream& f)
    {
        gate_proj.load(f);
        up_proj.load(f);
        down_proj.load(f);
    }
};

class CGMLayer
{
private:
    SwiGLUFFN ffn;
public:
    CGMLayer(const int dim) : ffn(dim) {}
    graph forward(const graph& x)
    {
        x->reshape({x->dim[0], x->dim[1], 1, x->dim[2]*x->dim[3]}); 
        auto normed = GraphOperations::RMSNorm(x, 0);
        auto ffn_out = ffn.forward(normed);
        return x + ffn_out;
    }

    void save(std::ofstream& f) const{ffn.save(f);}
    void load(std::ifstream& f){ffn.load(f);}

};

class Encoder
{
private:
    std::vector<CGMLayer*> layers;
public:
    Encoder(const int dim = 784, const int num_layers = 4)
    {
        layers = std::vector<CGMLayer*>(num_layers);
        for(auto& layer : layers) layer = new CGMLayer(dim);
    }

    graph forward(const graph& x)
    {
        auto out = x;
        for (auto& layer : layers) out = layer->forward(out);
        out = GraphOperations::RMSNorm(out, 0);
        return out;
    }

    void save(std::ofstream& f) const
    {
        for (auto& layer : layers) layer->save(f);
    }

    void load(std::ifstream& f)
    {
        for (auto& layer : layers) layer->load(f);
    }

};

class Decoder
{
private:
    std::vector<CGMLayer*> layers;
    Linear out_proj;
public:
    Decoder(const int dim = 784, const int num_layers = 4) : out_proj(dim, dim, "Decoder Output Projection", true)
    {
        layers = std::vector<CGMLayer*>(num_layers);
        for(auto& layer : layers) layer = new CGMLayer(dim);
    }

    graph forward(const graph& x)
    {
        auto out = x;
        for (auto& layer : layers) out = layer->forward(out);
        out = GraphOperations::RMSNorm(out, 0);
        out = out_proj.forward(out);
        return GraphOperations::SIGMOID(out);
    }

    void save(std::ofstream& f) const
    {
        for (auto& layer : layers) layer->save(f);
        out_proj.save(f);
    }
    void load(std::ifstream& f) 
    {
        for (auto& layer : layers) layer->load(f);
        out_proj.load(f);
    }

};

class ContrastiveLoss
{
public:
    float temperature;
    ContrastiveLoss(const float temp = 1.0f) : temperature(temp) {}
    graph forward(const graph& z)
    {
        z->reshape({1, 1, z->dim[0], z->dim[1]*z->dim[2]*z->dim[3]}); 
        auto sim  = GraphOperations::BMMABT(z, z);
        auto scaled_sim = GraphOperations::Scale(sim, 1.0f / temperature);
        auto target = GraphOperations::Subtract(GraphOperations::Scale(GraphOperations::identity_like(sim), 2.f), GraphOperations::ones_like(sim)); 
        auto loss = GraphOperations::MeanSquaredError(scaled_sim, target, false);
        return loss;
    }

};

class ConstructiveGenerativeModel
{
private:
    Encoder enc;
    Decoder dec;
public:
    ConstructiveGenerativeModel(const int dim = 784) : enc(dim), dec(dim) {}
    std::pair<graph, graph> forward(const graph& x)
    {
        auto z = enc.forward(x); z->op_name = "Latent Representation";
        auto recon = dec.forward(z); recon->op_name = "Reconstruction";
        return {z, recon};
    }
    graph generate(const graph& input)
    {
        auto sample = GraphOperations::RMSNorm(GraphOperations::GaussianNoise_like(input, 0.f, 1.f),0);
        auto out = dec.forward(sample);
        return out;
    }
    void save(const std::string& filename) const
    {        
        std::ofstream f(filename, std::ios::binary);
        enc.save(f);
        dec.save(f);
        f.close();
        std::cout << "Model save successfully to " << filename << "\n";
    }
    void load(const std::string& filename)
    {
        std::ifstream f(filename, std::ios::binary);
        enc.load(f);
        dec.load(f);
        f.close();
        std::cout << "Model loaded successfully from " << filename << "\n";
    }

};

class GPM
{
    SwiGLUFFN ffn;
public:
    
    GPM(const int dim) : ffn(dim) {}
    template<typename ActFn, typename NormFN>
    graph forward(const graph& X, ActFn activation, NormFN norm)
    {
        auto h = ffn.forward(X);
        auto act = activation(h);
        return norm(X + act);
    }
    void save(std::ofstream& f) const{ffn.save(f);}
    void load(std::ifstream& f){ffn.load(f);}
};

class Denoiser
{
    std::vector<GPM*> layers;
    Linear out_proj;
public:
    Denoiser(const int dim, const int num_layers = 4) :
    out_proj(dim, dim)
    {
        layers = std::vector<GPM*>(num_layers);
        for(auto& l : layers) l = new GPM(dim); 
    }
    template<typename ActFn, typename NormFn>
    graph forward(const graph& X, ActFn activation, NormFn normalization)
    {
        auto inp = X;
        for(const auto& l: layers) inp = l->forward(inp, activation, normalization);
        return out_proj.forward(inp);
    }

    void save(const std::string& filename) const
    {        
        std::ofstream f(filename, std::ios::binary);
        for (auto& layer : layers) layer->save(f);
        out_proj.save(f);

        f.close();
        std::cout << "Model save successfully to " << filename << "\n";
    }
    void load(const std::string& filename)
    {
        std::ifstream f(filename, std::ios::binary);
        for (auto& layer : layers) layer->load(f);
        out_proj.load(f);
        f.close();
        std::cout << "Model loaded successfully from " << filename << "\n";
    }

};

int main()
{
    GraphOperations go;
    MNISTLoader load_mnist(2048, "../mnist/mnist_train.csv");
    ConstructiveGenerativeModel model;
    model.load("../best_models/cgm.bin");
    ContrastiveLoss contrast_loss(0.07f);
    const bool train = !true;
    if(train)
    {
        auto [input, labels] = load_mnist.load(2048);
        auto [z, recon] = model.forward(input);
        auto target = contrast_loss.forward(z);
        auto recon_loss = go.MeanSquaredError(recon, input, false);
        auto loss = go.Add(target, recon_loss, true);
        go.nodes = topological_sort(loss);
        const int epochs = 2000;
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            go.zero_grad();
            go.forward(nullptr, true);
            go.backward();
            go.ParameterUpdate(nullptr,false, 3e-4f);
            if (epoch % 1 == 0)  printf("Epoch %i, Loss: %f \n", epoch+1, go.loss);
            if (epoch % 100 == 0) {model.save("../best_models/cgm.bin");}
        }
    }

    go.clear_graph();
    auto samp = std::make_shared<NodeBackProp>("Sample Input", 1, 1, 28, 28, 1);
    auto generated = model.generate(samp);
    for(int i = 0; i < 10; i++)
    {
    
        go.forward(generated);
        StandardDeNorm(generated, 255.f,0.f,1.f);
        generated->reshape({generated->dim[0]*generated->dim[1], 1, 28,28});
        BPrintImage(generated, 1, 512, 512);
    }
    return 0;

}
