#include "engine.h"
#include "image_loader.h"

int FOD(const int row, const int col, const std::vector<Convolute2D>& vecs) // Figure out dimensions
{
    int out_row = row;
    int out_col = col;
    for(const auto p: vecs)
    {
        out_row = ((out_row - p.c + 2 * p.pad) / p.stride) + 1;
        out_col = ((out_col - p.d + 2 * p.pad) / p.stride) + 1;
    }
    return vecs.back().out * out_row * out_col;
} 

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

class Model
{
private:
    Convolute2D conv1, conv2, conv3;
    Linear fc1, fc2;
public:
    Model(): 
    conv1(1,32,5,5,1,0, "Conv1"),conv2(32,32,5,5,1,0,"Conv2"),conv3(32,1,5,5,1,0, "Conv3"),
    fc1(FOD(28, 28, {conv1, conv2, conv3}), 64, "First Linear"), fc2(64, 10, "Projection"){}
    template<typename NormFn, typename ActFn>
    graph forward(const graph& Xm, NormFn norm, ActFn act)
    {
        auto A1 = act(norm(conv1.forward(Xm)));
        auto A2 = act(norm(conv2.forward(A1)));
        auto A3 = act(norm(conv3.forward(A2)));
        auto A4 = GraphOperations::Copy(A3);
        A4->reshape({A3->dim[0], 1, 1, (int)A3->total / A3->dim[0]});
        auto L1 = act(fc1.forward(A4));
        auto out = fc2.forward(L1); out->op_name = "Prediction";
        return out;
    }
};

int main()
{
    GraphOperations go;
    MNISTLoader loader(32,"../mnist/mnist_train.csv");
    Model model;
    auto [data, labels] = loader.load(32);
    auto prediction = model.forward(data, go.LayerNorm, go.GELU);
    auto loss = go.SoftMaxCrossEntropy(prediction, labels, true);
    go.nodes = topological_sort(loss);
    for(int i = 0; i < 10000; i++)
    {
        go.zero_grad();
        go.forward(nullptr, true);
        CheckError("Forward");
        go.backward();
        go.ParameterUpdate(nullptr, false, 1e-3f);
        if(i%100==0) printf("%i: loss: %f \n",i, go.loss);

    }
    go.clear_graph();loader.destroy();
    MNISTLoader new_loader(1, "../mnist/mnist_test.csv");
    auto [new_data, new_labels] = new_loader.load(1);
    auto new_prediction = model.forward(new_data, go.LayerNorm, go.GELU);
    double accuracy = 0.f;
    auto probs = go.SOFTMAX(new_prediction);
    for(int i = 0; i < 10000; i++)
    {
        go.forward(probs);
        accuracy += (ArgMaxToCPU(probs) == ArgMaxToCPU(new_labels)) ? 1.f : 0.f;
        if(i % 1000 == 0)
        {

            printGPU(probs);
            printGPU(new_labels);
            new_data->op_name = "Model Predicted: " + std::to_string(ArgMaxToCPU(new_prediction));
            BPrintImage(new_data, 5, 256, 256);
        }


        if(i == 10000 - 1) accuracy /= 10000.f;
    }
    printf("Prediction Accuracy: %.3f%% \n", 100 *accuracy);
    return 0;
}