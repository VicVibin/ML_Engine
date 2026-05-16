#include "diffusion.h"

VisionAttention::VisionAttention(GraphOperations &go_ref, const int Channels): go(go_ref), channels(Channels)
{
        Q = new Convolute2D(go, channels, channels,1,1,1,0, "Q"); 
        K = new Convolute2D(go, channels, channels,1,1,1,0, "K");
        V = new Convolute2D(go, channels, channels,1,1,1,0, "V");
        P = new Convolute2D(go, channels, channels,1,1,1,0, "P");
    

}
void VisionAttention::save(std::ofstream& f) const{Q->save(f); K->save(f); V->save(f); P->save(f);}
void VisionAttention::load(std::ifstream& f){Q->load(f); K->load(f); V->load(f); P->load(f);}
graph VisionAttention::forward(const graph& X)
{
    if(X->dim[1] != channels)
    {
        std::cout << "Shape mismatch in Attention::forward()... Expected: " 
        << channels << "channels | \t " << " Received: " << X->dim[1] << " channels \n";
    }
     
    auto X_in = go.LayerNorm(X);
    auto query = Q->forward(X_in);
    auto key = K->forward(X_in);     
    auto value = V->forward(X_in);   

    std::vector<int> init = {key->dim[0], key->dim[1], key->dim[2], key->dim[3]};
    std::vector<int> reshaped = {key->dim[0],1, key->dim[1], key->dim[2]*key->dim[3]};
    query->reshape(reshaped); key->reshape(reshaped); value->reshape(reshaped);
    auto pquery = go.Permute(query,0,1,3,2); // batch x seq_len x channels
    auto pkey = go.Permute(key,0,1,3,2);     // batch x seq_len x channels
    auto attn_scores = go.BMMABT(pquery,pkey); // batch x seq_len x seq_len
    auto scaled_attn = go.Scale(attn_scores, 1.0f/sqrtf((float)channels));
    auto softmax_attn = go.SOFTMAX(scaled_attn, 1); // batch x seq_len x seq_len
    auto attn_output = go.BMM(softmax_attn,go.Permute(value, 0,1,3,2)); // batch x seq_len x channels
    auto p_attn_output = go.Permute(attn_output,0,1,3,2); // batch x channels x seq_len
    p_attn_output->reshape(init);
    auto project = P->forward(p_attn_output); 
    auto output = go.LayerNorm(go.Add(X_in, project)); output->op_name = "Attention Output";
    return output;
}

VisionCrossAttention::VisionCrossAttention(GraphOperations &go_ref,const int Channels, const int ContextLen, const int EmbedDim): 
go(go_ref), channels(Channels), context_len(ContextLen), embed_dim(EmbedDim)
{   
        if(channels != embed_dim)
        {
            std::cout << "ERROR: channels (" << channels << ") must equal embed_dim ("<< embed_dim << ") for cross-attention\n";
            std::exit(1);
        }
        Q = new Convolute2D(go, channels,  channels, 1, 1, 1, 0, "Q");
        K = new Convolute2D(go, embed_dim,embed_dim, 1, 1, 1, 0, "K");
        V = new Convolute2D(go, embed_dim,embed_dim, 1, 1, 1, 0, "V");
        P = new Convolute2D(go, channels,  channels, 1, 1, 1, 0, "P");
}
void VisionCrossAttention::save(std::ofstream& f) const{Q->save(f); K->save(f); V->save(f); P->save(f);}
void VisionCrossAttention::load(std::ifstream& f) {Q->load(f); K->load(f); V->load(f); P->load(f);}
graph VisionCrossAttention::forward(const graph& X_in, const graph& Context)
{   
        if(Context == nullptr)
        {
            std::cout << "Context node is null in VisionCrossAttention::forward \n";
            std::exit(1);
        }

        if(X_in->dim[1] != channels)
        {
            std::cout << "Shape mismatch in VisionCrossAttention::forward()... Expected: " << batch << " x " << channels 
                      << " Received: " << X_in->dim[0] << " x " << X_in->dim[1] << "\n";
            std::exit(1);
        }

        if(Context->dim[0] != X_in->dim[0])
        {
            std::cout << "Context batch mismatch in AttentionT::forward()\n";
            std::exit(1);
        }
        
        if(embed_dim != channels)
        {
            std::cout << "Embed dim and channels must be equal in VisionCrossAttention::forward()\n";
            std::exit(1);
        }

        auto query = Q->forward(Context); // batch x embed dim == channels x seq_len
        auto key = K->forward(X_in);  // batch x channels x seq_len
        auto value = V->forward(X_in); // batch x channels x seq_len
        
        std::vector<int> init = {key->dim[0], key->dim[1], key->dim[2], key->dim[3]};
        std::vector<int> reshaped = {key->dim[0],1, key->dim[1], key->dim[2]*key->dim[3]};
        query->reshape(reshaped); key->reshape(reshaped); value->reshape(reshaped);

        auto pquery = go.Permute(query,0,1,3,2); // batch x seq_len x channels
        auto pkey = go.Permute(key,0,1,3,2);     // batch x seq_len x channels
        auto attn_scores = go.BMMABT(pquery,pkey); // batch x seq_len x seq_len
        auto scaled_attn = go.Scale(attn_scores, 1.0f/sqrtf((float)channels));
        auto softmax_attn = go.SOFTMAX(scaled_attn, 1); // batch x seq_len x seq_len
        auto attn_output = go.BMM(softmax_attn,go.Permute(value,0,1,3,2)); // batch x seq_len x channels
        auto p_attn_output = go.Permute(attn_output,0,1,3,2); // batch x channels x seq_len
        p_attn_output->reshape(init);
        auto project = P->forward(p_attn_output);
        auto output = go.Add(X_in, project); output->op_name = "Cross-Attention Output";
        return output;
        


}

ResidualBlock::ResidualBlock(GraphOperations &go_ref, const int in_channels, const int out_channels, const int t_hidden, const int stride): 
    go(go_ref), in(in_channels), out(out_channels), hidden(t_hidden), stride(stride)
    {
        conv1 = new Convolute2D(go, in_channels,  out_channels,3,3,stride,1, "ResBlock Conv1");
        conv2 = new Convolute2D(go, out_channels, out_channels,3,3,1,1, "ResBlock Conv2");
        time_mlp = new Linear(go, t_hidden, out_channels, "ResBlock Time MLP");
        skipconv = new Convolute2D(go,in_channels, out_channels,1,1,stride,0, "ResBlock SkipConv");
        skip = new Identity(go, "Resblock Identity SkipConv");
        
    }
void ResidualBlock::save(std::ofstream& f) const{
        conv1->save(f);
        conv2->save(f);
        skipconv->save(f);
        time_mlp->save(f);
    }
void ResidualBlock::load(std::ifstream& f){
        conv1->load(f);
        conv2->load(f);
        skipconv->load(f);
        time_mlp->load(f);}
graph ResidualBlock::forward(const graph& x, const graph & t_emb)
    {
        auto h = go.SILU(go.GroupNorm(conv1->forward(x)));
        auto time = time_mlp->forward(t_emb);
        h = go.Broadcast_Channel(h, time);
        h = go.GroupNorm(conv2->forward(h));
        auto skipnet = (in != out || stride != 1) ? skipconv->forward(x) : skip->forward(x);
        return go.SILU(go.Add(h, skipnet));
    }

LinearBlock::LinearBlock(GraphOperations &go_ref, const int in_features, const int out_features, const int t_hidden, const int stride): 
    go(go_ref), in(in_features), out(out_features), hidden(t_hidden), stride(stride)
    {
        conv1 = new Linear(go, in, out, "Linear 1");
        conv2 = new Linear(go, out, out, "Linear 1");
        time_mlp = new Linear(go, t_hidden, out, "ResBlock Time MLP");
        skipconv = new Linear(go, in, out, "ResBlock SkipConv");
        skip = new Identity(go, "Resblock Identity SkipConv");
        
    }
void LinearBlock::save(std::ofstream& f) const{
        conv1->save(f);
        conv2->save(f);
        skipconv->save(f);
        time_mlp->save(f);
    }
void LinearBlock::load(std::ifstream& f){
        conv1->load(f);
        conv2->load(f);
        skipconv->load(f);
        time_mlp->load(f);}
graph LinearBlock::forward(const graph& x, const graph & t_emb)
    {
        auto h = go.SILU(go.LayerNorm(conv1->forward(x)));
        auto time = time_mlp->forward(t_emb);
        h = go.Bias_Add(h, time);
        h = go.LayerNorm(conv2->forward(h));
        auto skipnet = (in != out || stride != 1) ? skipconv->forward(x) : skip->forward(x);
        return go.SILU(go.Add(h, skipnet));
    }

L_NET::L_NET(GraphOperations& goref,const graph& input, const graph& target, const graph& text_embed, 
    const int in_channels, const int out_channels, const int init_depth, const int t_node): 
    batch(input->dim[0]), go(goref), t_embed_dim(t_node), t_hidden(t_node/2), 
    input(input), target(target), text_embed(text_embed)
    
    {
    if(SHOULDNORM) 
    {
        SafeCudaMalloc("Global Scale", global_scale, 1); 
        fillKernel<<<1,1>>>(global_scale, 0.0, 1);
        cudaDeviceSynchronize(); 
        //CheckError("Fill Kernel for Global Scale");
    }
        
    time_mlp = new TimeMLPBlock(go, t_embed_dim, t_hidden);
    enc1 = new LinearBlock(go, in_channels, init_depth, t_hidden);
    enc2 = new LinearBlock(go, init_depth, init_depth * 2, t_hidden, 2);
    enc3 = new LinearBlock(go, init_depth * 2, init_depth * 4, t_hidden, 2);
    enc4 = new LinearBlock(go, init_depth * 4, init_depth * 8, t_hidden, 2);

    b0 = new LinearBlock(go, init_depth * 8, init_depth * 8, t_hidden);
    b1 = new LinearBlock(go, init_depth * 8, init_depth * 8, t_hidden);

    up1 = new Linear(go, init_depth * 8, init_depth * 8, "UpLinear 1");
    dec1 = new LinearBlock(go, init_depth * 8 + init_depth * 4, init_depth * 4, t_hidden);

    up2 = new Linear(go, init_depth * 4, init_depth * 4, "UpLinear 2");
    dec2 = new LinearBlock(go, init_depth * 4 + init_depth * 2, init_depth * 2, t_hidden);

    up3 = new Linear(go, init_depth * 2, init_depth * 2, "UpLinear 3");
    dec3 = new LinearBlock(go, init_depth * 2 + init_depth, init_depth, t_hidden);

    out = new Linear(go, init_depth, out_channels, "Output Layer");
    }

void L_NET::save(const str& filename) const
{

        std::filesystem::path path(filename);
        std::filesystem::path dir = path.parent_path();

        if (!dir.empty())
        {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec)
            {
                std::cerr << "Failed to create directory: " << dir << "\n";
                std::exit(1);
            }
        }

        str tmp = filename + ".tmp";

        std::ofstream f(tmp, std::ios::binary);
        if (!f)
        {
            std::cerr << "Error opening temp file: " << tmp << "\n";
            std::exit(1);
        }

        time_mlp->save(f);
        enc1->save(f); enc2->save(f); enc3->save(f); enc4->save(f);
        b0->save(f); b1->save(f);
        dec1->save(f); dec2->save(f); dec3->save(f);
        up1->save(f); up2->save(f); up3->save(f);
        out->save(f);
        

        if (!f.good())
        {
            std::cerr << "Error during write to temp file\n";
            f.close();
            std::filesystem::remove(tmp);
            std::exit(1);
        }

   
        f.flush();
        f.close();

        std::error_code ec;

    std::filesystem::rename(tmp, filename, ec);

    if (ec)
    {
        std::error_code remove_ec;
        std::filesystem::remove(filename, remove_ec);

        std::filesystem::rename(tmp, filename, ec);

        if (ec)
        {
            std::cerr << "Failed to rename temp file to final: " << filename << "\n";
            std::cerr << "Temp file left at: " << tmp << "\n";
            std::exit(1);
        }
    }

    std::cout << "Model saved successfully to " << filename << "\n";

}

void L_NET::load(const str& filename)
    {
        std::ifstream f(filename, std::ios::binary);
    if (!f){
        std::cerr << "Error opening file for loading: " << filename << "\n";
        std::exit(1);
        }

    time_mlp->load(f);
    enc1->load(f); enc2->load(f); enc3->load(f); enc4->load(f);
    b0->load(f); b1->load(f);
    dec1->load(f); dec2->load(f); dec3->load(f);
    up1->load(f); up2->load(f); up3->load(f);
    out->load(f);

    f.peek();
    if (!f.eof())
        std::cerr << "Warning: file '" << filename << "' has leftover bytes after loading — "
                  << "save/load structure may be mismatched\n";

    f.close();
    std::cout << "Model loaded successfully from " << filename << "\n";
    }

void L_NET::build_train(){
        auto t_emb = go.PositionalEncoding(t,t_embed_dim);   
        auto t_mlp = time_mlp->forward(t_emb); 

        auto e1 = enc1->forward(input, t_mlp);  
        auto e2 = enc2->forward(e1, t_mlp); 
        auto e3 = enc3->forward(e2, t_mlp);     
        auto e4 = enc4->forward(e3, t_mlp);

        // Attention
        auto b_0  = b0->forward(e4, t_mlp);
        auto b_1  = b1->forward(b_0, t_mlp);
        
        // FIX: Upsample THEN concatenate
        auto up_1 = up1->forward(b_1);   auto d1_in = go.CopyConcat(up_1, e3);  auto d1 = dec1->forward(d1_in, t_mlp); 
        auto up_2 = up2->forward(d1);    auto d2_in = go.CopyConcat(up_2, e2);  auto d2 = dec2->forward(d2_in, t_mlp);
        auto up_3 = up3->forward(d2);    auto d3_in = go.CopyConcat(up_3, e1);  auto d3 = dec3->forward(d3_in, t_mlp);
        auto logits = out->forward(d3);
        auto loss = go.MeanSquaredError(logits, target, true);

        go.nodes = topological_sort(loss);
        prediction = logits;

        

    }

void L_NET::build_inference(const graph& test_input)
    {
        auto t_emb = go.PositionalEncoding(t,t_embed_dim);   
        auto t_mlp = time_mlp->forward(t_emb); 

        auto e1 = enc1->forward(test_input, t_mlp);  
        auto e2 = enc2->forward(e1, t_mlp); 
        auto e3 = enc3->forward(e2, t_mlp);     
        auto e4 = enc4->forward(e3, t_mlp);

        auto b_0  = b0->forward(e4, t_mlp);
        auto b_1  = b1->forward(b_0, t_mlp);
        
        auto up_1 = up1->forward(b_1);   auto d1_in = go.CopyConcat(up_1, e3);  auto d1 = dec1->forward(d1_in, t_mlp); 
        auto up_2 = up2->forward(d1);    auto d2_in = go.CopyConcat(up_2, e2);  auto d2 = dec2->forward(d2_in, t_mlp);
        auto up_3 = up3->forward(d2);    auto d3_in = go.CopyConcat(up_3, e1);  auto d3 = dec3->forward(d3_in, t_mlp);
        prediction = out->forward(d3);
        go.nodes = topological_sort(prediction);
    }

void L_NET::zero_grad(){go.zero_grad();}
void L_NET::forward(){go.forward(); loss = go.loss;}
void L_NET::backward(){ go.backward();if(SHOULDNORM) {cudaMemset(global_scale, 0, sizeof(double)); go.accumulate(global_scale); go.clipNorm(global_scale);}}
void L_NET::parameterUpdate(){go.ParameterUpdate();}
void L_NET::printvals(){for (const auto&node: go.nodes) printHeadGPU(node);}   
void L_NET::printgrads(){for (const auto&node: go.nodes) printHeadGPU(node, 1);}
void L_NET::printparams(){for (const auto&node: go.nodes) if(node->printparams) printHeadGPU(node);}
void L_NET::train()
    {
        Timing forw("Forward"), back("Backward");
        zero_grad(); forward(); 
         backward(); parameterUpdate();
    }


U_NET::U_NET(GraphOperations& goref,const graph& input, const graph& target, const graph& text_embed, 
    const int in_channels, const int out_channels, const int init_depth, const int t_node): 
    batch(input->dim[0]), go(goref), t_embed_dim(t_node), t_hidden(t_node/2), 
    input(input), target(target), text_embed(text_embed)
    
    {
    if(SHOULDNORM) 
    {
        SafeCudaMalloc("Global Scale", global_scale, 1); 
        fillKernel<<<1,1>>>(global_scale, 0.0, 1);
        cudaDeviceSynchronize(); 
        //CheckError("Fill Kernel for Global Scale");
    }
        
    time_mlp = new TimeMLPBlock(go, t_embed_dim, t_hidden);
    enc1 = new ResidualBlock(go, in_channels, init_depth, t_hidden);
    enc2 = new ResidualBlock(go, init_depth, init_depth * 2, t_hidden, 2);
    enc3 = new ResidualBlock(go, init_depth * 2, init_depth * 4, t_hidden, 2);
    enc4 = new ResidualBlock(go, init_depth * 4, init_depth * 8, t_hidden, 2);

    b0 = new ResidualBlock(go, init_depth * 8, init_depth * 8, t_hidden);
    b1 = new ResidualBlock(go, init_depth * 8, init_depth * 8, t_hidden);

    up1 = new Convolute2DT(go, init_depth * 8, init_depth * 8, 2, 2, 2, 0, "UpConv 1");
    dec1 = new ResidualBlock(go, init_depth * 8 + init_depth * 4, init_depth * 4, t_hidden);

    up2 = new Convolute2DT(go, init_depth * 4, init_depth * 4, 2, 2, 2, 0, "UpConv 2");
    dec2 = new ResidualBlock(go, init_depth * 4 + init_depth * 2, init_depth * 2, t_hidden);

    up3 = new Convolute2DT(go, init_depth * 2, init_depth * 2, 2, 2, 2, 0, "UpConv 3");
    dec3 = new ResidualBlock(go, init_depth * 2 + init_depth, init_depth, t_hidden);

    out = new Convolute2D(go, init_depth, out_channels, 1,1,1,0, "Output Convolution");
    }

void U_NET::save(const str& filename) const
{

        std::filesystem::path path(filename);
        std::filesystem::path dir = path.parent_path();

        if (!dir.empty())
        {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec)
            {
                std::cerr << "Failed to create directory: " << dir << "\n";
                std::exit(1);
            }
        }

        str tmp = filename + ".tmp";

        std::ofstream f(tmp, std::ios::binary);
        if (!f)
        {
            std::cerr << "Error opening temp file: " << tmp << "\n";
            std::exit(1);
        }

        time_mlp->save(f);
        enc1->save(f); enc2->save(f); enc3->save(f); enc4->save(f);
        b0->save(f); b1->save(f);
        dec1->save(f); dec2->save(f); dec3->save(f);
        up1->save(f); up2->save(f); up3->save(f);
        out->save(f);
        

        if (!f.good())
        {
            std::cerr << "Error during write to temp file\n";
            f.close();
            std::filesystem::remove(tmp);
            std::exit(1);
        }

   
        f.flush();
        f.close();

        std::error_code ec;

    std::filesystem::rename(tmp, filename, ec);

    if (ec)
    {
        std::error_code remove_ec;
        std::filesystem::remove(filename, remove_ec);

        std::filesystem::rename(tmp, filename, ec);

        if (ec)
        {
            std::cerr << "Failed to rename temp file to final: " << filename << "\n";
            std::cerr << "Temp file left at: " << tmp << "\n";
            std::exit(1);
        }
    }

    std::cout << "Model saved successfully to " << filename << "\n";

}

void U_NET::load(const str& filename)
    {
        std::ifstream f(filename, std::ios::binary);
        if(!f.is_open())
        {
            std::cerr << "Failed to open: " << strerror(errno) << "\n";
            std::exit(1);
        }
        if (!f){
        std::cerr << "Error opening file for loading: " << filename << "\n";
        std::exit(1);
        }

    time_mlp->load(f);
    enc1->load(f); enc2->load(f); enc3->load(f); enc4->load(f);
    b0->load(f); b1->load(f);
    dec1->load(f); dec2->load(f); dec3->load(f);
    up1->load(f); up2->load(f); up3->load(f);
    out->load(f);

    f.peek();
    if (!f.eof())
        std::cerr << "Warning: file '" << filename << "' has leftover bytes after loading — "
                  << "save/load structure may be mismatched\n";

    f.close();
    std::cout << "Model loaded successfully from " << filename << "\n";
    }

void U_NET::build_train(){
        auto t_emb = go.PositionalEncoding(t,t_embed_dim);   
        auto t_mlp = time_mlp->forward(t_emb); 

        auto e1 = enc1->forward(input, t_mlp);  
        auto e2 = enc2->forward(e1, t_mlp); 
        auto e3 = enc3->forward(e2, t_mlp);     
        auto e4 = enc4->forward(e3, t_mlp);

        // Attention
        auto b_0  = b0->forward(e4, t_mlp);
        auto b_1  = b1->forward(b_0, t_mlp);
        
        // FIX: Upsample THEN concatenate
        auto up_1 = up1->forward(b_1);   auto d1_in = go.CopyCrop(up_1, e3);  auto d1 = dec1->forward(d1_in, t_mlp); 
        auto up_2 = up2->forward(d1);    auto d2_in = go.CopyCrop(up_2, e2);  auto d2 = dec2->forward(d2_in, t_mlp);
        auto up_3 = up3->forward(d2);    auto d3_in = go.CopyCrop(up_3, e1);  auto d3 = dec3->forward(d3_in, t_mlp);
        auto logits = out->forward(d3);

        auto loss = go.MeanSquaredError(logits, target, true);
        go.nodes = topological_sort(loss);
        prediction = logits;

        

    }

void U_NET::build_inference(const graph& test_input)
    {
        auto t_emb = go.PositionalEncoding(t,t_embed_dim);   
        auto t_mlp = time_mlp->forward(t_emb); 

        auto e1 = enc1->forward(test_input, t_mlp);  
        auto e2 = enc2->forward(e1, t_mlp); 
        auto e3 = enc3->forward(e2, t_mlp);     
        auto e4 = enc4->forward(e3, t_mlp);

        auto b_0  = b0->forward(e4, t_mlp);
        auto b_1  = b1->forward(b_0, t_mlp);
        
        auto up_1 = up1->forward(b_1);   auto d1_in = go.CopyCrop(up_1, e3);  auto d1 = dec1->forward(d1_in, t_mlp); 
        auto up_2 = up2->forward(d1);    auto d2_in = go.CopyCrop(up_2, e2);  auto d2 = dec2->forward(d2_in, t_mlp);
        auto up_3 = up3->forward(d2);    auto d3_in = go.CopyCrop(up_3, e1);  auto d3 = dec3->forward(d3_in, t_mlp);
        prediction = out->forward(d3);
        go.nodes = topological_sort(prediction);
    }

void U_NET::zero_grad(){go.zero_grad();}
void U_NET::forward(){go.forward(); loss = go.loss;}
void U_NET::backward(){ go.backward();if(SHOULDNORM) {cudaMemset(global_scale, 0, sizeof(double)); go.accumulate(global_scale); go.clipNorm(global_scale);}}
void U_NET::parameterUpdate(){go.ParameterUpdate();}
void U_NET::printvals(){for (const auto&node: go.nodes) printHeadGPU(node);}   
void U_NET::printgrads(){for (const auto&node: go.nodes) printHeadGPU(node, 1);}
void U_NET::printparams(){for (const auto&node: go.nodes) if(node->printparams) printHeadGPU(node);}
void U_NET::train()
    {
        Timing forw("Forward"), back("Backward");
        zero_grad(); forward(); 
         backward(); parameterUpdate();
    }



/*
int main()
{
    GraphOperations go; 
    const int T = 1000, init_depth = 32, t_hidden = 128, img_size = 128, epochs = 30000;
    int t;

    std::random_device rd;
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, T-1);

    auto base = Bi2n("KPOP", 1, img_size, img_size); // Loads a batch of images and converts to a node 
    StandardNorm(base); // Normalizes using (X/255 - 0.5) / 0.5
    auto input  = go.like(base, "Input Image"), target = go.like(base,"Target Image");

    U_NET model(go,input,target,nullptr,3,3,init_depth,t_hidden); 
    model.load("best_models/2huntrix.bin");
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

    model.build_inference(input);
    auto test = go.like(base, "Test Image");
    Noise(test);
    Sampling<U_NET> sampler(model, test, T-1,T);
    BPrintImage(test);
    sampler.loop(T-1, 0);
    sampler.display(512,512);
    return 0;
}




int main(){
    const int MAX_VOCAB_SIZE = 25000;
    const int MAX_BATCH_SIZE = 64;
    const int MAX_CONTEXT_LEN = 256;
    const int EMBED_DIM = 256;
    const int context_len = 32;
    const int HIDDEN_DIM = 256;

    GraphOperations go;
    TextualEmbedding embedder(EMBED_DIM,MAX_BATCH_SIZE, MAX_CONTEXT_LEN, MAX_VOCAB_SIZE, true);
    Text Db = LoadStory("C:/Users/victo/Documents/Coding_Projects/DeepSeek/sherlock");
    Text Database = read_words(Db, 0, Db.size());
    embedder.updateVocabulary(Database);
    printf("Total Vocabulary Size: %i | Total Word Count: %i \n", embedder.Vocabulary.size(), Database.size());
    LLM model(go, embedder, Database, MAX_BATCH_SIZE, context_len, HIDDEN_DIM);
    model.train(2000, 100);
    model.generate({"so","sherlock", "holmes", "said"}, 128);

 
}
*/