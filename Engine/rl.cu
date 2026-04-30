
#include "includes/rl.h"

Multi_Linear_Residual_Block::Multi_Linear_Residual_Block(GraphOperations& go, const int input, const int output, const int num_residuals, const int layers, const int hidden_size): 
    go(go), input_dim(input), output_dim(output), residuals(num_residuals), hidden_dim(hidden_size), layers(layers)
    {
        sequence.push_back(new Linear(go,input, hidden_size));
        for (int i = 1; i < num_residuals * layers; ++i)sequence.push_back(new Linear(go,hidden_size, hidden_size));
        sequence.push_back(new Linear(go, hidden_size, output));
    }

graph Multi_Linear_Residual_Block::forward(const graph& X, graphFn activation, graphFn norm)
{
        auto H = sequence[0]->forward(X);
        for (int r = 0; r < residuals; ++r){auto A = H; 
        for (int j = 0; j < layers; ++j){
            int idx = r * layers + j + 1; 
            if(idx < residuals * layers) A = activation(sequence[idx]->forward(A));
        } H = norm(go.Add(H, A));}
        return sequence[residuals * layers]->forward(H);
}

RL_Replay::RL_Replay(const int batch_size, const int trajectory_size, const int state_dim) : 
    batch(batch_size), traj_size(trajectory_size), state_dim(state_dim)
    {
        SafeCudaMalloc("Batch Transition States", state, batch_size * trajectory_size * state_dim);
        SafeCudaMalloc("Batch Transition Trajectory", traj, batch_size * traj_size * 5);
        SafeCudaMalloc("RL Replay Advantages", advantages, batch*trajectory_size);
        SafeCudaMalloc("RL Replay Returns", returns, batch*trajectory_size);

    }
    
RL_Replay::~RL_Replay()
    {
        cudaFree(state);cudaFree(returns);
        cudaFree(traj); cudaFree(advantages);
    }

Actor_Critic::Actor_Critic(GraphOperations& go, const int in_dim, const int action_dim, const int hidden_dim) : go(go), in_dim(in_dim), 
    action_dim(action_dim), hidden(hidden_dim)
    {
        A1 = new Multi_Linear_Residual_Block(go, in_dim, hidden_dim, 1,2,hidden_dim);
        A2 = new Multi_Linear_Residual_Block(go, hidden_dim, hidden_dim, 1,2,hidden_dim);
        Q = new Linear(go, hidden_dim, action_dim, "π(a|s)");
        V = new Linear(go, hidden_dim, 1, "V(S)");
    }

std::pair<graph,graph> Actor_Critic::build_train(const graph& S_t)
{
        auto h = A1->forward(S_t, [&](const graph x){return go.LeakyRELU(x);}, [&](const graph x){return go.LayerNorm(x);});
        h = A2->forward(h, [&](const graph x){return go.SILU(x);}, [&](const graph x){return go.LayerNorm(x);});
        return {Q->forward(h), V->forward(h)};
}

PPOTrainer::PPOTrainer(GraphOperations& go, Actor_Critic& ac, RL_Replay& replay, const int ppo_epochs = 4, 
               const int mini_batch = 64, const float c1_ = 0.5f, const float c2_ = 0.01f, 
               const float clip_eps_ = 0.2f): go(go), ac(ac), rep(replay), batch(mini_batch), 
               epochs(ppo_epochs), c1(c1_), c2(c2_), clip_eps(clip_eps_)
    {}

graph PPOTrainer::PPO_loss(const graph& states, const graph& actions, const graph& log_prob, const graph& advantages, const graph& returns)
{ 
        auto [logits, v_pred] = ac.build_train(states); // [B*T x 1 x 1 x A, B*T x 1 x 1 x 1]
        auto log_probs_new = go.Log(go.SOFTMAX(logits)); // [B*T x 1 x 1 x A]
        auto log_pa_new    = go.GatherAction(log_probs_new, actions); //[B*T x 1 x 1 x 1]
        auto log_ratio = go.Add(log_pa_new, go.Scale(log_prob, -1.0f)); // [B * T x 1 x 1 x 1]
        auto ratio     = go.Exp(log_ratio); // [B* T x 1 x 1 x 1]

        auto obj1   = go.Multiply(ratio, advantages);
        auto obj2   = go.Multiply(go.Clamp(ratio, 1.f - clip_eps, 1.f + clip_eps), advantages);
        auto L_clip = go.Scale(go.LAYERMEAN(go.Min(obj1, obj2)), -1.0f);
        auto L_vf   = go.Scale(go.MeanSquaredError(v_pred, returns, false), c1);
        auto L_ent  = go.Scale(go.LAYERMEAN(go.Entropy(logits)), -1.0f * c2);
        auto Last   = go.Add(go.Add(L_clip, L_vf), L_ent,true);
        return Last;
}

void PPOTrainer::PPO_loop(const graph &states_g, const graph &actions_g, const graph &log_probs_old, const graph& advantages_g, const graph& returns_g)
{
        const int total = rep.batch * rep.traj_size;
        const int tpb   = THREADSPERBLOCK;
        int* d_idx;

        SafeCudaMalloc("epoch_idx", d_idx, total);
        std::vector<int> h_idx(total);
        std::iota(h_idx.begin(), h_idx.end(), 0);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            std::shuffle(h_idx.begin(), h_idx.end(), rng);
            cudaMemcpy(d_idx, h_idx.data(), total * sizeof(int), cudaMemcpyHostToDevice);
            for (int start = 0; start < total; start += batch){
            if ((total - start) < batch) continue;
            static_assign_values<<<(rep.state_dim*batch+tpb-1)/tpb,tpb>>>(rep.traj, rep.advantages, rep.returns, rep.state, d_idx,
            actions_g->output,log_probs_old->output,advantages_g->output,returns_g->output,states_g->output,
            batch,rep.traj_size,rep.state_dim,start); CheckError("static_assign_trajectory");
            go.forward();go.backward();go.ParameterUpdate(); go.zero_grad();
        }}

        cudaFree(d_idx);
}

void PPOTrainer::update(const float gamma = 0.99f, const float lam =0.95f)
{
    /*
    @brief:
    Before update is called Replay Buffer generated from past experiences..
    Requires States and Trajectory to be filled before call makes sense
    */
    int N = rep.traj_size * rep.batch;
    rl_discounted_returns<<<rep.batch, 1>>>(rep.traj, rep.advantages, rep.returns,gamma, lam, rep.batch, rep.traj_size); 
    CheckError("RL Discounted returns");
    float* mean, *std;
    SafeCudaMalloc("Batch Mean", mean, 1);
    SafeCudaMalloc("Batch Std", std, 1);

    // ========================= \\ Advantage Normalization
    BatchMean<<<1,THREADSPERBLOCK>>>(rep.advantages,mean, N,1,1,1,true);
    BatchStd <<<1,THREADSPERBLOCK>>>(rep.advantages,mean,std,N,1,1,1);
    BNorm(rep.advantages,mean,std,N, 1,1,1,1.0f,0.0f, 1e-9f);
    // ========================= \\ 

    auto states_g        = std::make_shared<NodeBackProp>("States",       batch, 1, 1, rep.state_dim, 1);
    auto actions_g       = std::make_shared<NodeBackProp>("Actions",      batch, 1, 1, 1, 1); // Argmax of (Model.Q)
    auto log_probs_old_g = std::make_shared<NodeBackProp>("π_old(a | s)", batch, 1, 1, 1, 1);
    auto advantages_g    = std::make_shared<NodeBackProp>("A_t",          batch, 1, 1, 1, 1);
    auto returns_g       = std::make_shared<NodeBackProp>("G_t",          batch, 1, 1, 1, 1);
    auto loss            = PPO_loss(states_g, actions_g, log_probs_old_g, advantages_g, returns_g);
    go.nodes             = topological_sort(loss);

    PPO_loop(states_g, actions_g,log_probs_old_g, advantages_g, returns_g);
    states_g->clear();actions_g->clear(); log_probs_old_g->clear();
    advantages_g->clear(); returns_g->clear(); cudaFree(mean); cudaFree(std);
}



