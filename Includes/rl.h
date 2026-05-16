#pragma once
#include "engine.h"
#include <numeric>

namespace REPLAY {constexpr int ACTION=0; constexpr int REWARD=1; constexpr int LOGPROB=2; constexpr int VALUE=3; constexpr int DONE=4;};

struct RL_Replay{  
/**
 * @brief PPO replay/training tensor layout documentation.
 *
 * @details
 * Trajectory sizes are inconsistent because episodes may terminate
 * early or continue much longer depending on gameplay.
 *
 * V can represent any N-dimensional state space (currently limited
 * to 3 dimensions because NodeBackProp only supports up to 4D tensors, B x 3D).
 *
 * Tensor Layouts:
 *
 * Model.Q:
 *   [T x 1 x 1 x V] -> [T x 1 x 1 x A]
 *   Produces action logits/values used for PPO action selection.
 *
 * Model.V:
 *   [T x 1 x 1 x V] -> [T x 1 x 1 x 1]
 *   Produces state value estimates.
 *
 * State:
 *   [T x 1 x 1 x V]
 *   Batched state tensor across trajectory timesteps.
 *
 * Traj:
 *   [T x 1 x 1 x 5]
 *   Stores:
 *     [action, reward, log_prob_old, value_old, done]
 *
 * Trajectory indexing:
 *   trajectories[a_off + REPLAY::ACTION]   -> action
 *   trajectories[a_off + REPLAY::REWARD]   -> reward
 *   trajectories[a_off + REPLAY::LOGPROB]  -> old log probability
 *   trajectories[a_off + REPLAY::VALUE]    -> old value estimate
 *   trajectories[a_off + REPLAY::DONE]     -> done flag
 *
 * Advantage:
 *   [T x 1 x 1 x 1]
 *   Monte-Carlo / GAE advantage estimates.
 *
 * Returns:
 *   [T x 1 x 1 x 1]
 *   Monte-Carlo discounted returns.
 */
    int total, state_total;
    float* state, *traj, *advantages, *returns;
    
    RL_Replay(const int total, const int state_total);
    ~RL_Replay();
};



template<class AC>
struct PPOTrainer 
{
    AC &ac;
    RL_Replay &rep;
    GraphOperations &go;
    float clip_eps, c1, c2;
    int   epochs, batch;
    std::mt19937 rng{std::random_device{}()};

    PPOTrainer(GraphOperations& go, AC& ac, RL_Replay& replay, const int ppo_epochs = 4, const int mini_batch = 64, 
    const float c1_ = 0.5f, const float c2_ = 0.01f, const float clip_eps_ = 0.2f) : go(go), ac(ac), rep(replay), batch(mini_batch),
    epochs(ppo_epochs), c1(c1_), c2(c2_), clip_eps(clip_eps_){};
            
   graph PPO_loss(const graph& states, const graph& actions, const graph& log_prob, const graph& advantages, const graph& returns)
    {
        auto [probs, v_pred] = ac.build_train(states); // [B x 4], [B x 1] 
        auto log_probs_new = go.Log(probs);  log_probs_new->op_name = "New Log Probabilities"; // [B x A]
        auto log_pa_new    = go.GatherAction(log_probs_new, actions); log_pa_new->op_name = "New Log Action Probability"; // [B x 1]
        auto log_ratio = go.Add(log_pa_new, go.Scale(log_prob, -1.0f)); log_ratio->op_name = "Log Ratio"; // [B x 1]
        auto ratio     = go.Exp(log_ratio); ratio->op_name = "Ratio"; //[B x 1]
        auto obj1   = go.Multiply(ratio, advantages); obj1->op_name = "Objective 1"; // [B x 1]
        auto obj2   = go.Multiply(go.Clamp(ratio, 1.0f - clip_eps, 1.0f + clip_eps), advantages); obj2->op_name = "Objective 2"; //[B x 1]
        auto L_clip = go.Scale(go.BATCHMEAN(go.Min(obj1, obj2)), -1.0f); L_clip->op_name = "Policy Loss Function"; //[1 x 1]
        auto L_vf   = go.Scale(go.MeanSquaredError(v_pred, returns, false), c1); L_vf->op_name = "Value Function Loss"; //[1x1]
        auto L_ent  = go.Scale(go.BATCHMEAN(go.Entropy(probs)), c2); L_ent->op_name = "Entropy Loss";
        auto Last   = go.Add(go.Add(L_clip, L_vf), L_ent, true); Last->op_name = "Final PPO Loss";
        return Last;
    };

    void PPO_loop(const graph &states_g, const graph &actions_g, const graph &log_probs_old, const graph& advantages_g, const graph& returns_g)
    {
        const int total = rep.total; 
        const int tpb   = THREADSPERBLOCK;
        int* d_idx;
        SafeCudaMalloc("epoch_idx", d_idx, total);
        std::vector<int> h_idx(total);
        std::iota(h_idx.begin(), h_idx.end(), 0);
        int tot = go.nodes.size();
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
        std::shuffle(h_idx.begin(), h_idx.end(), rng);
        cudaMemcpy(d_idx, h_idx.data(), total * sizeof(int), cudaMemcpyHostToDevice);
        for (int start = 0; start < total; start += batch)
        {
            if ((total - start) < batch) continue;  

            const int bpg = (rep.state_total * batch + tpb - 1) / tpb;
            static_assign_values<<<bpg, tpb>>>(rep.traj,rep.advantages,rep.returns,rep.state,d_idx,actions_g->output,
            log_probs_old->output,advantages_g->output,returns_g->output,states_g->output,batch,rep.state_total,start);
            CheckError("Static Assign Values");
            go.zero_grad();
            go.forward();
            go.backward();
            go.ParameterUpdate();

        }}
        cudaMemcpy(&ac.pi_loss, go.nodes.end()[-1]->inputs[0]->inputs[0]->output, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ac.v_loss , go.nodes.end()[-1]->inputs[0]->inputs[1]->inputs[0]->output, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ac.entropy, go.nodes.end()[-1]->inputs[1]->inputs[0]->output, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_idx);
    };

    void update(const graph&states_g, const float gamma = 0.99f, const float lam =0.95f, const float next_val = 0.0f)
    {
        /*
            @brief:
            Before update is called Replay Buffer generated from past experiences..
            Requires States and Batch trajectory to be filled before call makes sense
        */

        returns(gamma, lam, next_val);
        auto actions_g       = std::make_shared<NodeBackProp>("Actions", batch, 1, 1, 1, 1); // Argmax of (Model.Q)
        auto log_probs_old_g = go.like(actions_g, "π_old(a|s)");
        auto advantages_g    = go.like(actions_g, "A_t");
        auto returns_g       = go.like(actions_g, "G_t");

        states_g->forward = [=](){};
        actions_g->forward = [=](){};
        actions_g->backward = [=](){};
        actions_g->free = [=](){actions_g->clear();};

        auto loss            = PPO_loss(states_g, actions_g, log_probs_old_g, advantages_g, returns_g);
        go.nodes             = topological_sort(loss);

        PPO_loop(states_g, actions_g,log_probs_old_g, advantages_g, returns_g);
        go.clear_graph();
    
    };
    
    void returns(const float gamma, const float lam, const float next_val = 0.0f) 
    {
        int N = rep.total;
        std::vector<float> traj(N * 5);
        cudaMemcpy(traj.data(), rep.traj, N * 5 * sizeof(float), cudaMemcpyDeviceToHost);
        std::vector<float> advantages(N), returns(N);

        float gae        = 0.f;
        float next_value = next_val;
        for (int t = N - 1; t >= 0; --t)
        {
            int   a_off   = t * 5;
            float reward  = traj[a_off + REPLAY::REWARD];
            float value   = traj[a_off + REPLAY::VALUE];
            float done    = traj[a_off + REPLAY::DONE];

            float mask    = 1.f - done;                          
            float delta   = reward + gamma * next_value * mask - value; 
            gae           = delta  + gamma * lam * mask * gae;
            advantages[t] = gae;
            returns[t]    = gae + value; 
            next_value    = value;
        }

        float mean = 0.f;
        for (const float &a : advantages) mean += a;
        mean /= (float)N;

        float var = 0.f;
        for (const float &a : advantages) var += (a - mean) * (a - mean);
        var /= (float)N;

        float inv_std = 1.f / sqrtf(var + 1e-27f);
        for (float &a : advantages) a = (a - mean) * inv_std;

        cudaMemcpy(rep.advantages, advantages.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(rep.returns,    returns.data(),    N * sizeof(float), cudaMemcpyHostToDevice);
    };
};

template<class DQN>
struct DQNTrainer
{
    DQN &Q, &Qx;
    RL_Replay &rep;
    GraphOperations &go;
    std::mt19937 rng{std::random_device{}()};
    float gamma;
    int batch, epochs;
    DQNTrainer(GraphOperations& go, DQN& Q_on, DQN&Q_max, RL_Replay& replay, const float gam,
        const int b, const int e): go(go), Q(Q_on), Qx(Q_max), rep(replay), gamma(gam),
        batch(b), epochs(e) {}
    
    void update(const graph& state, float* target, float* target_indices)
    {
        /*
            Clear graph before function call
            Prefilled state and next state in rep.state, rep.next_state;
            Prefilled buffer of traj = {action, reward, 0,Q_max(next_state),done}
        */
        auto logits = Q.build_train(state);
        auto loss = go.MeanSquaredError(logits, target, target_indices, true);
        const int total = rep.total; 
        const int tpb   = THREADSPERBLOCK;

        int* d_idx;
        SafeCudaMalloc("epoch_idx", d_idx, total);
        std::vector<int> h_idx(total);
        std::iota(h_idx.begin(), h_idx.end(), 0);
        int tot = go.nodes.size();
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
        std::shuffle(h_idx.begin(), h_idx.end(), rng);
        cudaMemcpy(d_idx, h_idx.data(), total * sizeof(int), cudaMemcpyHostToDevice);
        
        for (int start = 0; start < total; start += batch)
        {
            if ((total - start) < batch) continue;  
            const int bpg = (rep.state_total * batch + tpb - 1) / tpb;
            dqn_assign_values<<<bpg,tpb>>>(rep.state, rep.traj, state->output,
                target, target_indices,d_idx,rep.state_total,batch,start,gamma);
            CheckError("dqn_assign_values");
            go.zero_grad(loss);
            go.forward();
            go.backward();
            go.ParameterUpdate();
        }}
        
        cudaMemcpy(&Q.pi_loss,loss->output,sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(d_idx);
        go.clear_graph(loss);
    };
    
};


