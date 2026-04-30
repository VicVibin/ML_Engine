#include "includes/engine.h"
#include <numeric>

class Multi_Linear_Residual_Block
{
    /*
    @brief: Required Activation and Normalization layer lambdas with reference capture for activation and normalization.. 
    */
private:
    GraphOperations& go;
public:
    const int input_dim, output_dim, residuals, hidden_dim, layers;
    std::vector<Linear*> sequence;
    Multi_Linear_Residual_Block(GraphOperations& go, const int input, const int output, const int num_residuals, const int layers, const int hidden_size);
    graph forward(const graph& X, graphFn activation, graphFn norm);
};

struct RL_Replay
{  
    /*
    @brief : Model.Q([B * T x 1 x 1 x V])-> [B  * T x 1 x 1 x A] ->Argmax -> Action values for PPO
            Model.V([B * T x 1 x 1 x V])-> [B  * T x 1 x 1 x 1]  
    @param: State: [B x 1 x T x V] Batches of Monte Carlo Trajectories
    @param: Traj:  [B x 1 x T x 5] [action, reward, log_prob_old, value_old, done]
    @param @brief: trajectories[a_off + 0] // action
            trajectories[a_off + 1] // reward
            trajectories[a_off + 2] // log_prob_old
            trajectories[a_off + 3] // value_old
            trajectories[a_off + 4] // done
    @param: Advantage: [B x 1 x T x 1]: Actual Advantage calculated from MC simulation
    @param: Returns: [B x 1 x T x 1]: Actual return calculated from MC simulation
 
    */
    int batch, traj_size, state_dim;
    float* state, *traj, *advantages, *returns;
    RL_Replay(const int batch_size, const int trajectory_size, const int state_dim) : 
    batch(batch_size), traj_size(trajectory_size), state_dim(state_dim)
    {
        SafeCudaMalloc("Batch Transition States", state, batch_size * trajectory_size * state_dim);
        SafeCudaMalloc("Batch Transition Trajectory", traj, batch_size * traj_size * 5);
        SafeCudaMalloc("RL Replay Advantages", advantages, batch*trajectory_size);
        SafeCudaMalloc("RL Replay Returns", returns, batch*trajectory_size);

    }
    
    ~RL_Replay()
    {
        cudaFree(state);cudaFree(returns);
        cudaFree(traj); cudaFree(advantages);
    }

};

class Actor_Critic
{ 
    public:
    GraphOperations& go;
    Multi_Linear_Residual_Block *A1, *A2;
    Linear *Q, *V;
    int in_dim, action_dim, hidden;
    Actor_Critic(GraphOperations& go, const int in_dim, const int action_dim, const int hidden_dim);
    std::pair<graph,graph> build_train(const graph& S_t);
};

struct PPOTrainer 
{
    Actor_Critic   &ac;
    RL_Replay &rep;
    GraphOperations &go;
    float clip_eps, c1, c2;
    int   epochs, batch;
    std::mt19937 rng{std::random_device{}()};
    PPOTrainer(GraphOperations& go, Actor_Critic& ac, RL_Replay& replay, const int ppo_epochs = 4, const int mini_batch = 64, 
               const float c1_ = 0.5f, const float c2_ = 0.01f, const float clip_eps_ = 0.2f);
    graph PPO_loss(const graph& states, const graph& actions, const graph& log_prob, const graph& advantages, const graph& returns);
    void PPO_loop(const graph &states_g, const graph &actions_g, const graph &log_probs_old, const graph& advantages_g, const graph& returns_g);
    void update(const float gamma = 0.99f, const float lam =0.95f);
};

