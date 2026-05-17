#include "rl.h"


RL_Replay::RL_Replay(const int total, const int state_total) : total(total), state_total(state_total)
{
    SafeCudaMalloc("State", state, total * state_total);
    SafeCudaMalloc("Traj", traj, 5*total);
    SafeCudaMalloc("Returns", returns, total);
    SafeCudaMalloc("Advantages", advantages, total);
    next_state = nullptr;
}

RL_Replay::RL_Replay(const int total, const int state_total, const bool isDQN): total(total), state_total(state_total)
{
    SafeCudaMalloc("State", state, total * state_total);
    SafeCudaMalloc("Next State", next_state, total * state_total);
    SafeCudaMalloc("Traj", traj, 5*total);
    returns = nullptr; advantages = nullptr;
}

RL_Replay::~RL_Replay(){cudaFree(next_state); cudaFree(state); cudaFree(traj); cudaFree(returns); cudaFree(advantages);}