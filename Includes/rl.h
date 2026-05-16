#pragma once
#include "rl.h"

RL_Replay::RL_Replay(const int total, const int state_total)
{
    SafeCudaMalloc("State", state, total * state_total);
    SafeCudaMalloc("Traj", traj, total);
    SafeCudaMalloc("Returns", returns, total);
    SafeCudaMalloc("Advantages", advantages, total);
}

RL_Replay::~RL_Replay(const int total, const int state_total){cudaFree(state); cudaFree(traj); cudaFree(returns); cudaFree(advantages);}