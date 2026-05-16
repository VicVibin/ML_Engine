#pragma once
#include <vector>
#include <deque>
#include <random>
#include <cstdint>
#include <array>

static constexpr int GRID_W   = 20;
static constexpr int GRID_H   = 20;
static constexpr int OBS_DIM  = 14;  
static constexpr int ACT_DIM  = 4;

using STATE_OBS = std::array<float, OBS_DIM>;

enum class Action : int { UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3 };

struct Point { int x, y; };

struct StepResult
{
    STATE_OBS obs;
    float      reward;
    bool       done;
    int        score;
};

class SnakeEnv {
public:
    SnakeEnv() : rng_(std::random_device{}()) { reset(); }

    void reset()
    {
        snake_.clear();
        int cx = GRID_W / 2, cy = GRID_H / 2;
        snake_.push_back({cx,     cy});
        snake_.push_back({cx - 1, cy});
        snake_.push_back({cx - 2, cy});
        dir_   = {1, 0};
        score_ = 0;
        steps_ = 0;
        spawnFood();
        return;
    }

    StepResult step(Action a)
    {
        const int dx[4] = { 0,  1, 0, -1};
        const int dy[4] = {-1,  0, 1,  0};
        int ax = dx[(int)a], ay = dy[(int)a];

        // Prevent 180-degree reversal
        if (ax == -dir_.x && ay == -dir_.y) { ax = dir_.x; ay = dir_.y; }
        dir_ = {ax, ay};

        Point head = {snake_.front().x + ax, snake_.front().y + ay};
        ++steps_;

        float reward = -0.01f;
        bool  done   = false;

        if (head.x < 0 || head.x >= GRID_W ||
            head.y < 0 || head.y >= GRID_H ||
            isBody(head))
        {
            reward = -10.f;
            done   = true;
        }
        else if (head.x == food_.x && head.y == food_.y) {
    
        float length_bonus = 3.f + (float)snake_.size() / (float)(GRID_W * GRID_H);
        reward = 5.f * length_bonus;   // ranges from ~5.0 early to ~10.0 at max length
    
        ++score_;
        snake_.push_front(head);
        spawnFood();
        steps_ = 0;
        }
        else
        {
            snake_.push_front(head);
            snake_.pop_back();
        }

        if (!done && steps_ > GRID_W * GRID_H * 10)
        {
            reward = -1.f;
            done   = true;
        }

        return { getObs(), reward, done, score_ };
    }

    const std::deque<Point>& snake() const { return snake_; }
    Point food()  const { return food_;  }
    int   score() const { return score_; }

private:
    std::deque<Point> snake_;
    Point             food_{};
    Point             dir_{1, 0};
    int               score_ = 0;
    int               steps_ = 0;
    std::mt19937      rng_;

    bool isBody(Point p) const
    {
        for (auto& s : snake_) if (s.x == p.x && s.y == p.y) return true;
        return false;
    }

    void spawnFood()
    {
        std::uniform_int_distribution<int> dx(0, GRID_W - 1);
        std::uniform_int_distribution<int> dy(0, GRID_H - 1);
        do { food_ = {dx(rng_), dy(rng_)}; } while (isBody(food_));
    }

    // Returns how close the nearest obstacle is in direction (dx,dy),
    // as a value in (0, 1].  1.0 = wall/body immediately adjacent,
    // smaller = obstacle further away, 0.0 = clear for all N steps.
    // This gives the network a proportional "urgency" signal rather
    // than a flat binary that provides no gradient at distance > 1.
    float dangerN(int dx, int dy, int N) const
    {
        Point h = snake_.front();
        for (int i = 1; i <= N; ++i)
        {
            int nx = h.x + dx * i;
            int ny = h.y + dy * i;
            if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H)
                return 1.f - (float)(i - 1) / N;   // closer wall → higher value
            for (auto& s : snake_)
                if (s.x == nx && s.y == ny)
                    return 1.f - (float)(i - 1) / N;
        }
        return 0.f;
    }

    STATE_OBS getObs() const
    {
        STATE_OBS o{};
        Point h = snake_.front();

        // ── Relative danger with 4-step proportional lookahead ──────────────
        // Ahead, left (relative), right (relative)
        // Value: 1.0 = blocked next cell, 0.25 = blocked 4 cells away, 0.0 = clear
        o[0] = dangerN( dir_.x,  dir_.y,  4);   // straight ahead
        o[1] = dangerN( dir_.y, -dir_.x,  4);   // left  (rotate CCW)
        o[2] = dangerN(-dir_.y,  dir_.x,  4);   // right (rotate CW)

        // ── Current heading one-hots ─────────────────────────────────────────
        o[3] = (dir_.x ==  0 && dir_.y == -1) ? 1.f : 0.f;  // UP
        o[4] = (dir_.x ==  1 && dir_.y ==  0) ? 1.f : 0.f;  // RIGHT
        o[5] = (dir_.x ==  0 && dir_.y ==  1) ? 1.f : 0.f;  // DOWN
        o[6] = (dir_.x == -1 && dir_.y ==  0) ? 1.f : 0.f;  // LEFT

        // ── Food direction flags ─────────────────────────────────────────────
        o[7]  = (food_.x < h.x) ? 1.f : 0.f;   // food is left
        o[8]  = (food_.x > h.x) ? 1.f : 0.f;   // food is right
        o[9]  = (food_.y < h.y) ? 1.f : 0.f;   // food is above  (y=0 is top)
        o[10] = (food_.y > h.y) ? 1.f : 0.f;   // food is below

        // ── Normalized Manhattan distance to food ────────────────────────────
        // Gives the network a continuous gradient toward food rather than
        // binary direction flags alone.
        o[11] = std::abs(food_.x - h.x) / (float)GRID_W;
        o[12] = std::abs(food_.y - h.y) / (float)GRID_H;

        // ── Normalized body length ───────────────────────────────────────────
        // As the snake grows, self-collision risk rises; this lets the network
        // calibrate how cautious to be.
        o[13] = (float)snake_.size() / (float)(GRID_W * GRID_H);

        return o;
    }

    GRID_OBS getGridObs() const
    {
        constexpr int W = GRID_W + 2;
        constexpr int H = GRID_H + 2;
        GRID_OBS obs{};  
        for (int x = 0; x < W; ++x) { obs.grid[0 * W + x] = -1.f; obs.grid[(H-1) * W + x] = -1.f; }
        // Border: left and right columns
        
        for (int y = 0; y < H; ++y) { obs.grid[y * W + 0] = -1.f; obs.grid[y * W + (W-1)] = -1.f; }

        // Snake body
        for (size_t i = 1; i < snake_.size(); ++i) obs.grid[(snake_[i].y + 1) * W + (snake_[i].x + 1)] = -1.f;

        // Head
        if (!snake_.empty()) obs.grid[(snake_[0].y + 1) * W + (snake_[0].x + 1)] = 1.f;

        // Food
        obs.grid[(food_.y + 1) * W + (food_.x + 1)] = 5.f;

        return obs;
    }
};