#pragma once
#include <vector>
#include <deque>
#include <random>
#include <cstdint>
#include <array>

static constexpr int GRID_W   = 20;
static constexpr int GRID_H   = 20;
static constexpr int OBS_DIM  = 17;  
static constexpr int ACT_DIM  = 4;
static constexpr int MAT_TOT = (GRID_W + 2) * (GRID_H + 2);

struct STATE_OBS
{
    std::array<float, MAT_TOT> matrix{};
    std::array<float, OBS_DIM> grid{};
    const int dim[2] = {GRID_W + 2, GRID_H + 2};
    static constexpr int total = OBS_DIM;
    static constexpr int mtotal = MAT_TOT;
    
    void operator =(const STATE_OBS& other)
    {
        matrix = other.matrix;
        grid  = other.grid;
    }

};

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

    StepResult Obstep(Action a)
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
 
    STATE_OBS getObs() const
    {
        STATE_OBS o;
        o.grid = gridObs();
        o.matrix = matrixObs();
        return o;
    }

private:
    std::deque<Point> snake_;
    Point             food_{};
    Point             dir_{1, 0};
    int               score_ = 0;
    int               steps_ = 0;
    std::mt19937      rng_;

    bool insideGrid(Point p) const { return ( p.x >= 0 && p.x < GRID_W && p.y >= 0 && p.y < GRID_H);}
        
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

    float dangerN(int dx, int dy, int N) const
    {
        Point h = snake_.front();
        for (int i = 1; i <= N; ++i)
        {
            int nx = h.x + dx * i;
            int ny = h.y + dy * i;
            if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H)
                return 1.f - (float)(i - 1) / N;  
            for (auto& s : snake_)
                if (s.x == nx && s.y == ny)
                    return 1.f - (float)(i - 1) / N;
        }
        return 0.f;
    }

    float bodyDangerN(int dx, int dy, int max_steps) const 
    {
        Point h = snake_.front();
        for (int step = 1; step <= max_steps; ++step) {
            Point p = {h.x + dx * step, h.y + dy * step};
            if (!insideGrid(p)) break;               // stop at wall – not a body danger
            if (isBody(p))  return 1.0f / step;                  // closer body → higher danger
        }
        return 0.0f;
    }

    std::array<float, OBS_DIM> gridObs() const 
    {
        std::array<float, OBS_DIM> o{};
        Point h = snake_.front();

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
        o[14] = bodyDangerN( dir_.x,  dir_.y, 4);   // body ahead
        o[15] = bodyDangerN( dir_.y, -dir_.x, 4);   // body left
        o[16] = bodyDangerN(-dir_.y,  dir_.x, 4);   // body right
        return o;
    };

    std::array<float, MAT_TOT> matrixObs() const 
    {

        std::array<float, MAT_TOT> o{};
        constexpr int W = GRID_W + 2;  
        constexpr int H = GRID_H + 2; 

        auto idx = [&](int px, int py) { return py * W + px; };
        for (int px = 0; px < W; ++px) 
        {
            o[idx(px, 0)]     = -1.f;   // top row
            o[idx(px, H - 1)] = -1.f;   // bottom row
        }

        for (int py = 0; py < H; ++py) 
        {
            o[idx(0,     py)] = -1.f;   // left col
            o[idx(W - 1, py)] = -1.f;   // right col
        }
        
        o[idx(food_.x + 1, food_.y + 1)] = 1.f;

        const int n = (int)snake_.size();
        for (int i = 0; i < n; ++i) {
            const Point& p = snake_[i];
            int flat = idx(p.x + 1, p.y + 1);

            if (i == 0) { o[flat] = 0.8f;} 
            else { float t = (n > 2) ? (float)(i - 1) / (float)(n - 2) : 1.f; o[flat] = 0.5f - t * 0.4f;}
        }

        return o;
    }
};