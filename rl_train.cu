#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <algorithm>
#include <deque>
#include <numeric>
#include <span>
#include "snake_env.h"
#include "rl.h"

__global__ void replace_targets(const float* X, float* traj, const int total, const int act_dim, const float gamma)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= total) return;
    float qmax = -FLT_MAX;
    for (int a = 0; a < act_dim; ++a)
        qmax = fmaxf(qmax, X[idx * act_dim + a]);

    const float r    = traj[5 * idx + REPLAY::REWARD];
    const float done = traj[5 * idx + REPLAY::DONE];
    traj[5 * idx + REPLAY::VALUE] = r + gamma * qmax * (1.f - done);
}

static constexpr int CELL           = 24;
static constexpr int GAME_W         = GRID_W * CELL;
static constexpr int GAME_H         = GRID_H * CELL;
static constexpr int PANEL_W        = 320;
static constexpr int WIN_W          = GAME_W + PANEL_W;
static constexpr int WIN_H          = GAME_H;
static constexpr int MAX_SCORE_HIST = 200;
static constexpr int ROLLOUT        = 2048;

// ======  PPO ===== // 
static constexpr int PPO_EPOCHS = 32;
static constexpr int HIDDEN_DIM = 256;
static constexpr int MINI_BATCH = 128;

// ======== DQN ==== //
static constexpr int WARMUP = ROLLOUT - 1;
static constexpr int DISC_STEPS = 150000;
static constexpr float GAMMA = 0.99f;
static constexpr int EQUALIZER = 15418;
static constexpr float INIT_EPSILON = 1.0f;

class Actor_Critic
{
private:
    Linear *A1, *A2;
    Linear *Q, *V;
    int action_dim, hidden;
public:
    GraphOperations &go;
    float s, v_loss, entropy;
    float pi_loss;
    Actor_Critic(GraphOperations& go, const graph& in_state, const int action_dim, const int hidden_dim) : go(go),
    action_dim(action_dim), hidden(hidden_dim)
    {
        A1 = new Linear(go, in_state->dim[3],hidden);
        A2 = new Linear(go, hidden, hidden);
        Q  = new Linear(go, hidden, action_dim, "Q(a|s)");  
        V  = new Linear(go, hidden, 1, "V(s)");  
    }
    
    void save(const str& filename) const
    {
        std::ofstream f(filename, std::ios::binary);
        if (!f) 
        {
            std::cerr << "Error opening file for saving: " << filename << std::endl;
            std::exit(1);
        }
        A1->save(f); A2->save(f); Q->save(f); V->save(f);f.close();
        std::cout << "Model saved successfully to " << filename << "\n";
    }

    void load(const str& filename)
    {
        std::ifstream f(filename, std::ios::binary);
        if (!f) {
            std::cerr << "Error opening file for loading: " << filename << std::endl;
            std::exit(1);
        }
        A1->load(f); A2->load(f); Q->load(f);  V->load(f); f.close();
        std::cout << "Model loaded successfully from " << filename << "\n";
    }

    std::pair<graph,graph> build_train(const graph& S_t)
    {
        auto f1 = go.SILU(A1->forward(S_t));
        auto f2 = go.SILU(A2->forward(f1));
        auto probs = go.SOFTMAX(Q->forward(f2)); probs->op_name = "π(a|s)";
        return {probs, V->forward(f2)};
    }

};

class DQN
{
private:
    Linear *A1, *A2, *Q;
    int action_dim, hidden;
public:
    GraphOperations &go;
    float s, v_loss, entropy;
    float pi_loss;
    DQN(GraphOperations& go, const graph& in_state, const int action_dim, const int hidden_dim) : go(go),
    action_dim(action_dim), hidden(hidden_dim)
    {
        A1 = new Linear(go, in_state->dim[3],hidden);
        A2 = new Linear(go, hidden, hidden);
        Q  = new Linear(go, hidden, action_dim, "Q_π(a|s)"); 
    }
    
    void save(const str& filename) const
    {
        std::ofstream f(filename, std::ios::binary);
        if (!f) 
        {
            std::cerr << "Error opening file for saving: " << filename << std::endl;
            std::exit(1);
        }
        A1->save(f); A2->save(f); Q->save(f); f.close();
        std::cout << "Model saved successfully to " << filename << "\n";
    }

    void load(const str& filename)
    {
        std::ifstream f(filename, std::ios::binary);
        if (!f) {
            std::cerr << "Error opening file for loading: " << filename << std::endl;
            std::exit(1);
        }
        A1->load(f); A2->load(f); Q->load(f); f.close();
        std::cout << "Model loaded successfully from " << filename << "\n";
    }

    graph build_train(const graph& S_t)
    {
        auto f1 = go.SILU(A1->forward(S_t));
        auto f2 = go.SILU(A2->forward(f1));
        auto value = Q->forward(f2);
        return value;
    }

    void operator = (const DQN& other)
    {
        *A1 = *other.A1; 
        *A2 = *other.A2; 
        *Q =  *other.Q; 
    }
};

struct Transition {float action, reward, log_prob, value, done;};
enum class AlgoMode { DQN, PPO };
enum class AppMode { MENU, PLAY, WATCH_TRAIN, WATCH_AGENT, BLANK_TRAIN };
SDL_Color C_BG        = {15,  17,  26,  255};
SDL_Color C_GRID      = {25,  28,  40,  255};
SDL_Color C_HEAD      = {80,  220, 120, 255};
SDL_Color C_BODY      = {40,  160,  80, 255};
SDL_Color C_FOOD      = {230,  80,  80, 255};
SDL_Color C_PANEL     = {20,  22,  35,  255};
SDL_Color C_TEXT      = {200, 205, 220, 255};
SDL_Color C_ACCENT    = {100, 160, 240, 255};
SDL_Color C_CHART_BG  = {12,  14,  22,  255};
SDL_Color C_CHART_LN  = {80,  200, 160, 255};
SDL_Color C_DISABLED  = {80,  85, 100,  255};
SDL_Color C_HOVER     = {130, 190, 255, 255};
SDL_Color C_WHITE     = {255, 255, 255, 255};
SDL_Color C_DIM       = {60,  65,  85,  255};

inline void setColor(SDL_Renderer* r, SDL_Color c) { SDL_SetRenderDrawColor(r, c.r, c.g, c.b, c.a); }
inline void fillRect(SDL_Renderer* r, int x, int y, int w, int h) { SDL_Rect rc{x,y,w,h}; SDL_RenderFillRect(r,&rc); }
inline void drawRect(SDL_Renderer* r, int x, int y, int w, int h) { SDL_Rect rc{x,y,w,h}; SDL_RenderDrawRect(r,&rc); }

void blitText(SDL_Renderer* r, TTF_Font* f, const std::string& s, SDL_Color c, int x, int y) 
{
    if (!f || s.empty()) return;
    SDL_Surface* surf = TTF_RenderText_Blended(f, s.c_str(), c);
    if (!surf) return;
    SDL_Texture* tex = SDL_CreateTextureFromSurface(r, surf);
    SDL_FreeSurface(surf);
    if (!tex) return;
    int w, h; SDL_QueryTexture(tex, nullptr, nullptr, &w, &h);
    SDL_Rect dst{x,y,w,h}; SDL_RenderCopy(r, tex, nullptr, &dst);
    SDL_DestroyTexture(tex);
}

void blitTextCentered(SDL_Renderer* r, TTF_Font* f, const std::string& s, SDL_Color c, int cx, int y) {
    if (!f || s.empty()) return;
    SDL_Surface* surf = TTF_RenderText_Blended(f, s.c_str(), c);
    if (!surf) return;
    SDL_Texture* tex = SDL_CreateTextureFromSurface(r, surf);
    SDL_FreeSurface(surf);
    if (!tex) return;
    int w, h; SDL_QueryTexture(tex, nullptr, nullptr, &w, &h);
    SDL_Rect dst{cx-w/2, y, w, h}; SDL_RenderCopy(r, tex, nullptr, &dst);
    SDL_DestroyTexture(tex);
}

TTF_Font* loadFont(int ptSize) 
{
    const char* paths[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        nullptr
    };
    for (int i = 0; paths[i]; ++i) { TTF_Font* f = TTF_OpenFont(paths[i], ptSize); if (f) return f; }
    return nullptr;
}

struct MenuItem { std::string label, sublabel; AppMode mode; bool enabled; };

AlgoMode runAlgoPicker(SDL_Renderer* ren, TTF_Font* fntMd, TTF_Font* fntSm)
{
    struct AlgoItem { std::string label, sublabel; AlgoMode mode; };
    std::vector<AlgoItem> items = {
        {"DQN",  "Deep Q-Network (experience replay + target net)", AlgoMode::DQN},
        {"PPO",  "Proximal Policy Optimisation (actor-critic)",      AlgoMode::PPO},
    };

    int hovered = 0;
    bool quit = false;
    AlgoMode chosen = AlgoMode::DQN;

    while (!quit)
    {
        SDL_Event ev;
        while (SDL_PollEvent(&ev))
        {
            if (ev.type == SDL_QUIT) { quit = true; break; }
            if (ev.type == SDL_KEYDOWN) switch (ev.key.keysym.sym) {
                case SDLK_ESCAPE: quit = true; break;
                case SDLK_UP:
                    hovered = (hovered - 1 + (int)items.size()) % (int)items.size(); break;
                case SDLK_DOWN:
                    hovered = (hovered + 1) % (int)items.size(); break;
                case SDLK_RETURN: case SDLK_SPACE:
                    chosen = items[hovered].mode; quit = true; break;
            }
            if (ev.type == SDL_MOUSEMOTION)
                for (int i = 0; i < (int)items.size(); ++i) {
                    int iy = WIN_H/2 - 10 + i * 80;
                    if (ev.motion.y >= iy - 10 && ev.motion.y < iy + 60) hovered = i;
                }
            if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_LEFT)
                for (int i = 0; i < (int)items.size(); ++i) {
                    int iy = WIN_H/2 - 10 + i * 80;
                    if (ev.button.y >= iy - 10 && ev.button.y < iy + 60)
                        { chosen = items[i].mode; quit = true; }
                }
        }

        setColor(ren, C_BG); fillRect(ren, 0, 0, WIN_W, WIN_H);
        setColor(ren, C_GRID);
        for (int x = 0; x <= WIN_W; x += CELL) SDL_RenderDrawLine(ren, x, 0, x, WIN_H);
        for (int y = 0; y <= WIN_H; y += CELL) SDL_RenderDrawLine(ren, 0, y, WIN_W, y);

        blitTextCentered(ren, fntMd, "SELECT ALGORITHM", C_ACCENT, WIN_W/2, WIN_H/2 - 110);
        setColor(ren, C_ACCENT);
        SDL_RenderDrawLine(ren, WIN_W/2-120, WIN_H/2-82, WIN_W/2+120, WIN_H/2-82);

        for (int i = 0; i < (int)items.size(); ++i) {
            int iy = WIN_H/2 - 10 + i * 80;
            bool isHov = (i == hovered);
            if (isHov) {
                setColor(ren, C_DIM);   fillRect(ren, WIN_W/2-200, iy-8, 400, 54);
                setColor(ren, C_ACCENT); drawRect(ren, WIN_W/2-200, iy-8, 400, 54);
            }
            SDL_Color lc = isHov ? C_HOVER : C_TEXT;
            SDL_Color sc = C_ACCENT;
            blitTextCentered(ren, fntMd, items[i].label,    lc, WIN_W/2, iy);
            blitTextCentered(ren, fntSm, items[i].sublabel, sc, WIN_W/2, iy + 22);
        }
        blitTextCentered(ren, fntSm, "Arrow keys / Enter to choose   Esc = back",
                         C_DISABLED, WIN_W/2, WIN_H - 30);
        SDL_RenderPresent(ren);
        SDL_Delay(16);
    }
    return chosen;
}

AppMode runMenu(SDL_Renderer* ren, TTF_Font* fntLg, TTF_Font* fntMd, TTF_Font* fntSm)
{
    std::vector<MenuItem> items = {
        {"PLAY",               "Control the snake yourself",        AppMode::PLAY,        true },
        {"TRAIN (HEADLESS)",   "Max-speed training, no render",     AppMode::BLANK_TRAIN, true },
        {"WATCH AGENT TRAIN",  "See the agent learning in real-time", AppMode::WATCH_TRAIN, true },
        {"WATCH TRAINED AGENT","Load weights and watch it play",    AppMode::WATCH_AGENT, true },
    };

    int hovered = 0; bool quit = false; AppMode chosen = AppMode::MENU;

    while (!quit) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) { quit = true; break; }
            if (ev.type == SDL_KEYDOWN) {
                switch (ev.key.keysym.sym) {
                    case SDLK_ESCAPE: quit = true; break;
                    case SDLK_UP:
                        do { hovered = (hovered-1+(int)items.size())%(int)items.size(); } while (!items[hovered].enabled);
                        break;
                    case SDLK_DOWN:
                        do { hovered = (hovered+1)%(int)items.size(); } while (!items[hovered].enabled);
                        break;
                    case SDLK_RETURN: case SDLK_SPACE:
                        if (items[hovered].enabled) { chosen = items[hovered].mode; quit = true; }
                        break;
                }
            }
            if (ev.type == SDL_MOUSEMOTION) {
                for (int i = 0; i < (int)items.size(); ++i) {
                    int iy = WIN_H/2 - 20 + i*80;
                    if (ev.motion.y >= iy-10 && ev.motion.y < iy+60 && items[i].enabled) hovered = i;
                }
            }
            if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_LEFT) {
                for (int i = 0; i < (int)items.size(); ++i) {
                    int iy = WIN_H/2 - 20 + i*80;
                    if (ev.button.y >= iy-10 && ev.button.y < iy+60 && items[i].enabled)
                        { chosen = items[i].mode; quit = true; }
                }
            }
        }

        setColor(ren, C_BG); fillRect(ren, 0, 0, WIN_W, WIN_H);
        setColor(ren, C_GRID);
        for (int x = 0; x <= WIN_W; x += CELL) SDL_RenderDrawLine(ren, x, 0, x, WIN_H);
        for (int y = 0; y <= WIN_H; y += CELL) SDL_RenderDrawLine(ren, 0, y, WIN_W, y);

        blitTextCentered(ren, fntLg, "SNAKE",              C_HEAD,   WIN_W/2, WIN_H/2-160);
        blitTextCentered(ren, fntSm, "DQN / PPO EDITION",  C_ACCENT, WIN_W/2, WIN_H/2-120);
        setColor(ren, C_ACCENT);
        SDL_RenderDrawLine(ren, WIN_W/2-100, WIN_H/2-96, WIN_W/2+100, WIN_H/2-96);

        for (int i = 0; i < (int)items.size(); ++i) {
            int iy = WIN_H/2 - 20 + i*80;
            bool isHov = (i == hovered) && items[i].enabled;
            if (isHov) {
                setColor(ren, C_DIM);   fillRect(ren, WIN_W/2-200, iy-8, 400, 54);
                setColor(ren, C_ACCENT); drawRect(ren, WIN_W/2-200, iy-8, 400, 54);
            }
            SDL_Color lc = !items[i].enabled ? C_DISABLED : (isHov ? C_HOVER : C_TEXT);
            SDL_Color sc = !items[i].enabled ? C_DISABLED : C_ACCENT;
            blitTextCentered(ren, fntMd, items[i].label,    lc, WIN_W/2, iy);
            blitTextCentered(ren, fntSm, items[i].sublabel, sc, WIN_W/2, iy+22);
            if (!items[i].enabled)
                blitTextCentered(ren, fntSm, "(coming soon)", C_DISABLED, WIN_W/2, iy+38);
        }

        blitTextCentered(ren, fntSm, "Arrow keys to navigate   Enter to select   Esc to quit",
                         C_DISABLED, WIN_W/2, WIN_H-30);
        SDL_RenderPresent(ren);
        SDL_Delay(16);
    }
    return chosen;
}

void drawGameArea(SDL_Renderer* ren, TTF_Font* fnt, const SnakeEnv& env,
                  int /*episode*/, int /*step*/, float /*lastReward*/,
                  bool paused, bool dead)
{
    setColor(ren, C_BG); fillRect(ren, 0, 0, GAME_W, GAME_H);
    setColor(ren, C_GRID);
    for (int x = 0; x <= GRID_W; ++x) SDL_RenderDrawLine(ren, x*CELL, 0,     x*CELL, GAME_H);
    for (int y = 0; y <= GRID_H; ++y) SDL_RenderDrawLine(ren, 0,      y*CELL, GAME_W, y*CELL);

    Point f = env.food();
    setColor(ren, C_FOOD); fillRect(ren, f.x*CELL+3, f.y*CELL+3, CELL-6, CELL-6);

    const auto& sn = env.snake();
    for (size_t i = 1; i < sn.size(); ++i) {
        setColor(ren, C_BODY); fillRect(ren, sn[i].x*CELL+1, sn[i].y*CELL+1, CELL-2, CELL-2);
    }
    if (!sn.empty()) {
        SDL_Color hc = dead ? SDL_Color{180,60,60,255} : C_HEAD;
        setColor(ren, hc); fillRect(ren, sn[0].x*CELL+1, sn[0].y*CELL+1, CELL-2, CELL-2);
        setColor(ren, C_BG);
        fillRect(ren, sn[0].x*CELL+5,       sn[0].y*CELL+5, 4, 4);
        fillRect(ren, sn[0].x*CELL+CELL-9,  sn[0].y*CELL+5, 4, 4);
    }
    if (paused) blitTextCentered(ren, fnt, "PAUSED  (P to resume)",        C_ACCENT,              GAME_W/2, GAME_H/2-10);
    if (dead)   blitTextCentered(ren, fnt, "GAME OVER  (R to restart)",    {230,80,80,255},        GAME_W/2, GAME_H/2-10);
}

void drawPanel(SDL_Renderer* ren, TTF_Font* fnt, TTF_Font* fntSm, const std::string& modeLabel, int episode, long long totalSteps,
               float piLoss, float vLoss, float entropy, float avgScore, const std::deque<float>& scoreHist, float stepsPerSec, int curScore)
{
    int px = GAME_W;
    setColor(ren, C_PANEL);  fillRect(ren, px, 0, PANEL_W, WIN_H);
    setColor(ren, C_ACCENT); fillRect(ren, px, 0, 2, WIN_H);

    int y = 14;
    auto line   = [&](const std::string& s, SDL_Color c = C_TEXT) { blitText(ren, fnt,   s, c, px+12, y); y+=22; };
    auto lineSm = [&](const std::string& s, SDL_Color c = C_TEXT) { blitText(ren, fntSm, s, c, px+12, y); y+=18; };

    line("── " + modeLabel + " ──", C_ACCENT); y += 6;
    std::ostringstream ss;
    ss << "Episode     " << episode;      line(ss.str()); ss.str("");
    ss << "Total Steps " << totalSteps;   line(ss.str()); ss.str("");
    ss << "Steps/sec   " << (int)stepsPerSec; line(ss.str()); ss.str("");
    ss << "Score       " << curScore;     line(ss.str()); ss.str("");
    y += 6;

    line("── Policy ──", C_ACCENT);
    if (piLoss != 0.f || vLoss != 0.f) {
        ss << std::fixed << std::setprecision(4);
        ss << "Pi Loss    " << piLoss;  line(ss.str()); ss.str("");
        ss << "Val Loss   " << vLoss;   line(ss.str()); ss.str("");
        ss << "Entropy    " << entropy; line(ss.str()); ss.str("");
    } else {
        lineSm("(training not started)", C_DISABLED);
    }
    y += 6;

    line("── Score ──", C_ACCENT);
    ss << std::fixed << std::setprecision(2);
    ss << "Avg(last 20)  " << avgScore; line(ss.str()); ss.str("");
    if (!scoreHist.empty()) {
        ss << "Best          " << *std::max_element(scoreHist.begin(), scoreHist.end());
        line(ss.str()); ss.str("");
    }
    y += 10;

    int cw = PANEL_W-24, ch = 100;
    setColor(ren, C_CHART_BG); fillRect(ren, px+12, y, cw, ch);
    setColor(ren, C_ACCENT);   drawRect(ren, px+12, y, cw, ch);
    blitText(ren, fntSm, "Score History", C_TEXT, px+12, y+2);

    if (scoreHist.size() > 1) {
        float mx = *std::max_element(scoreHist.begin(), scoreHist.end());
        float mn = *std::min_element(scoreHist.begin(), scoreHist.end());
        if (mx == mn) mx = mn+1;
        int N = (int)scoreHist.size();
        float xStep = (float)cw / (N-1);
        setColor(ren, C_CHART_LN);
        for (int i = 1; i < N; ++i) {
            int x1 = px+12+(int)((i-1)*xStep), x2 = px+12+(int)(i*xStep);
            int y1 = y+ch-(int)((scoreHist[i-1]-mn)/(mx-mn)*(ch-14))-4;
            int y2 = y+ch-(int)((scoreHist[i  ]-mn)/(mx-mn)*(ch-14))-4;
            SDL_RenderDrawLine(ren, x1, y1, x2, y2);
        }
    }
    y += ch+10;
    lineSm("M  = main menu", C_DISABLED);
    lineSm("Q  = quit",      C_DISABLED);
}

bool runPlay(SDL_Renderer* ren, TTF_Font* fnt, TTF_Font* fntSm) {
    SnakeEnv env; std::deque<float> scoreHist;
    env.reset();
    Action pendingAction = Action::RIGHT;
    bool paused=false, dead=false, quit=false, toMenu=false;
    int ep=0; long long totalSteps=0; int epStep=0;
    auto t0=std::chrono::steady_clock::now(), tFPS=t0, tLastStep=t0;
    int frmCnt=0; float sps=0;
    const int STEP_MS = 130;

    while (!quit && !toMenu) {
        auto now = std::chrono::steady_clock::now();
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type==SDL_QUIT) { quit=true; break; }
            if (ev.type==SDL_KEYDOWN) switch(ev.key.keysym.sym) {
                case SDLK_ESCAPE: case SDLK_q: quit=true;         break;
                case SDLK_m:      toMenu=true;                     break;
                case SDLK_p:      paused=!paused;                  break;
                case SDLK_r:
                    if (dead) { env.reset(); pendingAction=Action::RIGHT; dead=paused=false; epStep=0; }
                    break;
                case SDLK_UP:    case SDLK_w: pendingAction=Action::UP;    break;
                case SDLK_DOWN:  case SDLK_s: pendingAction=Action::DOWN;  break;
                case SDLK_LEFT:  case SDLK_a: pendingAction=Action::LEFT;  break;
                case SDLK_RIGHT: case SDLK_d: pendingAction=Action::RIGHT; break;
            }
        }
        auto msStep = std::chrono::duration_cast<std::chrono::milliseconds>(now-tLastStep).count();
        if (!paused && !dead && msStep >= STEP_MS) {
            tLastStep=now;
            auto res=env.Obstep(pendingAction);
            ++totalSteps; ++epStep; ++frmCnt;
            if (res.done) { scoreHist.push_back((float)res.score); if ((int)scoreHist.size()>MAX_SCORE_HIST) scoreHist.pop_front(); ++ep; dead=true; }
        }
        { double el=std::chrono::duration<double>(now-tFPS).count(); if(el>=1.0){sps=frmCnt/(float)el;frmCnt=0;tFPS=now;} }

        setColor(ren,{0,0,0,255}); SDL_RenderClear(ren);
        drawGameArea(ren,fnt,env,ep,epStep,0.f,paused,dead);
        blitText(ren,fntSm,"WASD/Arrows=move  P=pause  R=restart  M=menu",C_DISABLED,4,GAME_H-16);

        float avg=0.f; { int n=std::min((int)scoreHist.size(),20); for(int i=(int)scoreHist.size()-n;i<(int)scoreHist.size();++i) avg+=scoreHist[i]; avg/=std::max(n,1); }
        drawPanel(ren,fnt,fntSm,"HUMAN PLAY",ep,totalSteps,0.f,0.f,0.f,avg,scoreHist,sps,env.score());
        SDL_RenderPresent(ren); SDL_Delay(8);
    }
    return toMenu;
}

bool runWatchAgent_DQN(SDL_Renderer* ren, TTF_Font* fnt, TTF_Font* fntSm)
{
    GraphOperations go;
    SnakeEnv env;
    std::deque<float> scoreHist;
    env.reset();
    auto obs  = env.getObs();
    auto STATE = std::make_shared<NodeBackProp>("State", 1, 1, 1, obs.total, 1);
    DQN agent(go, STATE, ACT_DIM, HIDDEN_DIM);

    try {
        std::cout << "Attempting to load DQN model\nPath: ";
        str path; std::getline(std::cin, path);
        agent.load(path);
    } catch (const std::exception& e) {
        std::cerr << "runWatchAgent_DQN: " << e.what() << " – running with random weights\n";
    }

    int* argmaxindex;
    SafeCudaMalloc("Argmax", argmaxindex, 1);
    auto logits = agent.build_train(STATE);

    int ep = 0; long long totalSteps = 0;
    bool quit = false, toMenu = false;

    auto tFPS = std::chrono::steady_clock::now(), tLastStep = tFPS;
    int frmCnt = 0; float sps = 0.f;
    const int STEP_MS = 80;

    while (!quit && !toMenu)
    {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) { quit = true; break; }
            if (ev.type == SDL_KEYDOWN) switch (ev.key.keysym.sym) {
                case SDLK_ESCAPE: case SDLK_q: quit = true;  break;
                case SDLK_m:      toMenu = true;              break;
            }
        }
        if (quit || toMenu) break;

        auto now   = std::chrono::steady_clock::now();
        auto msStep = std::chrono::duration_cast<std::chrono::milliseconds>(now - tLastStep).count();

        if (msStep >= STEP_MS)
        {
            tLastStep = now;
            go.forward(logits);
            auto res = env.Obstep(static_cast<Action>(ArgMaxToCPU(logits, argmaxindex)));
            obs = res.obs;
            ++totalSteps; ++frmCnt;

            if (res.done)
            {
                scoreHist.push_back((float)res.score);
                if ((int)scoreHist.size() > MAX_SCORE_HIST) scoreHist.pop_front();
                ++ep;
                env.reset();
                obs = env.getObs();
            }
            cudaMemcpy(STATE->output, obs.grid.data(), obs.total*sizeof(float), cudaMemcpyHostToDevice);
        }

        { double el = std::chrono::duration<double>(now - tFPS).count(); if (el >= 1.0) { sps = (float)(frmCnt / el); frmCnt = 0; tFPS = now; } }
        float avg = 0.f; { int n = std::min((int)scoreHist.size(), 20); for (int i = (int)scoreHist.size() - n; i < (int)scoreHist.size(); ++i) avg += scoreHist[i]; avg /= std::max(n, 1); }

        setColor(ren, {0, 0, 0, 255}); SDL_RenderClear(ren);
        drawGameArea(ren, fnt, env, ep, 0, 0.f, false, false);
        blitText(ren, fntSm, "Watching trained DQN agent   M=menu   Q=quit", C_DISABLED, 4, GAME_H - 16);
        drawPanel(ren, fnt, fntSm, "WATCH AGENT (DQN)", ep, totalSteps, 0.f, 0.f, 0.f, avg, scoreHist, sps, env.score());
        SDL_RenderPresent(ren);
        SDL_Delay(4);
    }

    go.clear_graph(); cudaFree(argmaxindex);
    return toMenu;
}

bool runWatchAgent_PPO(SDL_Renderer* ren, TTF_Font* fnt, TTF_Font* fntSm)
{
    SnakeEnv env;
    std::deque<float> scoreHist;
    env.reset();
    auto obs  = env.getObs();
    auto STATE = std::make_shared<NodeBackProp>("State", 1, 1, 1, obs.total, 1);
    GraphOperations go;
    RL_Replay       replay(ROLLOUT, obs.total);
    Actor_Critic    agent(go, STATE, ACT_DIM, HIDDEN_DIM);

    try 
    {
        std::cout << "Attempting to load PPO model\nPath: ";
        str path; std::getline(std::cin, path);
        agent.load(path);
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "runWatchAgent_PPO: " << e.what() << " – running with random weights\n";
    }

    auto [A_prob_state, V_state] = agent.build_train(STATE);
    go.nodes = topological_sort(go.track({A_prob_state, V_state}));

    int* argmaxindex;
    SafeCudaMalloc("Argmax", argmaxindex, 1);
    cudaMemcpy(STATE->output, obs.grid.data(), obs.total*sizeof(float), cudaMemcpyHostToDevice);


    int ep = 0; long long totalSteps = 0;
    bool quit = false, toMenu = false;

    auto tFPS = std::chrono::steady_clock::now(), tLastStep = tFPS;
    int frmCnt = 0; float sps = 0.f;
    const int STEP_MS = 80;

    while (!quit && !toMenu)
    {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) { quit = true; break; }
            if (ev.type == SDL_KEYDOWN) switch (ev.key.keysym.sym) {
                case SDLK_ESCAPE: case SDLK_q: quit = true;  break;
                case SDLK_m:      toMenu = true;              break;
            }
        }
        if (quit || toMenu) break;

        auto now    = std::chrono::steady_clock::now();
        auto msStep = std::chrono::duration_cast<std::chrono::milliseconds>(now - tLastStep).count();

        if (msStep >= STEP_MS)
        {
            tLastStep = now;
            go.forward();
            auto res = env.Obstep(static_cast<Action>(ArgMaxToCPU(A_prob_state, argmaxindex)));
            obs = res.obs;
            ++totalSteps; ++frmCnt;

            if (res.done)
            {
                scoreHist.push_back((float)res.score);
                if ((int)scoreHist.size() > MAX_SCORE_HIST) scoreHist.pop_front();
                ++ep;
                env.reset();
                obs = env.getObs();
            }
            cudaMemcpy(STATE->output, obs.grid.data(), obs.total*sizeof(float), cudaMemcpyHostToDevice);
        }

        { auto now2 = std::chrono::steady_clock::now(); double el = std::chrono::duration<double>(now2 - tFPS).count(); if (el >= 1.0) { sps = (float)(frmCnt / el); frmCnt = 0; tFPS = now2; } }
        float avg = 0.f; { int n = std::min((int)scoreHist.size(), 20); for (int i = (int)scoreHist.size() - n; i < (int)scoreHist.size(); ++i) avg += scoreHist[i]; avg /= std::max(n, 1); }

        setColor(ren, {0, 0, 0, 255}); SDL_RenderClear(ren);
        drawGameArea(ren, fnt, env, ep, 0, 0.f, false, false);
        blitText(ren, fntSm, "Watching trained PPO agent   M=menu   Q=quit", C_DISABLED, 4, GAME_H - 16);
        drawPanel(ren, fnt, fntSm, "WATCH AGENT (PPO)", ep, totalSteps, 0.f, 0.f, 0.f, avg, scoreHist, sps, env.score());
        SDL_RenderPresent(ren);
        SDL_Delay(4);
    }

    STATE->clear(); cudaFree(argmaxindex);
    return toMenu;
}


bool runWatchTrain_DQN(SDL_Renderer* ren, TTF_Font* fnt, TTF_Font* fntSm)
{
    GraphOperations go;
    SnakeEnv env; env.reset();
    std::deque<float> scoreHist;
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution real_dist(0.0, 1.0);
    std::uniform_int_distribution action_dist(0,ACT_DIM-1);

    auto obs = env.getObs();

    RL_Replay replay(ROLLOUT,obs.total,true);
    Transition seq;

    auto state          = std::make_shared<NodeBackProp>("Current State", 1,1,1,obs.total,1);
    auto next_state     = std::make_shared<NodeBackProp>("Next State",state->dim[0],state->dim[1],state->dim[2],state->dim[3],1);
    auto train_state    = std::make_shared<NodeBackProp>("Train state",MINI_BATCH,1,1,obs.total,1);
    auto fill_next_train= std::make_shared<NodeBackProp>("FULL next state",ROLLOUT,1,1,obs.total,1);

    DQN Q(go, state, ACT_DIM, HIDDEN_DIM), Qmax(go, next_state, ACT_DIM, HIDDEN_DIM);
    DQNTrainer<DQN> trainer(go, Q, Qmax, replay, train_state->dim[0], PPO_EPOCHS);

    auto logits      = Q.build_train(state);
    auto next_logits = Qmax.build_train(next_state);
    float epsilon    = (INIT_EPSILON < 1.0f) ? INIT_EPSILON : 1.0f;

    int   *argmaxindex;
    float *action_idx, *target;
    SafeCudaMalloc("Argmax Ptr", argmaxindex, 1);
    SafeCudaMalloc("Action_idx", action_idx, ROLLOUT);
    SafeCudaMalloc("Target",     target,     ROLLOUT);

    int       ep         = 0;
    long long totalSteps = 0;
    int       bufSteps   = 0;

    bool quit=false, toMenu=false;
    const int RENDER_EVERY = 2;

    auto tFPS = std::chrono::steady_clock::now();
    int  frmCnt = 0; float sps = 0.f;
    float piLoss=0.f, vLoss=0.f, entropy=0.f;

    while (!quit && !toMenu)
    {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type==SDL_QUIT) { quit=true; break; }
            if (ev.type==SDL_KEYDOWN) switch(ev.key.keysym.sym) {
                case SDLK_ESCAPE: case SDLK_q: quit=true;  break;
                case SDLK_m:      toMenu=true;              break;
                case SDLK_s:      Q.save("snake_conv.bin"); break;
            }
        }
        if (quit||toMenu) break;

        epsilon = std::max(0.05, INIT_EPSILON - (totalSteps / (double)DISC_STEPS));
        // State copy to tensor and buffer history //
        cudaMemcpy(state->output, obs.grid.data(), obs.total*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(replay.state + bufSteps * obs.total, state->output, obs.total * sizeof(float), cudaMemcpyDeviceToDevice);
        // ====================================== //

        int act;
        if (real_dist(gen) < epsilon || totalSteps < WARMUP) act = action_dist(gen);
        else { go.forward(logits); act = ArgMaxToCPU(logits, argmaxindex); }

        auto res = env.Obstep(static_cast<Action>(act));
        seq.action = act; seq.reward = res.reward; seq.log_prob = 0.0f; seq.done = res.done;

        // Next state copy to tensor and buffer history for fill Kernel when revaluating after Qmax = Q update //
        cudaMemcpy(next_state->output, res.obs.grid.data(), res.obs.total*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(replay.next_state + bufSteps * res.obs.total, next_state->output, res.obs.total * sizeof(float), cudaMemcpyDeviceToDevice);
        // ====================================== //

        go.forward(next_logits);
        float targ_id = ArgMaxToCPU(next_logits, argmaxindex);
        seq.value = res.reward + (float)(1 - (int)res.done)*GAMMA * ReadValueAt(next_logits, targ_id);

        // Important values for Computation in DQN Assign copy //
        cudaMemcpy(replay.traj + 5*bufSteps, &seq, sizeof(Transition), cudaMemcpyHostToDevice);
        // ================================================== //

        if (++bufSteps == ROLLOUT && totalSteps >= WARMUP)
        {
            go.clear_graph(logits);
            trainer.update(train_state, target, action_idx);
            logits = Q.build_train(state);
            if ((totalSteps / ROLLOUT) % EQUALIZER == 0)
            {
                Qmax = Q;
                go.clear_graph(next_logits);
                auto full = Qmax.build_train(fill_next_train);
                cudaMemcpy(fill_next_train->output, replay.next_state, ROLLOUT*obs.total*sizeof(float), cudaMemcpyDeviceToDevice);
                go.forward(full);
                replace_targets<<<(ROLLOUT+THREADSPERBLOCK-1)/THREADSPERBLOCK, THREADSPERBLOCK>>>(full->output, replay.traj, ROLLOUT, ACT_DIM, GAMMA);
                next_logits = Qmax.build_train(next_state);
                CheckError("Buffer Replacement");
            }
        }
        
        if(bufSteps == ROLLOUT) bufSteps = 0;

        ++totalSteps; ++frmCnt;
        obs = res.obs;

        if (res.done)
        {
            scoreHist.push_back((float)res.score);
            if ((int)scoreHist.size() > MAX_SCORE_HIST) scoreHist.pop_front();
            ++ep;
            env.reset();
            obs = env.getObs();
        }

        if (frmCnt % RENDER_EVERY == 0) {
            auto now = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(now-tFPS).count();
            if (el >= 1.0) { sps = (float)(frmCnt/el); frmCnt = 0; tFPS = now; }
            float avg=0.f; { int n=std::min((int)scoreHist.size(),20); for(int i=(int)scoreHist.size()-n;i<(int)scoreHist.size();++i) avg+=scoreHist[i]; avg/=std::max(n,1); }
            setColor(ren,{0,0,0,255}); SDL_RenderClear(ren);
            drawGameArea(ren,fnt,env,ep,0,0.f,false,false);
            blitText(ren,fntSm,"S=save weights   M=menu   Q=quit",C_DISABLED,4,GAME_H-16);
            drawPanel(ren,fnt,fntSm,"WATCH TRAIN (DQN)",ep,totalSteps,piLoss,vLoss,entropy,avg,scoreHist,sps,env.score());
            SDL_RenderPresent(ren);
        }
    }

    fill_next_train->clear();
    state->clear(); next_state->clear(); train_state->clear();
    go.clear_graph(logits); go.clean_clear_graph(next_logits);
    cudaFree(argmaxindex); cudaFree(target); cudaFree(action_idx);
    return toMenu;
}

bool runBlankTrain_DQN(SDL_Renderer* ren, TTF_Font* fntMd, TTF_Font* fntSm)
{
    GraphOperations go;
    SnakeEnv env; env.reset();
    std::deque<float> scoreHist;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution real_dist(0.0, 1.0);
    std::uniform_int_distribution action_dist(0,ACT_DIM-1);
    auto obs = env.getObs();

    RL_Replay replay(ROLLOUT,obs.total,true);
    Transition seq;

    auto state           = std::make_shared<NodeBackProp>("Current State", 1,1,1,obs.total,1);
    auto next_state      = std::make_shared<NodeBackProp>("Next State",state->dim[0],state->dim[1],state->dim[2],state->dim[3],1);
    auto train_state     = std::make_shared<NodeBackProp>("Train state",MINI_BATCH,1,1,obs.total,1);
    auto fill_next_train = std::make_shared<NodeBackProp>("FULL next state",ROLLOUT,1,1,obs.total,1);

    DQN Q(go, state, ACT_DIM, HIDDEN_DIM), Qmax(go, next_state, ACT_DIM, HIDDEN_DIM);
    DQNTrainer<DQN> trainer(go, Q, Qmax, replay, train_state->dim[0], PPO_EPOCHS);

    auto logits      = Q.build_train(state);
    auto next_logits = Qmax.build_train(next_state);
    float epsilon    = (INIT_EPSILON < 1.0f) ? INIT_EPSILON : 1.0f;

    int   *argmaxindex;
    float *action_idx, *target;
    SafeCudaMalloc("Argmax Ptr", argmaxindex, 1);
    SafeCudaMalloc("Action_idx", action_idx, ROLLOUT);
    SafeCudaMalloc("Target",     target,     ROLLOUT);

    int       ep         = 0;
    long long totalSteps = 0;
    int       bufSteps   = 0;
    auto t0      = std::chrono::steady_clock::now();
    auto tRender = t0;
    const int RENDER_MS = 500;
    bool quit = false, toMenu = false;

    auto renderStatus = [&]() {
        float avg = 0.f;
        { int n = std::min((int)scoreHist.size(), 20); for (int i = (int)scoreHist.size() - n; i < (int)scoreHist.size(); ++i) avg += scoreHist[i]; avg /= std::max(n, 1); }
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        float  sps = (elapsed > 0) ? (float)(totalSteps / elapsed) : 0.f;

        setColor(ren, C_BG); SDL_RenderClear(ren); fillRect(ren, 0, 0, WIN_W, WIN_H);
        setColor(ren, C_GRID);
        for (int x = 0; x <= WIN_W; x += CELL) SDL_RenderDrawLine(ren, x, 0, x, WIN_H);
        for (int y = 0; y <= WIN_H; y += CELL) SDL_RenderDrawLine(ren, 0, y, WIN_W, y);

        int cx = WIN_W / 2, y = WIN_H / 2 - 140;
        blitTextCentered(ren, fntMd, "HEADLESS TRAINING  [DQN]", C_ACCENT, cx, y); y += 34;
        setColor(ren, C_ACCENT); SDL_RenderDrawLine(ren, cx - 140, y, cx + 140, y); y += 16;

        std::ostringstream ss;
        ss << "Episodes    " << ep;          blitTextCentered(ren, fntSm, ss.str(), C_TEXT,   cx, y); y += 22; ss.str("");
        ss << "Total Steps " << totalSteps;  blitTextCentered(ren, fntSm, ss.str(), C_TEXT,   cx, y); y += 22; ss.str("");
        ss << "Steps / sec " << (int)sps;    blitTextCentered(ren, fntSm, ss.str(), C_ACCENT, cx, y); y += 34; ss.str("");

        blitTextCentered(ren, fntSm, "── Policy ──", C_ACCENT, cx, y); y += 22;
        ss << std::fixed << std::setprecision(4);
        ss << "MSE Loss  " << Q.pi_loss; blitTextCentered(ren, fntSm, ss.str(), C_TEXT, cx, y); y += 22; ss.str("");

        blitTextCentered(ren, fntSm, "── Score ──", C_ACCENT, cx, y); y += 22;
        ss << std::fixed << std::setprecision(2);
        ss << "Avg (last 20)  " << avg; blitTextCentered(ren, fntSm, ss.str(), C_TEXT, cx, y); y += 22; ss.str("");
        if (!scoreHist.empty()) {
            ss << "Best           " << *std::max_element(scoreHist.begin(), scoreHist.end());
            blitTextCentered(ren, fntSm, ss.str(), C_HEAD, cx, y); y += 34; ss.str("");
        }

        int cw = 400, ch = 80, chartX = cx - cw / 2;
        setColor(ren, C_CHART_BG); fillRect(ren, chartX, y, cw, ch);
        setColor(ren, C_ACCENT);   drawRect(ren, chartX, y, cw, ch);
        blitText(ren, fntSm, "Score History", C_TEXT, chartX + 4, y + 2);
        if (scoreHist.size() > 1) {
            float mx = *std::max_element(scoreHist.begin(), scoreHist.end());
            float mn = *std::min_element(scoreHist.begin(), scoreHist.end());
            if (mx == mn) mx = mn + 1;
            int N = (int)scoreHist.size(); float xStep = (float)cw / (N - 1);
            setColor(ren, C_CHART_LN);
            for (int i = 1; i < N; ++i) {
                int x1 = chartX + (int)((i-1)*xStep), x2 = chartX + (int)(i*xStep);
                int y1 = y + ch - (int)((scoreHist[i-1] - mn) / (mx - mn) * (ch - 14)) - 4;
                int y2 = y + ch - (int)((scoreHist[i]   - mn) / (mx - mn) * (ch - 14)) - 4;
                SDL_RenderDrawLine(ren, x1, y1, x2, y2);
            }
        }
        y += ch + 16;
        blitTextCentered(ren, fntSm, "S=save   M=main menu   Q=quit", C_DISABLED, cx, y);
        SDL_RenderPresent(ren);
    };

    renderStatus();

    while (!quit && !toMenu)
    {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) { quit = true; break; }
            if (ev.type == SDL_KEYDOWN) switch (ev.key.keysym.sym) {
                case SDLK_ESCAPE: case SDLK_q: quit = true; break;
                case SDLK_m:      toMenu = true;             break;
                case SDLK_s:
                    Q.save("../snake_conv.bin");
                    std::printf("[save] weights written to snake_conv.bin\n");
                    break;
            }
        }
        if (quit || toMenu) break;
        epsilon = std::max(0.05, INIT_EPSILON - (totalSteps / (double)DISC_STEPS));
        // State copy to tensor and buffer history //
        cudaMemcpy(state->output, obs.grid.data(), obs.total*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(replay.state + bufSteps * obs.total, state->output, obs.total * sizeof(float), cudaMemcpyDeviceToDevice);
        // ====================================== //

        int act;
        if (real_dist(gen) < epsilon || totalSteps < WARMUP) act = action_dist(gen);
        else { go.forward(logits); act = ArgMaxToCPU(logits, argmaxindex); }

        auto res = env.Obstep(static_cast<Action>(act));
        seq.action = act; seq.reward = res.reward; seq.log_prob = 0.0f; seq.done = res.done;

        // Next state copy to tensor and buffer history for fill Kernel when revaluating after Qmax = Q update //
        cudaMemcpy(next_state->output, res.obs.grid.data(), res.obs.total*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(replay.next_state + bufSteps * res.obs.total, next_state->output, res.obs.total * sizeof(float), cudaMemcpyDeviceToDevice);
        // ====================================== //

        go.forward(next_logits);
        float targ_id = ArgMaxToCPU(next_logits, argmaxindex);
        seq.value = res.reward + (float)(1 - (int)res.done)*GAMMA * ReadValueAt(next_logits, targ_id);

        // Important values for Computation in DQN Assign copy //
        cudaMemcpy(replay.traj + 5*bufSteps, &seq, sizeof(Transition), cudaMemcpyHostToDevice);
        // ================================================== //

        if (++bufSteps == ROLLOUT && totalSteps >= WARMUP)
        {
            go.clear_graph(logits);
            trainer.update(train_state, target, action_idx);
            logits = Q.build_train(state);
            if ((totalSteps / ROLLOUT) % EQUALIZER == 0)
            {
                Qmax = Q;
                go.clear_graph(next_logits);
                auto full = Qmax.build_train(fill_next_train);
                cudaMemcpy(fill_next_train->output, replay.next_state, ROLLOUT*obs.total*sizeof(float), cudaMemcpyDeviceToDevice);
                go.forward(full);
                replace_targets<<<(ROLLOUT+THREADSPERBLOCK-1)/THREADSPERBLOCK, THREADSPERBLOCK>>>(full->output, replay.traj, ROLLOUT, ACT_DIM, GAMMA);
                next_logits = Qmax.build_train(next_state);
                CheckError("Buffer Replacement");
            }
        }
        
        if(bufSteps == ROLLOUT) bufSteps = 0;

        ++totalSteps;
        obs = res.obs;
        
        if (res.done) {
            scoreHist.push_back((float)res.score);
            if ((int)scoreHist.size() > MAX_SCORE_HIST) scoreHist.pop_front();
            ++ep;
            env.reset();
            obs = env.getObs();
        }

        auto now = std::chrono::steady_clock::now();
        int ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(now - tRender).count();
        if (ms >= RENDER_MS) { renderStatus(); tRender = now; }
    }

    fill_next_train->clear();
    state->clear(); next_state->clear(); train_state->clear();
    go.clear_graph(logits); go.clean_clear_graph(next_logits);
    cudaFree(argmaxindex); cudaFree(target); cudaFree(action_idx);
    return toMenu;
}


bool runWatchTrain_PPO(SDL_Renderer* ren, TTF_Font* fnt, TTF_Font* fntSm)
{
    SnakeEnv env;
    env.reset();
    auto obs = env.getObs();
    auto STATE = std::make_shared<NodeBackProp>("STATE",1,1,1,obs.total,1);
    GraphOperations go;

    RL_Replay      replay(ROLLOUT, obs.total);
    Actor_Critic   agent(go, STATE, ACT_DIM, HIDDEN_DIM);
    // agent.load("..snake_ppo.bin");
    PPOTrainer<Actor_Critic> PPO(go, agent, replay, PPO_EPOCHS, MINI_BATCH);
    Transition seq;
    auto train_state = std::make_shared<NodeBackProp>("PPO State", PPO.batch, STATE->dim[1],STATE->dim[2],STATE->dim[3],1);
    std::deque<float> scoreHist;

    auto [A_prob_state, V_state] = agent.build_train(STATE);
    int* argmaxindex;
    SafeCudaMalloc("Argmax", argmaxindex, 1);
    cudaMemcpy(STATE->output, obs.grid.data(), obs.total*sizeof(float), cudaMemcpyHostToDevice);
    go.nodes = topological_sort(go.track({A_prob_state, V_state}));

    int       ep         = 0;
    long long totalSteps = 0;
    int       bufSteps   = 0;
    bool quit=false, toMenu=false;
    const int RENDER_EVERY = 2;

    auto tFPS = std::chrono::steady_clock::now();
    int  frmCnt = 0; float sps = 0.f;
    float piLoss=0.f, vLoss=0.f, entropy=0.f;

    while (!quit && !toMenu)
    {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type==SDL_QUIT) { quit=true; break; }
            if (ev.type==SDL_KEYDOWN) switch(ev.key.keysym.sym) {
                case SDLK_ESCAPE: case SDLK_q: quit=true;             break;
                case SDLK_m:      toMenu=true;                         break;
                case SDLK_s:      agent.save("snake_ppo.bin");         break;
            }
        }
        if (quit||toMenu) break;
        go.zero_grad();
        go.forward();

        seq.action   = TopKSampleToCPU(A_prob_state, argmaxindex, ACT_DIM);
        auto res      = env.Obstep(static_cast<Action>((int)seq.action));
        seq.reward    = res.reward;
        seq.log_prob  = logf(ReadValueAt(A_prob_state, (int)seq.action) + 1e-27f);
        seq.value     = ReadValueAt(V_state, 0);
        seq.done      = res.done;

        cudaMemcpy(replay.state + bufSteps * obs.total, STATE->output, obs.total*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(replay.traj  + bufSteps * 5, &seq, sizeof(Transition), cudaMemcpyHostToDevice);

        obs = res.obs;
        cudaMemcpy(STATE->output, obs.grid.data(), obs.total*sizeof(float), cudaMemcpyHostToDevice);
        ++totalSteps; ++bufSteps; ++frmCnt;

        if (res.done)
        {
            scoreHist.push_back((float)res.score);
            if ((int)scoreHist.size() > MAX_SCORE_HIST) scoreHist.pop_front();
            ++ep;
            env.reset();
            obs = env.getObs();
            cudaMemcpy(STATE->output, obs.grid.data(), obs.total*sizeof(float), cudaMemcpyHostToDevice);
        }

        if (bufSteps >= ROLLOUT)
        {
            isNan(V_state);
            agent.save("../snake_ppo.bin");
            float bootstrap_value = 0.f;
            cudaMemcpy(&bootstrap_value, V_state->output, sizeof(float), cudaMemcpyDeviceToHost);
            bootstrap_value *= (1 - res.done);
            go.clear_graph();
            PPO.update(train_state, 0.99f, 0.95f, bootstrap_value);
            bufSteps = 0;

            piLoss   = agent.pi_loss;
            vLoss    = agent.v_loss;
            entropy  = agent.entropy;

            std::tie(A_prob_state, V_state) = agent.build_train(STATE);
            go.nodes = topological_sort(go.track({A_prob_state, V_state}));
        }

        if (frmCnt % RENDER_EVERY == 0) {
            auto now = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(now-tFPS).count();
            if (el >= 1.0) { sps = (float)(frmCnt/el); frmCnt = 0; tFPS = now; }
            float avg=0.f; { int n=std::min((int)scoreHist.size(),20); for(int i=(int)scoreHist.size()-n;i<(int)scoreHist.size();++i) avg+=scoreHist[i]; avg/=std::max(n,1); }
            setColor(ren,{0,0,0,255}); SDL_RenderClear(ren);
            drawGameArea(ren,fnt,env,ep,0,0.f,false,false);
            blitText(ren,fntSm,"S=save weights   M=menu   Q=quit",C_DISABLED,4,GAME_H-16);
            drawPanel(ren,fnt,fntSm,"WATCH TRAIN (PPO)",ep,totalSteps,piLoss,vLoss,entropy,avg,scoreHist,sps,env.score());
            SDL_RenderPresent(ren);
        }
    }

    go.clear_graph(); train_state->clear();
    STATE->clear(); cudaFree(argmaxindex);
    return toMenu;
}

bool runBlankTrain_PPO(SDL_Renderer* ren, TTF_Font* fntMd, TTF_Font* fntSm)
{
    GraphOperations go;
    SnakeEnv env;
    std::deque<float> scoreHist;
    env.reset();
    auto obs = env.getObs();

    auto STATE = std::make_shared<NodeBackProp>("State", 1, 1, 1, obs.total, 1);
    RL_Replay       replay(ROLLOUT, obs.total);
    Actor_Critic    agent(go, STATE, ACT_DIM, HIDDEN_DIM);
    agent.load("../snake_ppo.bin");
    PPOTrainer<Actor_Critic> PPO(go, agent, replay, PPO_EPOCHS, MINI_BATCH);
    Transition seq;

    auto train_state = std::make_shared<NodeBackProp>("PPO State", PPO.batch, STATE->dim[1], STATE->dim[2], STATE->dim[3], 1);
    auto [A_prob_state, V_state] = agent.build_train(STATE);
    go.nodes = topological_sort(go.track({A_prob_state, V_state}));

    int* argmaxindex;
    SafeCudaMalloc("Argmax", argmaxindex, 1);
    cudaMemcpy(STATE->output, obs.grid.data(), obs.total*sizeof(float), cudaMemcpyHostToDevice);

    int       ep         = 0;
    long long totalSteps = 0;
    int       bufSteps   = 0;
    float piLoss = 0.f, vLoss = 0.f, entropy = 0.f;
    auto t0      = std::chrono::steady_clock::now();
    auto tPrint  = t0;
    auto tRender = t0;
    const int RENDER_MS = 500;
    bool quit = false, toMenu = false;

    auto renderStatus = [&]() {
        float avg = 0.f;
        { int n = std::min((int)scoreHist.size(), 20); for (int i = (int)scoreHist.size() - n; i < (int)scoreHist.size(); ++i) avg += scoreHist[i]; avg /= std::max(n, 1); }
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        float  sps = (elapsed > 0) ? (float)(totalSteps / elapsed) : 0.f;

        setColor(ren, C_BG); SDL_RenderClear(ren); fillRect(ren, 0, 0, WIN_W, WIN_H);
        setColor(ren, C_GRID);
        for (int x = 0; x <= WIN_W; x += CELL) SDL_RenderDrawLine(ren, x, 0, x, WIN_H);
        for (int y = 0; y <= WIN_H; y += CELL) SDL_RenderDrawLine(ren, 0, y, WIN_W, y);

        int cx = WIN_W / 2, y = WIN_H / 2 - 140;
        blitTextCentered(ren, fntMd, "HEADLESS TRAINING  [PPO]", C_ACCENT, cx, y); y += 34;
        setColor(ren, C_ACCENT); SDL_RenderDrawLine(ren, cx - 140, y, cx + 140, y); y += 16;

        std::ostringstream ss;
        ss << "Episodes    " << ep;          blitTextCentered(ren, fntSm, ss.str(), C_TEXT,   cx, y); y += 22; ss.str("");
        ss << "Total Steps " << totalSteps;  blitTextCentered(ren, fntSm, ss.str(), C_TEXT,   cx, y); y += 22; ss.str("");
        ss << "Steps / sec " << (int)sps;    blitTextCentered(ren, fntSm, ss.str(), C_ACCENT, cx, y); y += 34; ss.str("");

        blitTextCentered(ren, fntSm, "── Policy ──", C_ACCENT, cx, y); y += 22;
        ss << std::fixed << std::setprecision(4);
        ss << "Pi Loss   " << piLoss;  blitTextCentered(ren, fntSm, ss.str(), C_TEXT, cx, y); y += 22; ss.str("");
        ss << "Val Loss  " << vLoss;   blitTextCentered(ren, fntSm, ss.str(), C_TEXT, cx, y); y += 22; ss.str("");
        ss << "Entropy   " << entropy; blitTextCentered(ren, fntSm, ss.str(), C_TEXT, cx, y); y += 34; ss.str("");

        blitTextCentered(ren, fntSm, "── Score ──", C_ACCENT, cx, y); y += 22;
        ss << std::fixed << std::setprecision(2);
        ss << "Avg (last 20)  " << avg; blitTextCentered(ren, fntSm, ss.str(), C_TEXT, cx, y); y += 22; ss.str("");
        if (!scoreHist.empty()) {
            ss << "Best           " << *std::max_element(scoreHist.begin(), scoreHist.end());
            blitTextCentered(ren, fntSm, ss.str(), C_HEAD, cx, y); y += 34; ss.str("");
        }

        int cw = 400, ch = 80, chartX = cx - cw / 2;
        setColor(ren, C_CHART_BG); fillRect(ren, chartX, y, cw, ch);
        setColor(ren, C_ACCENT);   drawRect(ren, chartX, y, cw, ch);
        blitText(ren, fntSm, "Score History", C_TEXT, chartX + 4, y + 2);
        if (scoreHist.size() > 1) {
            float mx = *std::max_element(scoreHist.begin(), scoreHist.end());
            float mn = *std::min_element(scoreHist.begin(), scoreHist.end());
            if (mx == mn) mx = mn + 1;
            int N = (int)scoreHist.size(); float xStep = (float)cw / (N - 1);
            setColor(ren, C_CHART_LN);
            for (int i = 1; i < N; ++i) {
                int x1 = chartX + (int)((i - 1) * xStep), x2 = chartX + (int)(i * xStep);
                int y1 = y + ch - (int)((scoreHist[i - 1] - mn) / (mx - mn) * (ch - 14)) - 4;
                int y2 = y + ch - (int)((scoreHist[i]     - mn) / (mx - mn) * (ch - 14)) - 4;
                SDL_RenderDrawLine(ren, x1, y1, x2, y2);
            }
        }
        y += ch + 16;
        blitTextCentered(ren, fntSm, "S=save   M=main menu   Q=quit", C_DISABLED, cx, y);
        SDL_RenderPresent(ren);
    };

    renderStatus();

    while (!quit && !toMenu)
    {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) { quit = true; break; }
            if (ev.type == SDL_KEYDOWN) switch (ev.key.keysym.sym) {
                case SDLK_ESCAPE: case SDLK_q: quit = true;  break;
                case SDLK_m:      toMenu = true;              break;
                case SDLK_s:
                    agent.save("../snake_ppo.bin");
                    std::printf("[saved] weights written to snake_ppo.bin\n");
                    break;
            }
        }
        if (quit || toMenu) break;

        go.forward();

        seq.action   = TopKSampleToCPU(A_prob_state, argmaxindex, ACT_DIM);
        seq.log_prob = logf(ReadValueAt(A_prob_state, (int)seq.action) + 1e-27f);
        seq.value    = ReadValueAt(V_state, 0);

        auto res   = env.Obstep(static_cast<Action>((int)seq.action));
        seq.reward = res.reward;
        seq.done   = res.done;

        cudaMemcpy(replay.state + bufSteps * obs.total, STATE->output, obs.total*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(replay.traj  + bufSteps * 5, &seq, sizeof(Transition), cudaMemcpyHostToDevice);

        obs = res.obs;
        ++totalSteps; ++bufSteps;

        if (res.done) 
        {
            scoreHist.push_back((float)res.score);
            if ((int)scoreHist.size() > MAX_SCORE_HIST) scoreHist.pop_front();
            ++ep;
            env.reset();
            obs = env.getObs();
        }
        cudaMemcpy(STATE->output, obs.grid.data(), obs.total*sizeof(float), cudaMemcpyHostToDevice);

        if (bufSteps >= ROLLOUT)
        {
            isNan(V_state);
            agent.save("../snake_ppo.bin");
            float bootstrap_value = 0.f;
            cudaMemcpy(&bootstrap_value, V_state->output, sizeof(float), cudaMemcpyDeviceToHost);
            bootstrap_value *= (1 - res.done);
            go.clear_graph();
            PPO.update(train_state, 0.99f, 0.95f, bootstrap_value);
            bufSteps = 0;

            piLoss   = agent.pi_loss;
            vLoss    = agent.v_loss;
            entropy  = agent.entropy;

            std::tie(A_prob_state, V_state)  = agent.build_train(STATE);
            go.nodes = topological_sort(go.track({A_prob_state, V_state}));

            auto now = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(now - tPrint).count();
            if (el >= 2.0) {
                float avg = 0.f; int n = std::min((int)scoreHist.size(), 20);
                for (int i = (int)scoreHist.size() - n; i < (int)scoreHist.size(); ++i) avg += scoreHist[i];
                avg /= std::max(n, 1);
                double totalSec = std::chrono::duration<double>(now - t0).count();
                std::printf("[ep %6d|steps %9lld|%7.0fsps]pi=%.3f val=%.3f ent=%.3f avgScore=%.2f\n",
                            ep, totalSteps, totalSteps / totalSec,
                            piLoss, vLoss, entropy, avg);
                tPrint = now;
            }
        }

        auto now = std::chrono::steady_clock::now();
        int ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(now - tRender).count();
        if (ms >= RENDER_MS) { renderStatus(); tRender = now; }
    }

    go.clear_graph();
    STATE->clear(); train_state->clear(); cudaFree(argmaxindex);
    return toMenu;
}



bool runWatchTrain(SDL_Renderer* ren, TTF_Font* fnt, TTF_Font* fntSm, TTF_Font* fntMd)
{
    AlgoMode algo = runAlgoPicker(ren, fntMd, fntSm);
    if (algo == AlgoMode::DQN) return runWatchTrain_DQN(ren, fnt, fntSm);
    else                       return runWatchTrain_PPO(ren, fnt, fntSm);
}

bool runWatchAgent(SDL_Renderer* ren, TTF_Font* fnt, TTF_Font* fntSm, TTF_Font* fntMd)
{
    AlgoMode algo = runAlgoPicker(ren, fntMd, fntSm);
    if (algo == AlgoMode::DQN) return runWatchAgent_DQN(ren, fnt, fntSm);
    else                       return runWatchAgent_PPO(ren, fnt, fntSm);
}

bool runBlankTrain(SDL_Renderer* ren, TTF_Font* fntMd, TTF_Font* fntSm)
{
    AlgoMode algo = runAlgoPicker(ren, fntMd, fntSm);
    if (algo == AlgoMode::DQN) return runBlankTrain_DQN(ren, fntMd, fntSm);
    else                       return runBlankTrain_PPO(ren, fntMd, fntSm);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    if (SDL_Init(SDL_INIT_VIDEO)<0)   { std::cerr<<"SDL_Init: "<<SDL_GetError()<<"\n"; return 1; }
    if (TTF_Init()<0)                  { std::cerr<<"TTF_Init: "<<TTF_GetError()<<"\n"; return 1; }

    SDL_Window* win = SDL_CreateWindow("Snake + DQN/PPO",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIN_W, WIN_H, SDL_WINDOW_SHOWN);
    if (!win) { std::cerr<<"CreateWindow: "<<SDL_GetError()<<"\n"; return 1; }

    SDL_Renderer* ren = SDL_CreateRenderer(win,-1,
        SDL_RENDERER_ACCELERATED|SDL_RENDERER_PRESENTVSYNC);
    if (!ren) { std::cerr<<"CreateRenderer: "<<SDL_GetError()<<"\n"; return 1; }

    TTF_Font* fntLg = loadFont(28);
    TTF_Font* fntMd = loadFont(16);
    TTF_Font* fntSm = loadFont(11);
    TTF_Font* fnt   = loadFont(14);
    if (!fntLg||!fntMd||!fnt||!fntSm)
        std::cerr<<"Warning: one or more fonts failed to load – text may be blank\n";

    bool running = true;
    while (running) {
        AppMode mode = runMenu(ren, fntLg, fntMd, fntSm);
        bool back = false;
        switch (mode) {
            case AppMode::PLAY:
                back = runPlay(ren, fnt, fntSm);
                if (!back) running = false;
                break;
            case AppMode::BLANK_TRAIN:
                back = runBlankTrain(ren, fntMd, fntSm);   // shows algo picker internally
                if (!back) running = false;
                break;
            case AppMode::WATCH_TRAIN:
                back = runWatchTrain(ren, fnt, fntSm, fntMd);
                if (!back) running = false;
                break;
            case AppMode::WATCH_AGENT:
                back = runWatchAgent(ren, fnt, fntSm, fntMd);
                if (!back) running = false;
                break;
            case AppMode::MENU:
                running = false;
                break;
        }
    }

    TTF_CloseFont(fntLg); TTF_CloseFont(fntMd);
    TTF_CloseFont(fnt);   TTF_CloseFont(fntSm);
    SDL_DestroyRenderer(ren); SDL_DestroyWindow(win);
    TTF_Quit(); SDL_Quit();
    return 0;
}