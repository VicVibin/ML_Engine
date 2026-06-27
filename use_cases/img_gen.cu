#include "diffusion.h"

int main()
{
    GraphOperations go; 
    const int T = 1000, init_depth = 32, t_hidden = 128, img_size = 64, epochs = 6000;
    auto base = Bi2n("../images", 1, img_size, img_size); // Loads a batch of images and converts to a node 
    StandardNorm(base); // Normalizes using (X/255 - 0.5) / 0.5
    auto input  = go.like(base, "Input Image"), target = go.like(base, "Target Image");
    U_NET model(3,3,init_depth,t_hidden); 
    model.load("../best_models/up.bin");
    //DiffusionTrainer<U_NET> trainer(go, model, base, input, target, T);
    //printMem("After Graph allocation");
    //trainer.train(epochs, 100, true, 1000, "../best_models/up.bin");
    auto test = go.like(base, "Test Image");
    Noise(test);
    Sampling<U_NET> sampler(model, test, T-1,T);
    sampler.loop(T-1, 0);
    sampler.display("../best_models/test_sample.png", 5, 512, 512);
    return 0;
}