#pragma once
#include "engine.h"
#include <opencv2/opencv.hpp>

void BPrintImage(const graph &X, const int row_size =0, const int col_size = 0);

using Text = std::vector<std::string>;

struct Ipointer
{
    float *memory;
    int dimensions[4];
    long long total_size;
    Text labels;
};

Ipointer i2p(const std::string& filepath, int row_size = 0, int col_size = 0);
cv::Mat p2i(Ipointer X);
graph i2n(std::string filepath, int row_size = 0, int col_size = 0);
cv::Mat n2i(const graph X);
Ipointer Bi2p(const str& folder, const int num_images, const int row_size, const int col_size);
graph Bi2n(str filepath,const int num_images, const int row_size , const int col_size);
cv::Mat Bn2i(const graph X);
