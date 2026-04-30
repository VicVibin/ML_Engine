#include "includes/image_loader.h"


void BPrintImage(const graph &X, const int row_size, const int col_size)
{
    auto X_in = std::make_shared<NodeBackProp>(X->op_name, X->dim[0],X->dim[1],X->dim[2],X->dim[3],1);
    cudaMemcpy(X_in->output, X->output, X->total*sizeof(float), cudaMemcpyDeviceToDevice);
    CheckError("Memcpy");
    BMinMaxNorm(X_in);
    CheckError("BatchMinMaxNorm in BPrintImage");
    auto image = Bn2i(X_in);

    if(row_size > 0 && col_size > 0) cv::resize(image, image, cv::Size(row_size*X->dim[0], col_size), 0, 0, cv::INTER_AREA);
    
    cv::imshow(X_in->op_name, image);
    cv::waitKey(0);
    X_in->clear();

}

Ipointer i2p(const std::string& filepath, int row_size, int col_size) 
{
    Ipointer result;
    
    // Read image in color
    cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to load image!\n";
        std::exit(1);
    }

    // If resizing is requested
    if (row_size > 0 && col_size > 0) {
        cv::resize(img, img, cv::Size(col_size, row_size), 0, 0, cv::INTER_AREA);
    }

    // Allocate memory for channels × rows × cols
    int channels = img.channels();
    int rows = img.rows;
    int cols = img.cols;
    float* P = (float*)malloc(rows * cols * channels * sizeof(float));

    // Fill in channel-first format (C × H × W)
    for (int ch = 0; ch < channels; ch++) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(r, c);
                P[ch * rows * cols + r * cols + c] = (float)(pixel[ch]);
            }
        }
    }

    result.memory = P;
    result.dimensions[0] = 1;        
    result.dimensions[1] = channels;
    result.dimensions[2] = rows;
    result.dimensions[3] = cols;

    return result;
}

cv::Mat p2i(Ipointer X)
{
    int rows = X.dimensions[2];
    int cols = X.dimensions[3];
    int channels = X.dimensions[1];
    cv::Mat img(rows, cols, CV_8UC(channels));

    for (int ch = 0; ch < channels; ch++) 
    {
        for (int r = 0; r < rows; r++) 
        {
            for (int c = 0; c < cols; c++) 
            {
                img.at<cv::Vec3b>(r, c)[ch] = (uchar)X.memory[ch * img.rows * img.cols + r* img.cols + c];
            }
        }
    }
    
    return img;

}

graph i2n(std::string filepath, int row_size, int col_size)
{

    cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Failed to load image!\n";
        std::exit(1);
    }
    Ipointer X = i2p(filepath, row_size, col_size);
    auto node = std::make_shared<NodeBackProp>(filepath,X.dimensions[0],X.dimensions[1],X.dimensions[2], X.dimensions[3], 1);
    cudaMemcpy(node->output, X.memory, node->total * sizeof(float), cudaMemcpyHostToDevice);
    free(X.memory);
    return node;
}

cv::Mat n2i(const graph X)
{
    int rows = X->dim[2];
    int cols = X->dim[3];
    int channels = X->dim[1];
    float *cpu_img = (float*)malloc(X->total*sizeof(float));
    cudaMemcpy(cpu_img, X->output, X->total*sizeof(float), cudaMemcpyDeviceToHost);
    cv::Mat img(rows, cols, CV_8UC(channels));
    for (int ch = 0; ch < channels; ++ch) 
    {
        for (int r = 0; r < rows; ++r) 
        {
            for (int c = 0; c < cols; ++c) 
            {   
                uchar val = cpu_img[ch * img.rows * img.cols + r* img.cols + c];
                img.at<cv::Vec3b>(r, c)[ch] = val;
            }
        }
    }
    free(cpu_img);
    return img;

}

Ipointer Bi2p(const str& folder, const int num_images, const int row_size, const int col_size) {
    Text filepaths = FolderPaths(folder, num_images);

    if (filepaths.size() < num_images) {
        std::cerr << "Not enough images in folder\n";
        std::exit(1);
    }

    Ipointer result;
    result.dimensions[0] = num_images;
    result.dimensions[1] = 3; 
    result.dimensions[2] = row_size;
    result.dimensions[3] = col_size;
    result.labels = filepaths;

    float* P = (float*)malloc(3 * num_images * row_size * col_size * sizeof(float));
    if (!P) {
        std::cerr << "Memory allocation failed\n";
        std::exit(1);
    }

    // For each image
    for (int i = 0; i < num_images; i++) {
        cv::Mat img = cv::imread(folder+"/"+filepaths[i], cv::IMREAD_COLOR);
        if (img.empty()) 
        {
            std::cout << "Failed to load image: " << filepaths[i] << "\n";
            free(P);
            std::exit(1);

        }

        cv::resize(img, img, cv::Size(col_size, row_size), 0, 0, cv::INTER_AREA);

        int channels = img.channels();  
        int rows = img.rows;
        int cols = img.cols;
        if(channels != 3)
        {
            std::cout << "Image at path: " << filepaths[i] << " does not have 3 channels \n";
            std::exit(1);
        }

        // Fill in channel-first format (C × H × W)
        for (int ch = 0; ch < channels; ch++) {
        for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) 
        {
                    cv::Vec3b pixel = img.at<cv::Vec3b>(r, c);
                    size_t index = i*channels*rows*cols + ch*rows*cols + r*cols + c;
                    P[index] = (float)pixel[ch];
                }
            }
        }
    }
    result.memory = P;

    return result;
}

graph Bi2n(str filepath,const int num_images, const int row_size , const int col_size)
{

    Ipointer X = Bi2p(filepath, num_images, row_size, col_size);
    std::cout << "Loaded " << num_images << " images from " << X.labels.size() << "\n";
    for(auto & name : X.labels)
    {
        std::cout << name << "\n";
    }
    auto node = std::make_shared<NodeBackProp>(filepath,X.dimensions[0],X.dimensions[1],X.dimensions[2], X.dimensions[3], 1);
    cudaMemcpy(node->output, X.memory, node->total * sizeof(float), cudaMemcpyHostToDevice);
    free(X.memory);
    return node;
}

cv::Mat Bn2i(const graph X)
{
    int batch    = X->dim[0];  
    int channels = X->dim[1];  
    int rows     = X->dim[2];
    int cols     = X->dim[3];

    float* cpu_img = (float*)malloc(X->total * sizeof(float));
    cudaMemcpy(cpu_img, X->output, X->total * sizeof(float), cudaMemcpyDeviceToHost);

    cv::Mat img(rows, cols * batch, CV_8UC(channels), cv::Scalar(0));

    for (int b = 0; b < batch; ++b) {
    for (int ch = 0; ch < channels; ++ch) {
    for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
        int idx = b * (channels * rows * cols) + ch * (rows * cols) + r * cols + c;
        uchar val = static_cast<uchar>(cpu_img[idx]);
        img.at<cv::Vec3b>(r, b * cols + c)[ch] = val;
    }}}}

    free(cpu_img);
    return img;
}
