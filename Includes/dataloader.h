#pragma once
#include <vector>
#include <iostream>
#include <map>
#include <utility>
#include <functional>
#include <random>
#include <unordered_set>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#ifdef _WIN32
    #include <windows.h>
#else
    #include <dirent.h>
    #include <sys/stat.h>
#endif

using str = std::string;
using Text = std::vector<str>;
using BatchText = std::vector<Text>;

Text FolderPaths(const std::string& folder, const int filenums = -1);