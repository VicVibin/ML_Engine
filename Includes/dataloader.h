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
using Text = std::vector<std::string>;

Text FolderPaths(const std::string& folder, const int filenums);

class TextProcessor {
public:
    str toLower(const str& string);                    // DECLARATION only
    str removePunctuation(const str& string);          // DECLARATION only
    bool isAlpha(const str& word);                     // DECLARATION only
    Text tokenize(const str& line);                    // DECLARATION only
    Text getFilesInDirectory(const str& folderPath);   // DECLARATION only
    Text readAllStories(const str& folderPath);        // DECLARATION only
    Text cleanText(const Text& lines);                 // DECLARATION only

};

Text LoadStory(const str& path);
void Reading(Text string);
Text read_words(Text words, const int start, const int end);
