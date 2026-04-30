#include "includes/dataloader.h"

Text FolderPaths(const std::string& folder, const int filenums ) 
{
    Text files;

    str search_path = folder + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(search_path.c_str(), &fd);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                files.emplace_back(fd.cFileName);
            }
        } while (FindNextFileA(hFind, &fd));
        FindClose(hFind);
    }

    if (files.size() < filenums)
    {
        std::cout << "Not enough files in directory... num files: " << files.size() << "\n";
        if(files.size() > 0 ) for(const auto &i : files) std::cout << i << "\t";
    }

    if(files.size() == filenums)
    {
        return files;
    }

    Text new_files;
    for(int i =0; i < filenums;++i) new_files.push_back(files[i]);

    return new_files;
}

str TextProcessor::toLower(const str& string) 
{   

    str result = string;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

str TextProcessor::removePunctuation(const str& string) {
    str result;
    str punctuation = ",.\"'!@#$%^&*(){}?/;`~:<>+=-\\";
    for (char c : string) {
        if (punctuation.find(c) == str::npos) {
            result += c;
        }
    }
    return result;
}

bool TextProcessor::isAlpha(const str& word) {
    if (word.empty()) return false;
    return std::all_of(word.begin(), word.end(), ::isalpha);
}

Text TextProcessor::tokenize(const str& line) {
    Text tokens;
    std::istringstream iss(line);
    str word;
    while (iss >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

Text TextProcessor::getFilesInDirectory(const str& folderPath) {
    Text filenames;
    str searchPath = folderPath + "\\*";
    WIN32_FIND_DATAA findFileData;
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                filenames.push_back(folderPath + "\\" + findFileData.cFileName);
            }
        } while (FindNextFileA(hFind, &findFileData) != 0);
        FindClose(hFind);
    }
    return filenames;
}

Text TextProcessor::readAllStories(const str& folderPath) {
    Text allLines;
    Text filenames = getFilesInDirectory(folderPath);
    if (filenames.empty()) {
        std::cerr << "No files found in directory: " << folderPath << std::endl;
        return allLines;
    }

    std::cout << "Found " << filenames.size() << " files in directory" << std::endl;

    for (const str& filename : filenames) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open file " << filename << std::endl;
            continue;
        }

        str line;
        while (std::getline(file, line)) {
            if (line == "----------") break;
            if (!line.empty()) allLines.push_back(line);
        }

        file.close();
    }

    return allLines;
}

Text TextProcessor::cleanText(const Text& lines) {
    Text cleanedWords;
    for (const str& line : lines) {
        str lowerLine = toLower(line);
        str cleanLine = removePunctuation(lowerLine);
        Text tokens = tokenize(cleanLine);
        for (const str& word : tokens) {
            if (isAlpha(word)) {
                cleanedWords.push_back(word);
            }
        }
    }
    return cleanedWords;
}

Text LoadStory(const str& path)
{
    TextProcessor processor;
    Text data = processor.readAllStories(path);
    Text cleanedWords = processor.cleanText(data);
    return cleanedWords;
}

void Reading(Text string)
{
    for (const auto & word: string){std::cout << word << "\t";}
        std::cout << "\n";
        std::cout << "_____________________ \n";
}

Text read_words(Text words, const int start, const int end)
{
    Text selectedWords;
    for (int i = start; i < end && i < words.size(); ++i)
    {
        selectedWords.push_back(words[i]);
    }
    return selectedWords;
}
