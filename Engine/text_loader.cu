#include "text_loader.h"

Text TextProcessor::readAllStories(const str& folderPath) 
{
    Text allLines;
    Text filenames = FolderPaths(folderPath);
    if (filenames.empty()) 
    {
        std::cerr << "No files found in directory: " << folderPath << "\n";
        return allLines;
    }

    std::cout << "Found " << filenames.size() << " files in directory" << "\n";

    for (const str& filename : filenames) 
    {
        std::ifstream file(folderPath + "/" + filename);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open file " << filename << "\n";
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

bool TextProcessor::isAlpha(const str& word) {
    if (word.empty()) return false;
    return std::all_of(word.begin(), word.end(), ::isalpha);
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

Text TextProcessor::tokenize(const str& line) {
    Text tokens;
    std::istringstream iss(line);
    str word;
    while (iss >> word) {
        tokens.push_back(word);
    }
    return tokens;
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


