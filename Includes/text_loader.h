#pragma once
#include "dataloader.h"

struct BatchTexts
{
    BatchText encoder;
    BatchText decoder;
    BatchText target;

    BatchTexts(int batch_size,int clen):encoder(batch_size,Text(clen+1)),decoder(batch_size,Text(clen+1)),target(batch_size,Text(clen+1)){}
};

class TextProcessor {
public:
    str toLower(const str& string);              
    str removePunctuation(const str& string);       
    bool isAlpha(const str& word);                   
    Text tokenize(const str& line);                 
    Text readAllStories(const str& folderPath);     
    Text cleanText(const Text& lines);           

};

Text LoadStory(const str& path);
void Reading(Text string);
Text read_words(Text words, const int start, const int end);
