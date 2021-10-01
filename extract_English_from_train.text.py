#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Program to identify language of a given string
"""

import fasttext

class LanguageIdentification:
    """
    Class object for Language Identification
    """

    def __init__(self):
        """
        Initializing class objects
        """
        pretrained_lang_model = "./tmp/lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        """
        Attributes
        text: type str, the sentence for which language is to be identified
        """
        predictions = self.model.predict(text, k=1) # top 1 matching languages
        return predictions

if __name__ == '__main__':


    LANGUAGE = LanguageIdentification()
    with open("./data/eng_mal","r") as f:
        for line in f.readlines():
            line=line.strip("\n")
            LANG = LANGUAGE.predict_lang(line)
            #print(LANG[1][0])
            
            if(LANG[0][0]=='__label__en' and LANG[1][0]>=0.8):
                print(line)

