'''
Topics on GAB data
'''

import json
import spacy
spacy.load('en')
import sys
from spacy.lang.en import English
from gensim.corpora.textcorpus import TextCorpus
import nltk
from gensim import utils
import gensim.models.ldamodel
from gensim import corpora

en_stop = spacy.lang.en.stop_words.STOP_WORDS
parser = English()

def tokenize(text):
    lda_tokens = []
    tokens = text.split()
    for token in tokens:
        lda_tokens.append(token)
    return lda_tokens

def token_stream(posts_file):
    with open(posts_file, 'r', encoding='ascii',errors='ignore') as handle:
        for new_line in handle:
            tokens = [x for x in tokenize(new_line) if not x in en_stop]

            if len(tokens) < 15:
                continue

            yield tokens

class Corpus(object):
    def __init__(self, filename, dictionary):
        super().__init__()
        self.filename = filename
        self.dictionary = corpora.Dictionary.load(dictionary)

    def __iter__(self):
        for tokens in token_stream(self.filename):
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        i = 0
        for tokens in token_stream(self.filename):
            i += 1
        return i
            
if __name__ == '__main__':
    posts_file = sys.argv[1]
    dict_dest = sys.argv[2]

    dictionary = corpora.Dictionary(token_stream(posts_file))

    dictionary.save(dict_dest)

