#create an n-gram from a file
#https://towardsdatascience.com/natural-language-processing-count-vectorization-with-scikit-learn-e7804269bb5e

import re
from nltk.util import ngrams

def openDoc(file):
    doc1 = open(file, "r")
    ngram = doc1.read()
    return ngram

def format(sentence):
    output = sentence.lower()  # set sentence to lowercase

    token = re.split('([^a-zA-Z0-9])', output)

    for i in token:
        if i == '':
            token.remove(i)

    return token

def ngram(token, ngramNum):
    output = list(ngrams(token, ngramNum))  # uses ngram function from nltk
    return output

def output(file, ngramVal):
    #print("hey")
    token = format(openDoc(file))
    formatted = ngram(token, ngramVal)
    return formatted
