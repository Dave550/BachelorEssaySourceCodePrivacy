# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
#https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
#Data path is /Users/sophia/Development/ascanalysis/Data/
#from src import ngramCodeSimilarity as ng
import sys
sys.path.insert(0, '../src')

import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from pathlib import Path
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def main():
    print("main of countvectorizer")
    newfile = '/Users/blovs/CSCI Hw/AutomatedSourceCodeAnalysis-main/Data/reformattedNgSc.txt'
    #oldfile = fileProcessing('/Users/sophia/Development/ascanalysis/Data/alternate.txt')


    #csv = '/Users/sophia/Development/ascanalysis/Results/1gCodeJamSoln.csv'
    data = '/Users/blovs/CSCI Hw/AutomatedSourceCodeAnalysis-main/Data/alternate.txt'
    #reformat(data, newfile)

    file = fileProcessing('/Users/blovs/CSCI Hw/AutomatedSourceCodeAnalysis-main/Data/reformattedNgSc.txt')
    path = '/Users/blovs/CSCI Hw/AutomatedSourceCodeAnalysis-main/Results/3gCodeJamSoln.csv'

    vectorize(file, cust_tokenizer, 3 , True, path)
    #remove path later maybe


#custom tokenizer - consider calling another file
def cust_tokenizer(text):
    text = re.split('([\s.,;()]+)', text)
    text = list(filter(None, text))
    return text

#open file and read lines
def fileProcessing(file):
    script_location = Path(__file__).absolute().parent
    file_location = script_location / file
    doc1 = file_location.open()
    doc1 = doc1.readlines()

    return doc1

#vectorizer
def vectorize(file, tokenizer, n , boolPrint, path):
    #vectorize the document, fitting, etc.
    #min_df eliminates ngrams that appear in less than 2 documents
    vectorizer = CountVectorizer(file, ngram_range= (n,n), min_df=1, tokenizer=tokenizer)
    vectorizer.fit(file)
    vector = vectorizer.fit_transform(file)

    #different representations of the data
    if(boolPrint==True):
        print('Vocabulary: ')
        print(vectorizer.vocabulary_)

        print('Full vector: ')
        print(vector)

        print()
        print('values')
        print(vectorizer.get_feature_names())

        print()
        print('array')
        print(vector.toarray())

        #print()
        #print('stop words')
        #print(vectorizer.stop_words_)

        sum = 0
        for i in vectorizer.vocabulary_:
            sum = sum + 1

        print("sum ", sum)

    updateCSV(vector.toarray(), path)
    #redVector = dimReduct(vector)
    #updateCSV(redVector, path)

    #print('Vocabulary: ')
    #print(vectorizer.vocabulary_)

    sum = 0
    for i in vectorizer.vocabulary_:
        sum = sum + 1

    print("sum ", sum)

    sum = 0
    for i in redVector:
        for j in i:
            sum = sum + 1
        break

    print("sum new ", sum)

    return vector

def printNgram(file, num):
    #prints file as an ngram... necessary?
    #'/Users/sophia/Development/ascanalysis/Data/reformattedNgSc.txt'
    print(ng.output(file, num))

def outputVector():
    #outputs the vector... necessary?
    doc = fileProcessing('/Users/blovs/CSCI Hw/AutomatedSourceCodeAnalysis-main/Data/alternate.txt')
    vectorData = vectorize(doc, cust_tokenizer, 3, False)
    return vectorData

def updateCSV(array, path):

    strArr = str(array)
    temp = strArr.split("]")

    with open(path, 'a') as fd:
        for i in temp:
            #print(i)

            strArr = " ".join(i.split())
            fd.write('\n'+strArr.replace(' ',',').replace('[','').replace(']',''))


def reformat(file, newfile):
    #this reformats the source code into one line
    script_location = Path(__file__).absolute().parent
    file_location = script_location / file
    doc1 = file_location.open()
    doc1 = doc1.readlines()

    newfile_location = script_location / newfile
    doc2 = newfile_location

    with open(doc2, 'a') as output:
        for i in doc1:
            i = i.replace('\n', ' \\n ')
            output.write(i)
        output.write('\n')

    return output

def dimReduct(array):
    #reduce the dimensions of the ngrams bc holy moly
    #https://scikit-learn.org/stable/modules/feature_selection.html#

    arr = array.toarray()

    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    arr = sel.fit_transform(arr)

    arr = arr.tolist()
    arr = np.array(arr)

    return arr




if __name__ == '__main__':
    main()


