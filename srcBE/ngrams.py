#David Sorensen

import collections
import matplotlib.pyplot as plt
import pandas as pd
import nltk, re, pprint
from nltk import word_tokenize
import sklearn as skms
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree

    


class AI:
    TotalFreq= {}
    whatFreqOn=0
    

    def __init__(self):
        self.TotalFreq[0]=None
        self.whatFreqOn=0
    def LineToken(self):
        Line= input("What the line were going to break up:")
        token= word_tokenize(Line)
        print(token)
        print(len(token))


    def Line(self):
        Line= input("What the line were going to break up:")
        ngram= input("What ngram r we doing:")

        LenOfLine= len(Line)
        List= []
        size= int(ngram, base = 10)



        
        for i in range(LenOfLine-int(ngram)+1):
            List.append(Line[int(i): int(i) +int(ngram)])
        print(List)

        document = ["My name is David", "My name is Kelly", "David is 21"]


        # Create a Vectorizer Object
        vectorizer = CountVectorizer()

        vectorizer.fit(document)


        # Printing the identified Unique words along with their indices
        print("Vocabulary: ", vectorizer.vocabulary_)
  
        # Encode the Document
        vector = vectorizer.transform(document)
  
        # Summarizing the Encoded Texts
        print("Encoded Document is:")
        print(vector.toarray())


    def textFile(self):
        ngram= input("What ngram r we doing:")
        #ngram=3
        List= []
        Dict= {}
        fileName= input("Whats the name of the file?")
        Textfile= open(fileName, "r")
        Plain= Textfile.readlines()
        numLines= len(Plain)
        for i in range(numLines):
            size = len(Plain[i])
            for x in range(size- int(ngram)+1):
                List.append(Plain[i][int(x) : int(x) + int(ngram)])
        print(List)
        print(Plain)
        data= pd.Series(List)
        print(data.value_counts())
        FreqTable= data.value_counts()
        #print(FreqTable[4])

        self.TotalFreq[self.whatFreqOn]= FreqTable
        self.whatFreqOn= self.whatFreqOn + 1

        #print(self.TotalFreq[0])

        # Create a Vectorizer Object
        vectorizer = CountVectorizer(min_df = 10)

        vectorizer.fit(List)


        # Printing the identified Unique words along with their indices
        print("Vocabulary: ", vectorizer.vocabulary_)

        print(vectorizer.get_feature_names())
  
        # Encode the Document
        vector = vectorizer.transform(List)
  
        # Summarizing the Encoded Texts
        print("Encoded Document is:")
        print(vector.toarray())

    def textFiletoCSV(self):
        #ngram= input("What ngram r we doing:")
        for i in range(8):
            ngram=3
            List= []
            Dict= {}
            fileName= input("Whats the name of the file?")
            Textfile= open(fileName, "r")
            Plain= Textfile.readlines()
            numLines= len(Plain)
            for i in range(numLines):
                size = len(Plain[i])
                for x in range(size- int(ngram)+1):
                    List.append(Plain[i][int(x) : int(x) + int(ngram)])
            print(List)
            print(Plain)
            data= pd.Series(List)
            print(data.value_counts())
            FreqTable= data.value_counts()
            print(FreqTable[4])

            self.TotalFreq[self.whatFreqOn]= FreqTable
            self.whatFreqOn= self.whatFreqOn + 1

            print(self.TotalFreq[0])
        BenqData= pd.DataFrame({'Author': ['Author A', 'Author A', 'Author A', 'Author A', 'Author A', 'Author A', 'Author A'], 'FreqTables': [self.TotalFreq[0], self.TotalFreq[1], self.TotalFreq[2], self.TotalFreq[3], self.TotalFreq[4], self.TotalFreq[5], self.TotalFreq[6], self.TotalFreq[7]]})
        datatoexcel = pd.ExcelWriter('CarsData1.xlsx')
        BenqData.to_excel(datatoexcel)
        datatoexcel.save()
        print('DataFrame is written to Excel File successfully.')



    def textFileToken(self):
       fileName= input("Whats the name of the file?")
       Textfile= open(fileName, "r")
       Plain= Textfile.readlines()
       numLines= len(Plain)
       
       token= word_tokenize(Plain)
       print(token)

    #def editTextfile():
        

    def decisionTree(self):
        dtc= tree.DecisionTreeClassifier()
        output= [0,1]
        dtc.fit(self.TotalFreq, self.TotalFreq)

    def intTolong(self):
        #Line= input("What the line were going to break up:")
        #ngram= input("What ngram r we doing:")

        #LenOfLine= len(Line)
        #List= []

        #print(Line[0])

        #token= word_tokenize(Line)

        #print(token[0])

        f=open('newLong.txt', 'a')
        #s=open('newDouble.txt', 'r')
        fileName= input("Whats the name of the file?")
        s=open(fileName, 'r')

        Plain= s.readlines()
        numLines= len(Plain)

        #token= word_tokenize(Line)

        #print(token[0])

        stringFor= "for"

        stringFloat= "int"

        stringFloat2= "Int"
        for x in range(numLines):
            token= word_tokenize(Plain[x])
            if((stringFloat2 in token) or (stringFloat in token)):
                #2 comparator
                if((stringFor in token and (len(token)==15))):
                   whatSig= "long"
                   sizeInit= token[5]
                   name= token[3]
                   sizeMaxorMin= token[10]
                   operator1= token[8]
                   operator2= token[9]
                   f.write("for" + "(" + "long"+ name+ "=" + sizeInit + ";" + name + operator1 + operator2 + sizeMaxorMin + ";" + name+ "++" + ")")
                if((stringFor in token and (len(token)==14))):
                    sizeInit= token[5]
                    name= token[3]
                    sizeMaxorMin= token[9]
                    operator1= token[8]
                    f.write("for" + "(" + "long"+ name+ "=" + sizeInit + ";" + name + operator1 + sizeMaxorMin + ";" + name+ "++" + ")")
                if(token[-1].isnumeric()):
                    f.write("long" + " "+ token[1]+token[2]+ ";"+ "\n")
                else:
                    f.write(Plain[x])
                    f.write("\n")
                    
                    
            #output= [0]*len(token)
            #output[0]= "double"
            #for i in range(1,len(token)):
             #   output[i]= token[i]
            #print(output)
            else:
                f.write(Plain[x])
                f.write("\n")

        s.close()
        f.close()
    def floatTodouble(self):
       # Line= input("What the line were going to break up:")
       # ngram= input("What ngram r we doing:")

       # LenOfLine= len(Line)
       # List= []

        #print(Line[0])
        f=open('newDouble.txt', 'a')
        s=open('newFile.txt', 'r')
        Plain= s.readlines()
        numLines= len(Plain)

        #token= word_tokenize(Line)

        #print(token[0])

        stringFloat= "float"

        stringFloat2= "Float"
        for x in range(numLines):
            token= word_tokenize(Plain[x])
            if((stringFloat2 in token) | (stringFloat in token)):
                 f.write("double" + " "+ token[1]+token[2]+ ";"+ "\n")
            #output= [0]*len(token)
            #output[0]= "double"
            #for i in range(1,len(token)):
             #   output[i]= token[i]
            #print(output)
            else:
                f.write(Plain[x])
                f.write("\n")

        s.close()
        f.close()

        
    def forTowhile(self):
       # ngram= input("What ngram r we doing:")
        #ngram=3
       # List= []
      #  Dict= {}
        fileName= input("Whats the name of the file?")
        Textfile= open(fileName, "r")
        Plain= Textfile.readlines()
        numLines= len(Plain)
        f= open('newFile.txt', 'a')

        
        #print(Line[0])

       # token= word_tokenize(Plain[0])

        stringFor= "for"

        stringComment= "//"

        stringCom1= "/*"

        stringCom2= "*/"



        #Line= input("What the line were going to break up:")
        #token= word_tokenize(Line)

        for i in range(numLines):
            token= word_tokenize(Plain[i])
            if((stringFor in token) and not(stringComment in token) and not(stringCom1 in token) and not(stringCom2 in token)):
                #does it contain a two part comparator ie <=
                if(len(token)== 15):
                    f.write("programmer needs to write in the incrumenation \n")
                    whatSig= token[2]
                    sizeInit= token[5]
                    name= token[3]
                    sizeMaxorMin= token[10]
                    operator1= token[8]
                    operator2= token[9]
                    f.write(whatSig + " "+ sizeInit+ ";")
                    f.write("\n")
                    f.write("while"+ "(" + name + operator1 + operator2 + sizeMaxorMin+ ")"+ "{")
                    f.write("\n")
                else:
                    f.write("programmer needs to write in the incrumenation \n")
                    whatSig= token[2]
                    sizeInit= token[5]
                    name= token[3]
                    sizeMaxorMin= token[9]
                    operator1= token[8]
                    f.write(whatSig + " "+ sizeInit+ ";")
                    f.write("\n")
                    f.write("while"+ "(" + name + operator1 + sizeMaxorMin+ ")"+ "{")
                    f.write("\n")

            else:
                f.write(Plain[i])
                f.write("\n")



        f.close()
        Textfile.close()

            


                    

            
           
        



def main():
    test= AI()
    test.Line()
    #test.textFiletoCSV()
    #test.floatTodouble()

    #test.textFile()
    #test.LineToken()

    #for i in range(8):
      #  test.textFile()

    #test.decisionTree()

    #test.forTowhile()
    #test.floatTodouble()

    #test.intTolong()

   
    
        
          
                

main()
