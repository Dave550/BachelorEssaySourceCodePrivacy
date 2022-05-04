# BachelorEssaySourceCodePrivacy
This project is based on the idea of creating code transformation in order to achieve source code privacy

SRC folder: 
countvectorizer.py:
This is a count vectorizer created by Sophia Frankel. It is essential for running a n gram comparision in a machine learning algorithm. To start you need to uncomment reformat in the main, then run the file. You should go back to wherever you reformated text file is going to, create a new line for each file that is being reformated, go back to countvectorizer.py, comment out reformat, and run it again for the best results.

modeling.py:
This is a modeling file that contains all the code for machine learning. This was originally created by Sophia Frankel but I added onto the code and created my own method in which I used during all of my testing called MLPClass. You need to figure out the max columns length in whatever input that the machine will be training on and put that into n.

ngrams.py:
This file has the methods that change the signitures and for to while loop. Very straight foward method, some of it is pipedlined right now but that can be changed by adding in a print method that asks what the file name is and then openning that file instead of the default file that is being opened right now.

research.py:
This is a graph ADT implmentation that I used to pratice some research on Laplace Noise. Its interesting and a good visual of what stastical nosie actually does.

Data folder:
alternative.txt:
This textfile is the orignal spot where you need to put the source code for count vectorization.

reformattedNgSC:
This textfile is the reformatted alternative text file after running the count vectorization.

Results folder:
All of these excel sheets are count vectorizations of the level of ns. For example, the excel named 3gCodeJamSoln is a excel file for a 3 gram solution.

