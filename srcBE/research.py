#David Sorensen

#Graph ADT
import collections
from random import randrange
from array import*
import matplotlib.pyplot as plt
import numpy as np
class GraphADT:
    Matrix = [[]]
    Sens= 0
    numNodes= 0
    Dict= {}

    def __init__(self, sens, numnodes):
        self.Sens= sens
        self.numNodes= numnodes
        self.Matrix= [[0 for i in range(numnodes)] for j in range(numnodes)]
        for i in range(numnodes):
            self.Dict[i]= []

    def addEdge(self, start, end):
       self.Matrix[start][end]= 1
       self.Dict.get(start).append(end)


    def printMatrix(self):
       print("printmatrix")
       for i in range(self.numNodes):
           print(i, end="")
           print(":", "", end="")
           
           for x in range(self.numNodes):
               print(self.Matrix[i][x], "", end="")
           print()
           
    def numOfOutedges(self, whichNode):
        temp=0
        for i in range(self.numNodes):
            if(self.Matrix[whichNode][i]==1):
                temp= temp+1
                
        return temp
    

   
    def OutputOfNumDevelop(self, amI, whichRepo):
        if(amI == "contributor"):
            print("Im a contributor and want to know how many other contributor are working on this repocity")
            temp=len(self.Dict.get(whichRepo))
            #temp= self.numOfOutedges(whichRepo)
            temp2= np.random.laplace(self.getMean(), .5, 1)
            temp= temp+ temp2
            print(temp)
            

        else:
            print("Im the developer, Im the one who can see the true value")
            temp=len(self.Dict.get(whichRepo))
            #temp= self.numOfOutedges(whichRepo)
            print("these are the nodes contriubitng to the repo", whichRepo, self.Dict.get(whichRepo))
            print("this is the amount of nodes", temp)
                     

    def makeHistogram(self):
        data= []
        for i in range(self.numNodes):
            temp= self.numOfOutedges(i)
            data.append(temp)
        plt.hist(data, bins = self.numNodes)
        plt.show()

    def list(self, whichNode):
        print(self.Dict.get(whichNode))

    def getMean(self):
        temp=0
        for i in range (self.numNodes):
            temp= self.numOfOutedges(i)
        mean= temp/self.numNodes

        return mean
       
        
            

def main():
    
    obj= GraphADT(1, 10)
    #for i in range (30):
      #  start= randrange(10)
     #   end= randrange(10)
     #   obj.addEdge(start, end)
   
    obj.addEdge(0,1)
    obj.addEdge(3,1)
    obj.addEdge(3,5)
    obj.addEdge(2,6)
    obj.addEdge(0,9)
    obj.addEdge(6,1)
    obj.printMatrix()
    obj.OutputOfNumDevelop("contributor", 3)
    obj.OutputOfNumDevelop("developer", 3)
    obj.makeHistogram()

    obj.getMean()



main()
