import numpy as np
# import scipy
# import matplotlib.pyplot as plt
import time
from abc import ABC, abstractmethod

def function(x):
    return x*x+5


class Integral:
    def __init__(self, funct, bound, n):
        self.funct = funct
        self.bound = bound
        self.n = n
        self.name = ""
        self.result = None

    def __str__(self):
        return f"The {self.name} Integration result: {self.result}"
    
    @abstractmethod
    def integrate(self):
        pass

    def __iter__(self):
        self.n = 2
        return self
   
    def __next__(self):
       self.n +=1
       return self.integrate()

class RectangularRuleRight(Integral):
    def __init__(self,funct,bound,n):
        super().__init__(funct, bound, n)
        self.name = "Right Rectangular Rule"

    def integrate(self):
        h = (self.bound[1]-self.bound[0])/self.n
        return h*np.sum(self.funct(np.linspace(self.bound[0]+h,self.bound[1],int((self.bound[1]-self.bound[0])/h))))





# def RectangularRuleRight(funct,bound = [0,5],n=50000):
#     h = (bound[1]-bound[0])/n
#     return h*np.sum(funct(np.linspace(bound[0]+h,bound[1],int((bound[1]-bound[0])/h))))
#

integral = iter(RectangularRuleRight(function,[0,5],2))
# print(integral.integrate())
# gprint(integral)
for x in range (100):
    print(next(integral))
