import numpy as np
# import scipy
# import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

def function(x):
    return np.sin(x)


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


class RectangularRuleLeft(Integral):
    def __init__(self,funct,bound,n):
        super().__init__(funct, bound, n)
        self.name = "Left Rectangular Rule"

    def integrate(self):
        h = (self.bound[1]-self.bound[0])/self.n
        return h*np.sum(self.funct(np.linspace(self.bound[0],self.bound[1]-h,int((self.bound[1]-self.bound[0])/h))))

class RectangularRuleMix(Integral):
    def __init__(self,funct,bound,n):
        super().__init__(funct, bound, n)
        self.name = "Mix Rectangular Rule"

    # def integrate(self):
    #     return (RectangularRuleLeft(self.funct, n= self.n/2) + RectangularRuleRight(self.funct, n= self.n/2))/2

class TrapezoidalRule(Integral):
    def __init__(self,funct,bound,n):
        super().__init__(funct, bound, n)
        self.name = "Trapezoidal Rule Rule"

    def integrate(self):
        h = (self.bound[1]-self.bound[0])/self.n
        return (h/2)*( 2*np.sum(self.funct(np.linspace(self.bound[0]+h,self.bound[1]-h,int((self.bound[1]-self.bound[0])/h)-2)))+ self.funct(self.bound[0]) + self.funct(self.bound[1]) ) 


class MonteCarlo(Integral):
    def __init__(self,funct,bound,n):
        super().__init__(funct, bound, n)
        self.name = "Monte Carlo Method"

    def integrate(self):
        random_number= np.random.uniform(self.bound[0],self.bound[1],self.n)
        return np.mean(self.funct(random_number))*(self.bound[1]-self.bound[0])


class SimpsonRule(Integral):
    def __init__(self,funct,bound,n):
        super().__init__(funct, bound, n)
        self.name = "SimpsonRule Rule"

    def integrate(self):
        h = (self.bound[1]-self.bound[0])/self.n
        boundterm= self.funct(self.bound[0])+self.funct(self.bound[1])
        oddterm = np.sum(4*self.funct(h*np.array([self.bound[0]+(2*x+1) for x in range(int(self.n/2))])))
        eventerm = np.sum(2*self.funct(h*np.array([self.bound[0]+(2*x) for x in range(1,int(self.n/2))])))
        return (h/3)*(boundterm+oddterm+eventerm)

fun_set = [lambda x: x**2, lambda x: np.sin(x), lambda x: x**(0.5), lambda x: np.log(x)]
for function in fun_set:
    integral = iter(RectangularRuleRight(function,[1,5],2))
    integral2 = iter(RectangularRuleLeft(function,[1,5],2))
    integral3 = iter(RectangularRuleMix(function,[1,5],2))
    integral4 = iter(MonteCarlo(function,[1,5],2))
    integral5 = iter(SimpsonRule(function,[1,5],2))
    # print(integral.integrate())
    # gprint(integral)
    datares = []
    datares2 = []
    datares3 = []
    datares4 = []
    datares5 = []
    datapoint = []
    for x in range (100):
        datapoint.append(x)
        datares.append(next(integral))
        datares2.append(next(integral2))
        datares3.append(next(integral3))
        datares4.append(next(integral4))
        datares5.append(next(integral5))

    plt.plot(datapoint,datares, label ='Right rule')
    plt.plot(datapoint,datares2,label ='Left rule')
    plt.plot(datapoint,datares3, label ='Mix rule')
    plt.plot(datapoint,datares4, label ='Montecarlo rule')
    plt.plot(datapoint,datares5, label ='Simpson rule')
    plt.legend()
    plt.show()



