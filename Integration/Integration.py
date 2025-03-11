import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

def function(x):
    return x*x

# RiemannIntegral
# Don't forget to always multiply outside, not inside.
def RectangularRuleRight(funct,bound = [0,5],h=0.0001):
    return h*np.sum(funct(np.linspace(bound[0]+h,bound[1],int((bound[1]-bound[0])/h))))

def RectangularRuleLeft(funct,bound = [0,5],h=0.0001):
    return h*np.sum(funct(np.linspace(bound[0],bound[1]-h,int((bound[1]-bound[0])/h))))

def RectangularMix(funct,bound = [0,5], h=0.001):
    return (RectangularRuleLeft(funct, h= h/2) + RectangularRuleRight(funct, h= h/2))/2

def TrapzoidRule(funct, bound = [0,5], h=0.0001):
    return (h/2)*( 2*np.sum(funct(np.linspace(bound[0]+h,bound[1]-h,int((bound[1]-bound[0])/h)-2)))+ funct(bound[0]) + funct(bound[1]) ) 


def TrapzoidRule2(funct, bound = [0,5], h=0.0001, n=10000):
    h = (bound[1]-bound[0])/n
    return (h/2)*( 2*np.sum(funct(np.linspace(bound[0]+h,bound[1]-h,int((bound[1]-bound[0])/h)-2)))+ funct(bound[0]) + funct(bound[1]) ) 


def MonteCarlo(funct, bound = [0,5], n = 5):
    # random_number = [ np.random.uniform(0,5) for x in range(int(bound[1]-bound[0]))]
    random_number= np.random.uniform(bound[0],bound[1],n)
    return np.mean(function(random_number))*(bound[1]-bound[0])


def SimpsonRule(funct, bound = [0,5], h=0.0001):
    boundterm= funct(bound[0])+funct(bound[1])
    # oddterm = [for x in range()] 
    # eventerm = 
    return (h/3)

def IntegralPerf(funct,InteFunct):
    start = time.time()
    r = InteFunct(funct)
    end = time.time() - start
    return r,end

def IntegralTest(funct, InteFunct, n=100):
    return np.mean([IntegralPerf(funct,InteFunct) for x in range(100)])
    

# MonteCarlo(function)
#
# print(IntegralPerf(function,RectangularRuleRight))
# print(IntegralPerf(function,RectangularRuleLeft))
# print(IntegralPerf(function,RectangularMix))
# print(IntegralPerf(function,TrapzoidRule))

print(IntegralTest(function,TrapzoidRule))
# print(IntegralPerf(function,TrapzoidRule2))
# print(IntegralPerf(function,MonteCarlo))
#
