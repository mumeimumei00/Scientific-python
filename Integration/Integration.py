import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

def function(x):
    return x*x+5

# RiemannIntegral
# Don't forget to always multiply outside, not inside.
def RectangularRuleRight(funct,bound = [0,5],n=50000):
    h = (bound[1]-bound[0])/n
    return h*np.sum(funct(np.linspace(bound[0]+h,bound[1],int((bound[1]-bound[0])/h))))

def RectangularRuleLeft(funct,bound = [0,5],n=50000):
    h = (bound[1]-bound[0])/n
    return h*np.sum(funct(np.linspace(bound[0],bound[1]-h,int((bound[1]-bound[0])/h))))

def RectangularMix(funct,bound = [0,5], n= 50000):
    return (RectangularRuleLeft(funct, n= n/2) + RectangularRuleRight(funct, n= n/2))/2

def TrapzoidRule(funct, bound = [0,5], n=50000):
    h = (bound[1]-bound[0])/n
    return (h/2)*( 2*np.sum(funct(np.linspace(bound[0]+h,bound[1]-h,int((bound[1]-bound[0])/h)-2)))+ funct(bound[0]) + funct(bound[1]) ) 


def MonteCarlo(funct, bound = [0,5], n = 50000):
    # random_number = [ np.random.uniform(0,5) for x in range(int(bound[1]-bound[0]))]
    random_number= np.random.uniform(bound[0],bound[1],n)
    return np.mean(function(random_number))*(bound[1]-bound[0])


def SimpsonRule(funct, bound = [0,5], n = 1):
    h = (bound[1]-bound[0])/n
    boundterm= funct(bound[0])+funct(bound[1])
    oddterm = np.sum(4*funct(h*np.array([bound[0]+(2*x+1) for x in range(int(n/2))])))
    eventerm = np.sum(2*funct(h*np.array([bound[0]+(2*x) for x in range(1,int(n/2))])))
    return (h/3)*(boundterm+oddterm+eventerm)

def IntegralTime(funct,InteMethod):
    start = time.time()
    r = InteMethod(funct)
    end = time.time() - start
    return r,end

# def IntegralPerf(funct,InteMethod,accuracy = 0.95):


def IntegralTest(funct, InteMethod, n=100):
    result = 0
    time = 0
    for x in range(n):
        r,end = IntegralTime(funct,InteMethod)
        result += r
        time += end
    return result/n , time/n 
    

print("Rectangular Rule Right:",IntegralTest(function,RectangularRuleRight))
print("Rectangular Rule Left:",IntegralTest(function,RectangularRuleLeft))
print("Rectangular Rule Mix:",IntegralTest(function,RectangularMix))
print("Trapzoidal Rule:",IntegralTest(function,TrapzoidRule))
print("Monte Carlo:",IntegralTest(function,MonteCarlo))
print("Simpson Rule:",IntegralTest(function,SimpsonRule))

