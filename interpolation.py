import numpy as np
import random
import matplotlib.pyplot as plt
import time


def expTaylor(x,x0,nmax):
    t= 0
    for n in range(nmax+1):
        t = t + np.exp(x0)*(x-x0)**n / np.math.factorial(n)
    return t

expTaylor(1,0,10)

np.exp(1)

x_list = np.linspace(-5,5,101)
plt.plot(x_list,np.exp(x_list))
plt.scatter(x_list,expTaylor(x_list,0,2))
plt.scatter(x_list,expTaylor(x_list,0,5))
plt.scatter(x_list,expTaylor(x_list,0,10))



def sinTaylor(x,nmax):
    t= 0
    for n in range(nmax+1):
        t = t + (-1)**n * x**(2*n+1)/np.math.factorial(2*n+1)
    return t

x_list = np.linspace(-10,10,101)


plt.plot(x_list,np.sin(x_list))
plt.plot(x_list,sinTaylor(x_list,1),'red')
plt.plot(x_list,sinTaylor(x_list,6),'green')
# plt.scatter(x_list,sinTaylor(x_list,0,5))
# plt.scatter(x_list,sinTaylor(x_list,0,10))



(np.sin(1)-sinTaylor(1,3))/np.sin(1)

(np.exp(np.sin(1))-np.exp(sinTaylor(1,3)))/np.exp(np.sin(1))

### Derivatives
def sq(x):
    return x**2

def derivative(f,x,h):
    # f: Function
    # x: Argument for x
    # h: Stepsize
    return (f(x+h)-f(x))/h

derivative(sq,2,0.00001)


#Higher derivative

def nDerivative(f,x,h,n):
    # f: Function
    # x: Argument for x
    # h: Stepsize
    # n: nth Derivatives
    t = 0
    for k in range(n+1):
        t= t + ((-1)**(k+n))*f(x+k*h)*np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(n-k))
    return t / h**n

nDerivative(sq,2,0.00001,3)


# General function for Taylor series

def taylor(f,x,x0,nmax,h):
    # f: Function
    # x: Argument
    # x0: Argument at which the derivatives will be calculated
    # nmax: n at which the series will terminate
    # h: Stepsize
    t = 0
    for n in range(nmax+1):
        t = t + nDerivative(f,x0,h,n) * (x-x0)**n / np.math.factorial(n)
    return t

    

plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-4,4])
plt.ylim([-2,2])

x_list = np.linspace(-5,5,101)
plt.plot(x_list, np.sin(x_list))

plt.scatter(x_list, taylor(f = np.sin, x = x_list, x0 = 0, nmax = 5, h = 0.1))
plt.scatter(x_list, taylor(f = np.sin, x = x_list, x0 = 0, nmax = 5, h = 0.05),color = 'cyan')
plt.scatter(x_list, taylor(f = np.sin, x = x_list, x0 = 0, nmax = 5, h = 0.01),color = 'black')
plt.scatter(x_list, taylor(f = np.sin, x = x_list, x0 = 1, nmax = 5, h = 0.001), color = 'red')
plt.scatter(x_list, taylor(f = np.sin, x = x_list, x0 = 2, nmax = 5, h = 0.001), color = 'green')



def correctFunction(x):
    return 15 + 2.4*x - 0.5*x**2 - 0.35*x**3

npoints = 21
x_list = np.linspace(-5,5,npoints)
data0 = np.array([x_list,correctFunction(x_list)])

plt.xlabel('x')
plt.ylabel('y')


data = np.array([data0[0] + 0.25*(1-(2*np.random.rand(npoints))),
                 data0[1]+ 5*(1-(2*np.random.rand(npoints)))])

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(data[0], data[1])
plt.plot(data0[0], data0[1])

from scipy import interpolate

splineLinear0 = interpolate.interp1d(data0[0],data0[1], kind = 'linear')
# Piece wise connected dots
plt.xlabel('x')
plt.ylabel('y')
# plt.scatter(data[0], data[1])
plt.scatter(data0[0], data0[1])
plt.plot(data0[0], splineLinear0(data0[0]))






















