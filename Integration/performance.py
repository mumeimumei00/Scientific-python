
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


def SimpsonRule(funct, bound = [0,5], n = 50000):
    h = (bound[1]-bound[0])/n
    boundterm= funct(bound[0])+funct(bound[1])
    oddterm = np.sum(4*funct(h*np.array([bound[0]+(2*x+1) for x in range(int(n/2))])))
    eventerm = np.sum(2*funct(h*np.array([bound[0]+(2*x) for x in range(1,int(n/2))])))
    return (h/3)*(boundterm+oddterm+eventerm)

def IntegralTime(funct,InteMethod,n= 50000):
    start = time.time()
    r = InteMethod(funct, n=n)
    end = time.time() - start
    return r,end

def IntegralPerf(funct,InteMethod,real, accuracy = 0.05):
    n = 2
    while (int(accuracy*(10**6)) <= int(abs(InteMethod(funct,n=n)/real -1)*(10**6))):
        n += 1 

    return InteMethod(funct,n=n), n

def IntegralTest(funct, InteMethod, real, Iteration=100, accuracy= 0.05, randfunc = False):
    result = 0
    time = 0
    result0, n = IntegralPerf(funct,InteMethod,real, accuracy=0.0001)
    for x in range(Iteration):
        r,end = IntegralTime(funct,InteMethod,n)
        result += r
        time += end
    

    if randfunc:
        return result0 , time/Iteration, n 
    else:
        return result/Iteration , time/Iteration, n 
    
#
real = 66.66666

print("Rectangular Rule Right:",IntegralTest(function,RectangularRuleRight, real))
print("Rectangular Rule Left:",IntegralTest(function,RectangularRuleLeft, real))
print("Rectangular Rule Mix:",IntegralTest(function,RectangularMix, real))
print("Trapzoidal Rule:",IntegralTest(function,TrapzoidRule, real))
print("Monte Carlo:",IntegralTest(function,MonteCarlo, real, randfunc=True))
print("Simpson Rule:",IntegralTest(function,SimpsonRule, real))

print(MonteCarlo(function,n=5))

# How to properly test function: Increase n sample until all function reach the same precision. Use n and time as 
