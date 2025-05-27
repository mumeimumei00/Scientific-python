from math import sqrt
def fib(n):
    a = 0
    b = 1
    if n == 0:
        return a
    else:
        for x in range(n):
            temp = a
            a = b
            b += temp
            print(b)
        return b

print(fib(5))

def fib2(n):
    return (1/sqrt(5))*(((1+sqrt(5))/2)**n -((1-sqrt(5))/2)**n)

print(fib2(5+1))
