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

print(fib(100))
