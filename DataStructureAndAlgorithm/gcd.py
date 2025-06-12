def gcd(a,b):
    a = abs(a)
    b = abs(b)
    if a == 0:
        return b
    if b == 0: 
        return a

    elif a == b:
        return a
    elif ((a%2== 0) and (b%2 == 0)):
        return gcd(a>>1,b>>1) << 1

    elif ((a%2== 0) ^ (b%2 == 0)):
        if (a%2 == 0): 
            return gcd(a>>1,b)
        elif (b%2 == 0):
            return gcd(a,b>>1)
    else:
        if a > b:
            return gcd((a-b)>>1,b)
        else: 
            return gcd((a-b)>>1,a)
print(gcd(48,18))
print(gcd(54,24))
print(gcd(101,10))
print(gcd(17,3))
print(gcd(10,6))

def test():
    print("Our function:", gcd(48,18) ," and the real answer is 6. So it is : ", gcd(48,18) == 6)
    print("Our function:", gcd(54,24) ," and the real answer is 6. So it is : ", gcd(54,24) == 6)
    print("Our function:", gcd(101,10) ," and the real answer is 1. So it is : ", gcd(101,10) == 1)
    print("Our function:", gcd(17,3) ," and the real answer is 1. So it is : ", gcd(17,3) == 1)
    print("Our function:", gcd(10,6) ," and the real answer is 2. So it is : ", gcd(10,6) == 2)
    print("Our function:", gcd(0,5) ," and the real answer is 5. So it is : ", gcd(0,5) == 5)
    print("Our function:", gcd(0,0) ," and the real answer is 0. So it is : ", gcd(0,0) == 0)
    print("Our function:", gcd(-4,6) ," and the real answer is 2. So it is : ", gcd(-4,6) == 2)
test()
