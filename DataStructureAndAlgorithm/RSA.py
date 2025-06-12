import sympy as sy
import random

def primegenerator():
    exitcondition = True;
    while (exitcondition):
        p = random.randint(0,2097152)
        q = random.randint(0,2097152)
        if (sy.isprime(p) and sy.isprime(q) and p!=q):
            exitcondition = False


    return p,q


m = 8838

def RSA(m):
    p,q = primegenerator()
    n = p*q
    r = (p-1)*(q-1)
    e = 65537

    s,t,g = sy.gcdex(e,r)
    d = s%r
    return pow(m,e,n),d,n

encry, d,n = RSA(m)
print("this is the encrypted message: ", encry)
print("These are the keys: d:",d, "\n ", n)
dec = pow(encry,int(d),n)
print("\nThis is the decrypted message: ", dec)

