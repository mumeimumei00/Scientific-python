from gmpy2 import mpz,mpq,mpfr,powmod,gcd,invert
# mpz = multiple-precision integer, big int type
# mpq = rational number (fraction)
# mpfr = floating-point with arbitrary precsion

a = mpz(123456789123456789123456789)
b = mpz("123456789123456789123456789")
print(a+b)

x = mpz(2) ** 200
y = mpz(15)

print(x)
print(x.bit_length())

# change the base
print(y.digits(3))

# to be continued
