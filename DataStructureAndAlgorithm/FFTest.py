import cmath

def FFT(a,w):
    if len(a) == 1:
        return a
    else:
        array1 = FFT(a[0::2],w**2) # get the even elements
        array2 = FFT(a[1::2],w**2) # get the odd elements of the list
        print(array1, " and ", array2)
        r = [None] * len(a)
        for i in range(len(a)//2):
            r[i] = array1[i]+ (w**i)*array2[i]
            r[i+ len(a)//2] = array1[i] - (w**i)*array2[i]
        return r

print(FFT([1,1,1,1,1,1,1,1],cmath.exp(2j * cmath.pi / 8)))
