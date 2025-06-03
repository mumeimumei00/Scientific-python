# This code return the a list of number in the given base
def base(N: int,base:int=10)-> list[int]:
    assert (N>= 0 and base>=2)
    digits = []
    while N > 0:
        digits += [N % base]
        N //= base
    return digits[::-1]

print(base(15,3))


def base_addition(n1: list[int], n2: list[int], base:int=10)-> list[int]:
    assert all([(x//base < 1) for x in n1]) and all([(x//base < 1) for x in n2]), "invalid integer"
    max_len = max(len(n1),len(n2))
    carry = 0
    digits = []
    for i in range(max_len):
        if i < len(n1):
            a = n1[i]
        else:
            a = 0
        if i < len(n2):
            b = n2[i]
        else:
            b = 0
                       
        
        digits += [(a+b+carry)%base]
        carry = (a+b+carry)//base
        
        if i == max_len:
            digits += [carry]
    return digits
    
print(base_addition([9,3],[5,5,1],10))
