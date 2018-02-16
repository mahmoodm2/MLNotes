###################################################################
def insert_m_n(n,m, i ,j):
    # j >= i
    # prb: 5.1
    mask = (1  << ( j -i+1) ) -1 # a number with j-i ones
    mask <<= i # shifting mask to desired posiiton

    n= n & ( ~ mask)
    n = n | ( m << i)

    return n
###################################################################
def binary_represent(num):
    # prb: 5.2 : binary represenation of decimal numbers
    if num < 0 or num > 1 :
        return "Error"
    # num = 1/2 + 1/4 + 1/8 + ...

    binary_mul = 0.5
    binary =""
    while len(binary) < 32:
        if num > binary_mul :
            binary +="1"
            num -= binary_mul
        else:
            binary +="0"
    
        binary_mul /=2

    return "0."+binary
###################################################################
def max_min_binary(num):
    #prb : 5.3
    # if num=6 (00000110) -> min=(00000101)5 , max=  (00001001)
    mask =1
    ones=0
    max_len = 16
    while mask <= num:
        if (num & mask) > 0:
            ones +=1
        mask <<= 1
    min_bin = ( 1<< ones ) -1
    max_bin = min_bin << ( max_len - ones )

    return min_bin , max_bin
###################################################################