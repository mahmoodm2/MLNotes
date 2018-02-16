def gen_primes(n):
    p=[True for i in xrange(n+1)]   
    p[1]=False
    p[0] = False
    m= math.sqrt(n)
    for i in xrange(2, int(m)+1):
        # print(i)
        if p[i] == True:
            for k in xrange(i*i , n+1,i):
                # print(k)
                p[k]= False
                
    return [i for i in xrange(n+1) if p[i] == True]

###################################################################
def gcd( n,m):
    for i in xrange(min(n,m) , 0, -1 ):
        if m%i== 0 and n%i==0 :
            return i
###################################################################
def LCM( m ,n):
    return m * n / gcd_eff(m,n)
###################################################################
def gcd_eff(n,m):
    # print(m)
    if m==0 : return n
    else: return gcd_eff(m , n % m)


###################################################################
def addone(a, l=0):

    if l==0: return [1] + a

    if a[l-1] == 9:
        a[l-1] =0
        return addone(a, l-1)
    else:
        a[l-1] += 1
    return a

def  convert(number, base): # clculate value of number  in base
# assuming all chacaters are in 0-9 , a-f
    if base < 2 or (base > 10 and base !=16) : return -1
    power = 1
    res= 0 # result to eb returned
    for c in number[::-1]:
        digit = get_digit(c)
        res=  res + power * digit 
        power *= base
    return res
def get_digit(c):
    if c >= '0' and c <= '9': return c -'0'
    if c.upper() >= 'A' and c.upper() <= 'F': return c.upper() - 'A' +10
###################################################################
def get_min_integer(arr):
    arr = set(arr)
    
    for i in range(len(arr)):
        if i not in arr:
            return i

    return len(arr)

###################################################################        
def primefactors(maxnum):
    factors=[[] for _ in xrange(maxnum)]
    i=2
    while i < maxnum:

        if len(factors[i]) == 0 :
                
            for j in range( 2* i , maxnum , i):
                factors[j].append(i)
            
            factors[i].append(i)
        else:
            p=1
            for k in factors[i]:
                p *= k
            f =  factors[ i // p ]
            factors[i].extend(f)
            print( i , f)
        i+=1
    return factors   

###################################################################
def getRoot( num , n):
    # 1 < num ^ 1/n < n
    # initial guess = n/2, guess ^ n  - num <= eps
    # num:8 , n:3
    # ( 8,3,1,8) , m:4 ( ,1,4), m:2
    # ( 7,3,1,7) , m: 4 , ( ,1,4), m:2.5 , ( 1, 2.5), m:1.75 , ( ,1.75,2.5) : m:2.125
    def searchRoot(left , right):

        mid = left  + ( right - left ) / 2.0
        print(mid)
        err = mid**n - num
        if abs(err) <= eps:
            return mid

        if err > 0:
            return searchRoot( left , mid)
        else:
            return searchRoot( mid , right)

    eps = 1e-3
    if n ==1 :
        return num
    if num == 1:
            return 1
    if num < 0:
        return "-1"

    if num >1 : 
        return searchRoot( 1, num)
    else:
        return searchRoot(  0, 1)
###################################################################

def choose(n, k):
    if k == 0: return 1
    return (n * choose(n - 1, k - 1)) / k