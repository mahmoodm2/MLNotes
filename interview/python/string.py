def compress_string(input):
    if len(input) == 0:
        return 0
    last= input[0]
    i=1
    count=1
    res= ""
    for i in range(1,len(input)):
    	if input[i] == last :
    		count +=1
    	else:
    		res += str(count)+ last
    		count=1
    		last = input[i]

    res += str(count)+ last
    if len(res) <= len(input): return res
    else: return input
def is_uniqe(a):
    # All characters are qunique in a letter

	cnt = [False for _ in xrange(256)]
	for c in a:
		if cnt[ord(c)-1]:
			return False
		else: cnt[ord(c)-1] =True
		
	return True	
###################################################################
def permutation(a, b, g):
    # Questions: White spaces, lower/upper case 
    # a= a.strip().lower()
    # b= b.strip().lower()

    # Removing  scpaces and garbage characters
    a=''.join(c.lower() for c in a if c not in g)
    b=''.join(c.lower() for c in b if c not in g)

    if len(a) != len(b):
        return False

    sa = sorted(a)
    sb = sorted(b)

    return sa==sb
###################################################################
def permutation_count( a, b , g):
    # Questions: Whaite Space, UniCOde, Uper/lowercase

    # Removing  scpaces and garbage characters
    a=''.join(c.lower() for c in a if c not in g)
    b=''.join(c.lower() for c in b if c not in g)

    cnt = [0]*256 # if unicode -> 655536

    for c in a:
        cnt[ord(c)-1] +=1

    for c in b:
        cnt[ord(c)-1 ] -=1
        if cnt[ord(c)-1 ] < 0:
            return False

    return True
###################################################################
def palindrom(a, g):
    # Questions: Whaite Space, UniCOde, Uper/lowercase

    a= ''.join(c.lower() for c in a if c not in g)

    l = len(a) -1 
    for i in xrange(l//2 + 1):
	    if a[i] != a[ l-i] : return False
    return True	
###################################################################
def bag_of_words(a):
    bag = {}
    words = a.split()
    for w in words:
        
        if bag.has_key(w):
            bag[w] =1 + bag[w]
        else:
            bag[w] =1
    return bag
###################################################################     
def is_rotation(a,b):
    # Question: lower/uppercase
    # a= string , b= rotated 

    s=a
    for i in xrange(len(a)):
        s= s[1:] +s[0]
        if s==b: return True
    return False
###################################################################
def is_rotation_eff(a,b):
    # Question: lower/uppercase
    # a= string , b= rotated 

    if len(a) != len(b): return False
    s= a+a
    
    if s.find(b) > -1: return True
    else: return False

###################################################################
def  is_balanced(a,g)    :
    # Questions: alphabet and NOne {[()]}
    g +=" "
    a=[c for c in a if c not in g]
    stk = []
    for c in a:
        if c in '{[(':
            stk.append(c)
        elif c in '}])':
            if len(stk) > 0:
                top = stk.pop()
                if top =='{' and c !='}': return False
                if top =='[' and c !=']': return False
                if top =='(' and c !=')': return False
            else:
                return False

    if len(stk) == 0 : return True
    else: return False
###################################################################
def keyboard(word): # presing  letters of a using a keyboard
    w=5
    alph = 26
    # L=-1 , R=+1 , T=-1 , B=-1
    src=[0,0]
    word= word.upper()
    last=65
    keys=""
    for c in word:
        dist = ord(c) -65 
        dx = dist % w 
        dy= dist // w

        x= dx - src[0]
        y = dy -src[1]

        if x==y==0: keys +=" OK"
        else:
            if x < 0 :
                keys +=''.join(' LEFT' for _ in range(abs(x)))
            if x > 0 :
                keys +=''.join(' RIGHT' for _ in range(x))
            if y < 0 :
                keys +=''.join(' UP' for _ in range(abs(y)))
            if y > 0 :
                keys +=''.join(' DOWN' for _ in range(y))

            keys +=" OK"

            src=[dx,dy]
    return keys  
def permut(input):
    res=[]
    if len(input)== 0: return [""]
    if len(input) == 2:
        return [input[0]+input[1], input[1]+input[0]]
    else:        
        tmp= permut(input[1:])
        c = input[0]
        for t in tmp:
            for i in xrange(len(t)):
                res.append(t[:i+1]+c + t[i+1:])
            res.append( c + t)
    return res
###################################################################
def lengthOfShortesSubstring(s):
        """
        shortest uniq substring
        :type s: str
        :rtype: int
        """
        output =''
        #char_list=set([c for c in s])
        
        char_cnt ={}
        for c in s:            
                char_cnt[c] =0
        
        uniq_cnt = 0
        
        head = 0
        tail = 0
        min_len = len(s)
        while tail < len(s):
            c= s[tail]
            if c in char_cnt:
                if char_cnt[c] == 0:
                    uniq_cnt +=1
                    
                char_cnt[c] +=1
            
            while uniq_cnt == len(char_cnt):
                
                substr = s[head:tail+1]
                print( substr)
                len_str = len(substr)
                if len_str == len(char_cnt):
                        return len_str
                else:
                    if len_str < min_len:
                        output= substr
                        min_len = len_str
                    
                c = s[head] 
                if c in char_cnt:
                    char_cnt[c] -=1
                    if char_cnt[c] == 0:
                        uniq_cnt -=1
                head+=1
                    
            tail +=1        
                
        return min_len 
###################################################################
def lengthOfLongestUniqSubstring(s): 
    #Longest Uniq substring
    # abcddf --> abcd
    #     t
    # h
    # maxln= 4
    #vis :d f

    head = tail = 0
    visited= {}
    maxlen = 0
    maxhead = maxtail = 0
    for tail in xrange( len(s)):
        c= s[tail]
        if c in visited:
            #while head  < tail and s[head] in visited:
            #    head+=1
            head = visited.pop(c) +1
          
        visited[c] = tail
        if (tail - head + 1) > maxlen:
            maxlen = max( maxlen, tail - head + 1)
            maxhead = head
            maxtail = tail
  
    return maxlen , s[maxhead : maxtail+1]

##############################################################

def lengthOfLongestSubstring(s):
    """
    # longest substring of uniq characters
    :type s: str
    :rtype: int
    """
    max_len = 0
    head = tail =0
    visited = []
    
    sublen= 0
    while tail < len(s):
        c = s[tail]
        if c  in visited:                    
            while c in  visited:
                visited.remove(s[head])
                head +=1
                sublen -=1

        visited.append(c)               
        sublen +=1      
            
        if sublen > max_len:
                max_len = sublen
        
        tail +=1
            
    return max_len  
#######################################
def longestPalindrome(s): # Old version
        """
        :type s: str
        :rtype: str
        """        
        head= 0
        tail = len(s)
        maxpal =""
        pal =""
        
        while head < len(s) and tail - head > len(maxpal):
            
            while tail <= len(s) and (tail - head ) > len(maxpal): 
                if s[head] == s[tail-1]:
                    pal =is_palindrom( s[head:tail])
                    if pal != "" :

                        if len(pal) > len(maxpal):
                            maxpal = pal
                            break

                tail -=1
            
            head +=1
            tail = len(s)

        return maxpal    
###################################################################       
def is_palindrom(s):
        
        ls= len(s)  
        for i in range(ls // 2 ):
            if s[i] != s[ls - i - 1]:
                return ""
        return s  
################################################################### 

def convert(s, numRows): # zigzag convesion
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        n = numRows
        rows ={}
        
        for i in range(n):
            rows[i+1] =''
        
        dirc =1
        row=0
        if n==1:
            return s
        for i in range(len(s)):
            
            row += dirc
            rows[row] += s[i]
            
            if (row == n ) or (row ==1 and i> 0):
                dirc = 0 - dirc
            
        output=""
        for i in range(n) :
            output += rows[i+1]
            #output += ''.join([c for c in rows[i+1]])
            
        return output  
###################################################################
def longestPalindrom(s):
    # generating palndrom of at position 0 ,1,2,3,...
    # xabba
    def get_palindrom( s, left , right):
        
        while ( left >= 0 and right < len(s)  and  s[left] == s[right]):
            left -=1
            right +=1
        
        return right - left -1 

    maxlen= 0
    if len(s) <= 1: return s, len(s)
    i=0
    maxstr =""
    head= tail =0
    while i < len(s): 
        # i center of palindrom

        l1 = get_palindrom( s, i,i+1)
        l2 = get_palindrom( s, i,i )

        tmp = max(l1,l2)

        if tmp > (tail - head):
            head = i - tmp /2 + 1
            tail =  i + tmp/2

        # if len(s1) > len(s2):
        #     substr = s1
        # else:
        #     substr= s2
        # if len(substr) > maxlen:
        #     maxlen = max(maxlen , len(substr))
        #     maxstr= substr

        i+=1

    return head , tail , s[head: tail+1] , (tail - head +1)

###################################################################  
def  removeStr( s , chrlist):

    s = list(s)
    chrset1 = set(chrlist)
    flags = [ False for _ in xrange(128)]

    for i in xrange(len(chrlist)):
        chrset2[ ord(chrlist[i]) ] =  True
           
    des =0
    length = len(s) 
    for i in xrange(length):
        if not flags[ ord(s[i])] == True:            
            s[des] = s[i]
            des +=1


    # for i in xrange(len(s) -1 ):
    #     if s[i] in chrset1:            
    #         s[i] = s[i+1]


    return ''.join( c for c in s if flags[ord(c)] == False)

###################################################################
def firstNoneDuplicate(s):
    chars = [0 for _ in xrange(256)]
    # chrcnt = {}
    # for c in s:
    #     if not c in chrcnt:
    #         chrcnt[c] =False
    #     else: 
    #         chrcnt[c] =True

    for i in xrange(len(s)):
        chars[i] +=1
    for c in s:
        if chars[ord(c)] == 1:
            return c
    return "EMPTY"
###################################################################

###################################################################
def lengthLongestPath(input):
        """
        :type input: str
        :rtype: int
        """
        currLev =0
        maxlen = 0
        
        input="dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"
        s= list(input)
        pathlen= [0 for _ in xrange(len(s))]
        # dir/n/tfoler1/n/t/tfoler2/
        #        i
        # h:7 , t:2 , l:1  , substr:3
        head= tail= 0
        i=0
        while i < len(s)-1:
            if s[i:i+1]=="\n":
                tail = i-1
                print( s["head:tail+1"])                
                substr= tail - head +1                 
                if level ==0 :
                    pathlen[level] = substr
                    maxlen= susbtr
                    
                elif pathlen[level -1 ] + susbtr +1 > pathlen[level]:
                    pathlen[level] = pathlen[level -1 ] + susbtr + 1
                    maxlen = max(maxlen, pathlen[level] )                    
                i+=2
                
            elif s[i:i+1]=="\t":                 
                level =0
                while s[i:i+1] =="\t":
                    level+=1
                    i+=2
                                
                head= i
                
            i+=1   
        
        return maxlen
###################################################################

