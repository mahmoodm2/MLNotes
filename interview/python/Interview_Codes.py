# Google inteview
import random
import re
import math
import timeit
import sys

def testme(a):
    print(a)
################################################################
def move(from_p, with_p, to_p, h): # hanoi
    # Question :, if order is reveresed and the orign pole is descending the soltion is same
    if h >=1:
        move(from_p,  to_p, with_p, h-1 )
        print("Move from %s to %s"%(from_p, to_p))
        move(with_p , from_p, to_p, h-1 )
################################################################
# Queue using two stacks
def enq( s1,s2, input):
    s1.append(input)

def deq(s1,s2):
    # Dequeue from stack 2
    if len(s2) > 0:
        return s2.pop()
    else:
        # pop from s2 and push to s1 in reverse order
        while len(s1) > 0:
            s2.append(s1.pop())

        # pop from s2 which is the first item pushed in s1
        if len(s2) > 0:
            return s2.pop()
        return "EMPTY"
################################################################
# Sorted Stack
def so_push(s1, s2, a):
    # Check s1.top
    if len(s1) == 0:
        s1.append(a)
        return
    if s1[len(s1)-1] <= a:
        s1.append(a)
    else:
        top= s1[len(s1)-1] 
        while top > a:
            # pop from s1 and push to s2 ( sorted)
            s2.append(s1.pop())
            top = s1[len(s1)-1] 
        s1.append(a)
        # pop from s2 and push to s1 (sorted)
        while len(s2) > 0:
            s1.append(s2.pop())

def so_pop(s1):
    if len(s1) > 0:
        return s1.pop()
    else:
        return "EMPTY"
################################################################
class Node:
    def __init__(self, val):
        self.v=val
        self.left= None
        self.right=None
        self.parent= None
class Tree:
    def __init__(self,val):
        self.root= Node(val)
        

    def is_balanced(self):
        # t is complete and stored in an array( from 0)
        # parent(i) = (i-1) //2, left(i) = i*2 +1, right(i) = i*2+2
        # height(t) = log(len(t))
        dis= abs( self.height(self.root.left) - self.height(self.root.right))

        if dis <= 1 : return True
        else: return False
   
    def height(self, node):
        if node == None : return 0
        return self.height(node.left) + self.height(node.right) +1

    def add(self, val, node=None): # to build a BST 
        
        if node== None:
            res= self.add(val, self.root) # starting from root node
        else:
            if val < node.v:
                if node.left== None:
                    res= Node(val)
                    res.parent= node
                    node.left= res
                    
                else:
                    res = self.add(val, node.left)
            else:
                if node.right== None:
                    res= Node(val)
                    res.parent= node
                    node.right= res
                    
                else:
                    res= self.add(val, node.right)
        return res

    def inorder(self , node=None):
        if node== None: node= self.root
        
        if node.left!= None : self.inorder(node.left)   
        print(node.v)
        if node.right!= None : self.inorder(node.right) 

def print_tree(node=None):
        if node== None: return
        
        if node.left!= None : print_tree(node.left)   
        print(node.v)
        if node.right!= None : print_tree(node.right) 
######################
# Graph : dict{ node : adjacents list}, visited= set(nodes)
def dfs( g ,node, visited):
    print(node) # visit node
    visited.add(node)
    for n in g[node]: # adjacents
        if n not in visited:
            dfs( g, n , visited)
    return visited
############################################
def dfs_stack(g, start):
    # g: graph as dict { node: adjacnts}, start : node , visited:list
    visited =[]
    stack= [start]     
    while stack:
        node= stack.pop()
        if node not in visited:
            visited.append(node)
            
            stack.extend( g[node]) # adding adjacents of node

    return visited
 ############################################       
def bfs(g,start):
    # bfs can be used for shortest path
    # graph : dict{ node: adjacents}
    visited=[]
    adj=[start] # queue of adjacents
    
    while adj_q:
        node = adj_q.pop(0)

        if node not in visited:
            visited.append(node)

            adj.extend(g[node]) # adding adjacents og node to Que

    return visited
############################################
def found_rout(g,start,end):
    # graph : dict{ node: adjacents}
    visited=[]
    adj=[start] # queue of adjacents
    
    while adj:
        node = adj.pop(0)
        if node == end: return True
        if node not in visited:
            visited.append(node)

            adj.extend(g[node]) # adding adjacents og node to Que

    return False
########################################################################################
def build_bst(data): # build BST out of a sorted array
    root = (len(data) -1 )//2
    root_node= Node(data[root])

    node= root_node
    for d in reversed(data[:root]):
        tmp = Node(d)
        node.left = tmp
        node= tmp

    node= root_node
    for d in data[root+1:]:
        tmp = Node(d)
        node.right = tmp
        node= tmp
    return root_node
############################################
def build_lists(root , lists , level): # lsist of nodes in each level in tree
    # root: node of tree, lists : list of nodes at level 
    # lists: dict { level: nodes}
    if root== None: return
    if not level in lists:
        lists[level]= []

    lists[level].append(root.v)

    build_lists( root.left, lists , level+1)
    build_lists( root.right, lists , level+1)

    return
############################################
def is_bst(root, last): 
    # prb: 4.5
    # building a sorted array or last value and then check the order of  values
    # root : Node{ v, left, right}
    if root== None: return True
   
    res= True
    if not is_bst(root.left, last): return False

    if root.v < last : return False
    else: last = root.v

    if not is_bst( root.right, last): return False

    # if root.left != None:
    #     if root.left.v < v:
    #         res = res and is_bst(root.left)
    #     else: res= False
    # if root.right != None:
    #     if root.right.v > v:
    #         res = res and is_bst(root.left)
    #     else: res= False

    return res
############################################
def bst_next(node):
    # prb: 4.6
    # Node is a node in a BST tree, assuming has access to its parent
    if node.right != None:
        node= node.right
        while node.left:
            node= node.left
        return node
    else: #until node is in right subtree
        while node.parent:
            if node.parent.right== node:
                node= node.parent
            else: break
          
    return node.parent
############################################
def is_bst( root):
    #MM: Second round

    if root == None: return True
    # 6 10 15 17
    # min of right tree
    if root.right :
        root= root.right
        while root.left: # going down of tree to find min
            root= root.left 
        return root
    else: 
        while root.parent: # clibming up the tree to find the next bigger
            if root.parent.right == root:
                root = root.parent  
     
    return root.parent
    

def common_ancs(root, p , q):
    # prb:4.7
    if root== None : return None

    if p==q: return p

    if root == p or root == q : return root

    is_p_left = is_child(root.left , p)
    is_q_left = is_child( root.left, q)

    if is_p_left != is_q_left: # in different branches 
        return root
    if is_p_left:
        return common_ancs(root.left, p, q)
    else:
        return common_ancs(root.right, p, q)

def is_child( root, p):
    if  root == None : return False
    if root == p : return True

    return is_child( root.left, p) or is_child(root.right, p)
############################################
def is_subtree( t1, t2):
    # if the root of t2 is in t1 then match the left and right subtrees
    # prb: 4.8
    if t1==None: return False

    if t1.v==t2.v:
        return match_tree(t1,t2)
    else:
        return is_subtree(t1.left, t2) or is_subtree(t1.right, t2)

def match_tree(t1,t2):
    if t1==None and t2==None : return True
    if t1==None or t2==None: return False

    if t1.v != t2.v : return False
    else:
        return match_tree(t1.left , t2.left) or match_tree(t1.right, t2.right)
def init(num_day):
    """ Initilizaing the flower array """
    flw_list=random.sample(range(1 , num_day+1), num_day)
    print("fdf")
    print (flw_list)

def camlcase_snakecase(input):
    r1 = re.sub('(.)([A-Z][a-z]+)',r'\1_\2', input)
    print(r1)
    return re.sub('([a-z0-9])([A-Z])',r'\1_\2',r1).lower()

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
def find( pat, input):
    for i in range( len(input)- len(pat)+1):
        found= True
        for j in range(len(pat)):
            if input[i+j] != pat[j]:
                found= False
                break
        if found: #Found
            print("FOund at %d" %(i))
###################################################################
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

###################################################################
def gcd_eff(n,m):
    # print(m)
    if m==0 : return n
    else: return gcd_eff(m , n % m)
###################################################################
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
def addone(a, l=0):

    if l==0: return [1] + a

    if a[l-1] == 9:
        a[l-1] =0
        return addone(a, l-1)
    else:
        a[l-1] += 1
    return a
###################################################################
def subset(arr, total):
    print(arr)
    idx = [e for e in arr if e==total]
    if len(idx) == 1 : return idx
    
    if len(arr) > 1:
        fe= arr[0]
        arr.remove(fe)
        re = subset(arr, total - fe)
    else:
        return arr[0]
    return re.append(fe) 
###########################################
def equal_subsets(arr):

    ts = sum(arr) 
    if ts % 2 != 0: return False
    ts = ts //2
    print(ts)

    s1= subset(arr , ts)

    s2=[e for e in arr if e not in s1]
    if sum(s2) == ts:
        return s1,s2
    else: return False
###################################################################    
def find_busiest_period(data):
    # CPU = O(n), mem = O(1)
    # data is sorted ascending
    cnt=0
    last_date=data[0][0]   
    m_date=0
    max_cnt=0
    for d in data:
    
        if d[0] == last_date:
            if d[2] == 0:
                cnt -=d[1]
            else:
                cnt +=d[1]
        else:
        
            if cnt > max_cnt:
                m_date = last_date
                max_cnt = cnt
       
            last_date = d[0]
            cnt= d[1]     
            if d[2] == 0:
                cnt *=-1

    if cnt > max_cnt:
        m_date = last_date 

    return m_date   
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
###################################################################  
def push(stk_num, val):
    
    if ( stack_pointers[stk_num] >= stack_size -1 ):
        return "Full!! Exception"
    else:
        stack_pointers[stk_num]+=1
        index= get_index(stk_num)
        stack[index] = val
###################################################################
def pop(stk_num):
    if stack_pointers[stk_num]== -1 :
        return "Empty!!"

    index= get_index(stk_num)
    res= stack[index]
    stack[index] = 0
    stack_pointers[stk_num] -=1
    
    return res
###################################################################
def get_index(stk_num):

    return stk_num*stack_size + stack_pointers[stk_num]
###################################################################
# Stack to keep track of min values
def push_min(val):
    if ( val < get_min()):
        stk2.append(val)
    stack.append(val)

def pop_min():
    val = stack.pop()
    min = get_min()
    if val == min:
        min = stk2.pop()

    return val,min

def get_min():
    
    if len(stk2) == 0:
        return sys.maxsize
    else:
        return stk2[len(stk2)-1]
###################################################################
# l = permute( inpu[1:])
# lp= []
# for p in l: # bc --> abc, bac ,bca
#     for i in xrange(len(p)):
#         lp.append(p[:i] + s[i]  + p[i:])
#     lp.append(p + s[i] )
# return lp



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

'''
public String longestPalindrome(String s) {
    int start = 0, end = 0;
    for (int i = 0; i < s.length(); i++) {
        int len1 = expandAroundCenter(s, i, i);
        int len2 = expandAroundCenter(s, i, i + 1);
        int len = Math.max(len1, len2);
        if (len > end - start) {
            start = i - (len - 1) / 2;
            end = i + len / 2;
        }
    }
    return s.substring(start, end + 1);
}

private int expandAroundCenter(String s, int left, int right) {
    int L = left, R = right;
    while (L >= 0 && R < s.length() && s.charAt(L) == s.charAt(R)) {
        L--;
        R++;
    }
    return R - L - 1;
}
'''
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
def largestSubArray( arr):
    # sub array with maximum sum of elements
    head = tail = 0
    maxsum = 0

    largest = []
    # 3 100 -25 15 12 -16 10 17
    # 3 103  -15 15 27 11 21 38
    # h:4  t:2
    largets[0] = arr[0]
    for i in xrange(1,len(arr)):
        largest[i] = arr[i]

        if largest[i] < 0:
            tmp  = i # temprary head of sub array
            
        elif largest[i-1] > 0:
            largest[i] += largest[ i + 1]
        
        if largest[i] > maxsum:
            maxsum = largest[i]
            tail = i
            head= tmp

    
    return head, tail, maxsum
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
#2, 8 . -> low + (high - low) / 2
def search( arr , x ,left , right):

    if left >= right: return False


    mid = left + (right - left)//2
    mid = ( left + right ) // 2 #note
    if arr[mid] == x : return True

    if x > arr[mid]:
        return search ( arr , x , left , mid)
    else :
        return search( arr, x,  mid+1 , right)
###################################################################
'''
        that question - bots
        q's question about cell ##'s
        find 3 question
    */

    /*
        email MetaData:
            - sender  
            - reciever
            - timestamp
            
           BACK :  4 3 2 1 1 2 2 2 1 1 4 5 6 2 4 0 3 2 
                --> . --->


        PHONE NUMBERS STORED IN STRINGS
        you have 100 million phone numbers
        youw want to see if a specific number exists in the given set


        123-456-7980


        find 3

        input: [1, 3, 4, 7, 3, -4, 6, -8, 33, -5, 9 ]
        K: 1
        x+y = k , x , y=k- x
        return true or false if there are any 3 elements in "input" that sum
        to the target value "K"

        arr = set ([1, 3, 4, 7, 3, -4, 6, -8, 33, -5, 9 ])
        -8, -5, -4 , 1,3,3,4,6, 7,9,33 --> -8, 6, 3
        1, 0 , 3, -3

        for m in arr:  m + n + x =k
          for n in arr:
            if ( k - n -m ) in arr:
                output.append( [ n, m ,k-n -m])
       
    
'''
#################################################################################
def threeSumSmaller(self, nums, target):

    # nums : -1 2 0 5 -5 6 target : 1 , ans: -5 -1 0, -5 -1 2, -5 -1 5, -5 -1 6, -5 0 2 -5 0 5, -1 0 2
    
    nums= sorted(nums)
    ans =0
    left  = 0 
    
    for i in range( 0 ,len(nums)-2):
        right = len(nums) -1 
        left =i+1
        newtarget = target - nums[i]
        while right - left > 0: # 
            # n1 + n2 + n3 < target--  n3 < target - ( n1 + n2)
            # while right >0 and nums[right] == nums[right -1]:
            #      right -=1
            x= ( nums[left] + nums[right])
            if x < newtarget :
                ans += (right - left )
                left +=1
            else:
                right -=1
        
    return ans  
###################################################################
def kEmptySlots(self, flowers, k):
        """
        :type flowers: List[int]
        :type k: int
        :rtype: int        
         days:  1 5 2 6 4 3 , k=2 , t:0 , vis: 1,5 
         slots: 1 6 3 5 2 4
         slots :* *     *
        """        
        # [3,9,2,8,1,6,10,5,4,7] ,f: 2  t:     vis:3,9
        #flowers = [6,5,8,9,7,1,10,2,3,4]
       
        days= [0] * len(flowers)
        
        for day, slot in enumerate(flowers):
            days[slot-1] = day+1
        #print(days)
        head , tail = 0 , k+1
        i = head
        day = float('inf')
        while tail < len(days): 
            valid = True
            #print(days[head:tail+1])
            for i in range(head+1, tail): # validating elem of sliding windows            
                if days[i ] < days[head]  or days[i] < days[tail] : # invalid                   
                    
                    # head = i # updating slidign window range
                    # tail = head + k +1
                    valid = False
                    break
            
            if valid:                 
                    day = min (day , max( days[head] , days[tail]))
                    #print(day)                
            head = i # updating slidign window range
            tail = head + k+1
            
        return day if day != float('inf') else -1 
##############################
def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        # nums= sorted(nums)
        # for i in range( 1, len(nums)-1,2):
        #     nums[i] , nums[i+1] = nums[i+1] , nums[i]
        
        flag =1 # 1: Asc , 0 :Desc
        for i in range( len(nums)-1):
            if flag: # expect Asc if not swap
                if ( nums[i] > nums[i+1]): # Desc --> sawp
                    nums[i], nums[i+1] = nums[i+1] , nums[i]
            else: # expect Desc if not swap
                if ( nums[i] < nums[i+1]): # Asc --> sawp
                    nums[i], nums[i+1] = nums[i+1] , nums[i]
            flag =not flag
            
        print(nums)
#######################
def largestSmallerkey(root):
    prev_num=root.val

    while root :
        if target > root.val : 
            prev = root           
            root = root.right            
           
        else: 
            root = root.left          
           
    return prev
#######################   
def merge2sortedarray(arr1 , arr2):
    arr1 = sorted(arr1)
    arr2 = sorted(arr2)

    tail1 =len(arr1)-1
    tail2 =len(arr2)-1

    arr1.extend( [0] * len(arr2))
    

    while tail1 >=0  and tail2 >=0 :
        if arr1[tail1] > arr2[tail2]:
            arr1[tail1 + tail2 + 1 ] = arr1[tail1]
            tail1 -=1
        else:
            arr1[tail1 + tail2 + 1] = arr2[tail2]
            tail2 -=1

    while tail2 >=0:
        arr1[tail2] = arr2[tail2]
        tail2 -=1
####################################################
def cyclic(g):
    """
    Return True if the directed graph g has a cycle.
    g must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices. For example:

    cyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    cyclic({1: (2,), 2: (3,), 3: (4,)})
    False
    """
    path = set()
    visited = set()
    def visit(vertex):
        if vertex in visited:
            return False
        visited.add(vertex)
        path.add(vertex)
        for neighbour in g.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False

    return any(visit(v) for v in g)
#######################
def is_cyclic(g):
    g = {1:[2,3], 2:[4], 3:[4], 4:[6], 6:[5]}
    visited = set()
    path = [object()]
    path_set = set(path)
    stack = [iter(g)]
    while stack:
        for v in stack[-1]:
            if v in path_set:
                return True
            elif v not in visited:
                visited.add(v)
                path.append(v)
                path_set.add(v)
                stack.append(iter(g.get(v, ())))
                break
        else:
            path_set.remove(path.pop())
            stack.pop()
    print("====")
    print( path, pathset, visited, stack)
    return False
###############################################
def find_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not graph.has_key(start):
            return None
        for node in graph[start]:
            if node not in path:
                newpath = find_path(graph, node, end, path)
                if newpath: return newpath
        return None
###############################################    
def find_all_paths(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if not graph.has_key(start):
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths
###############################################
def find_shortest_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not graph.has_key(start):
            return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = find_shortest_path(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest
   
###############################################
def diameter(self):
        """ calculates the diameter of the graph """
        
        v = self.vertices() 
        pairs = [ (v[i],v[j]) for i in range(len(v)-1) for j in range(i+1, len(v))]
        smallest_paths = []
        for (s,e) in pairs:
            paths = self.find_all_paths(s,e)
            smallest = sorted(paths, key=len)[0]
            smallest_paths.append(smallest)

        smallest_paths.sort(key=len)

        # longest path is at the end of list, 
        # i.e. diameter corresponds to the length of this path
        diameter = len(smallest_paths[-1]) - 1
        return diameter
#####################
def is_connected(self, 
                     vertices_encountered = None, 
                     start_vertex=None):
        """ determines if the graph is connected """
        if vertices_encountered is None:
            vertices_encountered = set()
        gdict = self.__graph_dict        
        vertices = list(gdict.keys()) # "list" necessary in Python 3 
        if not start_vertex:
            # chosse a vertex from graph as a starting point
            start_vertex = vertices[0]
        vertices_encountered.add(start_vertex)
        if len(vertices_encountered) != len(vertices):
            for vertex in gdict[start_vertex]:
                if vertex not in vertices_encountered:
                    if self.is_connected(vertices_encountered, vertex):
                        return True
        else:
            return True
        return False
'''
// Function to sort arr[] of size n using bucket sort
void bucketSort(float arr[], int n)
{
    // 1) Create n empty buckets
    vector<float> b[n];
    
    // 2) Put array elements in different buckets
    for (int i=0; i<n; i++)
    {
       int bi = n*arr[i]; // Index in bucket
       b[bi].push_back(arr[i]);
    }
 
    // 3) Sort individual buckets
    for (int i=0; i<n; i++)
       sort(b[i].begin(), b[i].end());
 
    // 4) Concatenate all buckets into arr[]
    int index = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < b[i].size(); j++)
          arr[index++] = b[i][j];
}
'''
# Function to do insertion sort
def insertionSort(arr):
 
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):
 
        key = arr[i]
 
        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i-1
        while j >=0 and key < arr[j] :
                arr[j+1] = arr[j]
                j -= 1
        arr[j+1] = key
###############################
def dijsktra1(graph, initial):
  visited = {initial: 0}
  path = {}

  nodes = set(graph.nodes)

  while nodes: 
    min_node = None
    for node in nodes:
      if node in visited:
        if min_node is None:
          min_node = node
        elif visited[node] < visited[min_node]:
          min_node = node

    if min_node is None:
      break

    nodes.remove(min_node)
    current_weight = visited[min_node]

    for edge in graph.edges[min_node]:
      weight = current_weight + graph.distance[(min_node, edge)]
      if edge not in visited or weight < visited[edge]:
        visited[edge] = weight
        path[edge] = min_node

  return visited, path
###################
def dijkstra2(graph, start):
  # initializations
  S = set()
 
  # delta represents the length shortest distance paths from start -> v, for v in delta. 
  # We initialize it so that every vertex has a path of infinity (this line will break if you run python 2)
  delta = dict.fromkeys(list(graph.vertices), math.inf)
  previous = dict.fromkeys(list(graph.vertices), None)
 
  # then we set the path length of the start vertex to 0
  delta[start] = 0
 
  # while there exists a vertex v not in S
  while S != graph.vertices:
    # let v be the closest vertex that has not been visited...it will begin at 'start'
    v = min((set(delta.keys()) - S), key=delta.get)
 
    # for each neighbor of v not in S
    for neighbor in set(graph.edges[v]) - S:
      new_path = delta[v] + graph.weights[v,neighbor]
 
      # is the new path from neighbor through 
      if new_path < delta[neighbor]:
        # since it's optimal, update the shortest path for neighbor
        delta[neighbor] = new_path
 
        # set the previous vertex of neighbor to v
        previous[neighbor] = v
    S.add(v)
		
  return (delta, previous)
#####################

def choose(n, k):
    if k == 0: return 1
    return (n * choose(n - 1, k - 1)) / k
#####################
def topologicalSortUtil(self,v,visited,stack):
 
        # Mark the current node as visited.
        visited[v] = True
 
        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i,visited,stack)
 
        # Push current vertex to stack which stores result
        stack.insert(0,v)
 
    # The function to do Topological Sort. It uses recursive 
    # topologicalSortUtil()
def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = [False]*self.V
        stack =[]
 
        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i,visited,stack)
 
        # Print contents of stack
        print stack
##################################################
# public List<Integer> closestKValues(TreeNode root, double target, int k) {
#   List<Integer> res = new ArrayList<>();

#   Stack<Integer> s1 = new Stack<>(); // predecessors
#   Stack<Integer> s2 = new Stack<>(); // successors

#   inorder(root, target, false, s1);
#   inorder(root, target, true, s2);
  
#   while (k-- > 0) {
#     if (s1.isEmpty())
#       res.add(s2.pop());
#     else if (s2.isEmpty())
#       res.add(s1.pop());
#     else if (Math.abs(s1.peek() - target) < Math.abs(s2.peek() - target))
#       res.add(s1.pop());
#     else
#       res.add(s2.pop());
#   }
  
#   return res;
# }

# // inorder traversal

# void inorder(TreeNode root, double target, boolean reverse, Stack<Integer> stack) {
#   if (root == null) return;

#   inorder(reverse ? root.right : root.left, target, reverse, stack);
#   // early terminate, no need to traverse the whole tree
#   if ((reverse && root.val <= target) || (!reverse && root.val > target)) return;
#   // track the value of current node
#   stack.push(root.val);
#   inorder(reverse ? root.left : root.right, target, reverse, stack);
# } 
################
# Pseudocode:  Loop Detection HRabbit are and Turtule
# tortoise := firstNode
# hare := firstNode

# forever:

#   if hare == end 
#     return 'No Loop Found'

#   hare := hare.next

#   if hare == end
#     return 'No Loop Found'

#   hare = hare.next
#   tortoise = tortoise.next

#   if hare == tortoise
#     return 'Loop Found'
def num_of_paths_to_dest(n):
  #pass # your code goes here
  visit=[ [0] * n for _ in n]
  
  return path(n-1 , n-1)

  def path( x,y, path):
    
    if x< 0 or y < 0:
      return 0
    
    if y > x:
      return 0
   
    if x==y==0:
      visit[x][y]==1
  
    if visit[x][y] > 0:
      return visit[x][y]
    
    visit[x][y] = path( x-1 , y) + path (x , y-1)

    return  visit[x][y]

#########################################
def num_of_paths_to_dest(n):
  #pass # your code goes here
  visit=[ [0] * n for _ in range(n)]
  print(visit)
  
  def path( x,y):
    print(x,y)
    if x< 0 or y < 0:
      return 0
    
    elif y > x:
      visit[x][y] = 0
   
    elif x==y==0:
      visit[x][y]=1
  
    elif visit[x][y] > 0:
      return visit[x][y]
    
    else:
      visit[x][y] = path( x-1 , y) + path (x , y-1)
    
    return  visit[x][y]
    
  n = 2
  ans =  path( n-1 , n-1)
  
  return ans

def canwin(s ,memo):
    
    for i in range(len(s)-1):
        if s[i:i+2] == "++":
            #s[i:i+2] = '--'
            ss = s[:i] + "--" + s[i+2:]
            if not ss in memo :               
                 memo[ss] = canwin(ss,memo)

            win = not memo[ss] #canwin(ss,memo)
            #s[i:i+2] = '++'
            if win :return True
    
    return False

def minArea(image, x, y):
        """
        :type image: List[List[str]]
        :type x: int
        :type y: int
        :rtype: int
        """
        def getblack(r):
            left = len(image[r])
            right = -1
            for i in range(len(image[r])):
                if image[r][i] == '1':
                    # black
                    left = min(left , i)
                    right = max(right , i)
            
            return left , right
                
            
        minx, maxx= y,y
        h=0
        for row in range(len(image)):
            left , right = getblack(row)
            if right !=-1:
                h +=1 
                if left < minx :
                    minx= left
                if right > maxx:
                    maxx = right
            
                
                    
        return h * (maxx -minx +1)

# public class Solution {
#     private int top, bottom, left, right;
#     public int minArea(char[][] image, int x, int y) {
#         if(image.length == 0 || image[0].length == 0) return 0;
#         top = bottom = x;
#         left = right = y;
#         dfs(image, x, y);
#         return (right - left) * (bottom - top);
#     }
#     private void dfs(char[][] image, int x, int y){
#         if(x < 0 || y < 0 || x >= image.length || y >= image[0].length ||
#           image[x][y] == '0')
#             return;
#         image[x][y] = '0'; // mark visited black pixel as white
#         top = Math.min(top, x);
#         bottom = Math.max(bottom, x + 1);
#         left = Math.min(left, y);
#         right = Math.max(right, y + 1);
#         dfs(image, x + 1, y);
#         dfs(image, x - 1, y);
#         dfs(image, x, y - 1);
#         dfs(image, x, y + 1);
#     }
# }

def numIslands( grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        #grid= g
        height = len(grid)
        width = len( grid[0])
        islands=0
        
        def dfs( x , y):
            #termination cond
            if (x < 0) or  (y < 0) or (x >= height) or (y >= width) or (grid[x][y] == "0") :
                return
            
            #processing
            grid[x][y] = "0"
            
            # top = min(top , x)
            # bottom = max( bottom , x)
            # left = min(left , y)
            # right = max( right , y)
            
            # go deeper
            dfs( x+1 , y)
            dfs( x-1 , y)
            dfs ( x, y-1)
            dfs( x , y+1)
            
            return
            
            
       
               
        for x in range(height):
            for y in range( width):
                if grid[x][y] =='1':
                    dfs(x,y)
                    islands +=1                   
        
        # find one land point > find all connected land points and mask them > got find another land
        
        return islands
def allcomb(x) : # all out of 1,3 4
    
    memo={ c:[] for c in range(1,x+1)}
    memo[1], memo[2] , memo[3]  =['1'] ,['2', '1+1'], ['3', '2+1' , '1+2' , '1+1+1'] #, ['4'] #, '1+3', '1+1+1+1']

    def comb(x):
        for n in [1,2, 3]:
            if x > n:            
                if len(memo[x-n]) == 0 :            
                    memo[x-n]=comb(x-n)
                #if x in memo:
                memo[x].extend([str(n)+'+'+c for c in memo[x-n]])
      
        
        return memo[x]

    # memo[x] = comb(x-1)    
    # memo[x].extend( comb(x-3))
    # memo[x].extend(comb(x-4))
    return comb(x) #memo[x]
      

if __name__ == '__main__': 
    stack=[]
    stk2=[]
    test = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabcaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    #test ='abcabcbb'
    # factors=[[] for _ in xrange(10)]
    max=99
    pro=9779
    i=max
    while  i > pro/i :
        if(pro % i == 0 ): 
                   print(i)
               
        i-=1
        

    print(allcomb(6))
    hcounts={2:1, 3:2 ,1:3}
    print (sum(hcounts[x] for x in hcounts if x >= 4) )

    print( numIslands([["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]) )
    print( minArea ([["0","0","1","0"],["0","1","1","0"],["0","1","0","0"]] , 0 ,2))
    print(canwin( '++++' , {}))
    print(num_of_paths_to_dest(2))

    flowers=[1,2,3,6,4,5]
    n=5
    visit=[ [0] * n for _ in range(n)]
    prin(visit)
    is_cyclic(None)

    nums=[2,-1,3,0]
    nums= sorted(nums)
    print(nums)

    days = [0] * len(flowers)
    for day, position in enumerate(flowers, 1):
            days[position - 1] = day
    print(days)

    print(lengthOfLongestUniqSubstring("xabcddf"))
    print(lengthOfLongestUniqSubstring(""))
    print(lengthOfLongestUniqSubstring("x"))
    print(lengthOfLongestUniqSubstring("xabcdafg"))

    print(longestPalindrom( "x"))
    print(longestPalindrom( "abba"))
    print(longestPalindrom( "xabbcd"))
    print(longestPalindrom( "xabbcabbbbad"))

    print( getRoot( 0.1,3))

    print( primefactors(15))
    print(removeStr("Battle of the Vowels: Hawaii vs. Grozny", "aeiou"))
    print( firstNoneDuplicate("total"))

    print( stk2.append[3])
    print(convert("AB", 1))
    print(longestPalindrome(test))

    s2= ''.join([test[i] for i in range(len(test)-1 , -1, -1) ])
    print(s2)
    print(lengthOfLongestSubstring(test) )
    print( max_min_binary(6)) #
    print( max_min_binary(0))
    print(binary_represent(0.72))

    print(insert_m_n( 128 , 27 , 2,6))
    print(insert_m_n( 128 , 27 , 2,6))
############
    print(permut("abcd"))
########
    push_min(5)
    push_min(3)
    push_min(6)
    print( get_min()) #3
    print(pop_min()) # 6,3
    print(pop_min()) #3,3
    print(pop_min()) #5,5

    #print(keyboard("GEEK"))
    # print(is_balanced('{{()[]}}'))
    # print(is_balanced('{]}'))
    # print(is_balanced('{'))
    # print(is_balanced('}'))
    #print(is_rotation_eff("waterbottLe","erbottLewat"))
#     print(find_busiest_period( 
# [[1487799425,14,1],[1487799425,4,1],[1487799425,2,1],[1487800378,10,1],[1487801478,18,1],[1487901013,1,1],[1487901211,7,1],[1487901211,7,1]]))
    # print(equal_subsets([1,2,3,4,5,7]))
    print('')
    # print([1] +[0,0,0] )
    # print(addone([9,9,9], 3))
    #print(bag_of_words("a simple word of word"))

    #print( palindrom("ma,l,l;;a m]", '[];, '))
    # print(permutation_count('dog','D g  o  ','[], ')) 
    # print(permutation('dog','G, o[d]    ', '[], ')) 

    # print(is_uniqe("abc"))

    # find("ABAB", "ABABABCABABABCABABABC")
    # find("AABCAA", "AABCAABCAA")
    
    #print(compress_string("aabbbbcc"))

    # init(10)
    # print(camlcase_snakecase('CamelCamel'))
    # print(camlcase_snakecase('HTTPRequest'))
    # print(camlcase_snakecase('getResponseOFF'))

    # s= timeit.default_timer()
    # print(gcd(25*1000000,23*1000000))
    # print( timeit.default_timer() -s)

    # s= timeit.default_timer()
    # print(gcd_eff(25*1000000,23))
    # print( timeit.default_timer() -s)

    # print( 35 % 15)

    # print(gen_primes(200))


