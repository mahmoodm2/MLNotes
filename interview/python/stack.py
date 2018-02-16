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