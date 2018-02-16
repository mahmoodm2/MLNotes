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