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
        
        if node.left!= None : 
            self.inorder(node.left) 

        #print(node.v)

        if node.right!= None : 
            self.inorder(node.right) 

    def print_tree(self,node=None):
        if node== None: return
        
        if node.left!= None :
            self.print_tree(node.left) 
        
        #print(node.v)
        
        if node.right!= None : 
            self.print_tree(node.right) 

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
############################################
def match_tree(t1,t2):
    if t1==None and t2==None : return True
    if t1==None or t2==None: return False

    if t1.v != t2.v : return False
    else:
        return match_tree(t1.left , t2.left) or match_tree(t1.right, t2.right)
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
