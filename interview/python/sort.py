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
