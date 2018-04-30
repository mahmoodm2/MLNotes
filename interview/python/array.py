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
