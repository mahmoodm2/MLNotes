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

