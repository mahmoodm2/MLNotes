# Graph : dict{ node : adjacents list}, visited= set(nodes)

def dfs( g ,node, visited):

    print(node) # visit node
    visited.add(node)
    for n in g[node]: # adjacents
        if n not in visited:
            dfs( g, n , visited)
    return visited


def dfs_using_stack(g, start):
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