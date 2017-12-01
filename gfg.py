# Dynamic Programming | Set 16 (Floyd Warshall Algorithm)
def line_to_matrix(N,s):
    '''
    Args:
        s(list): 1D list of len N*N
        N(int): one dimension of the matrix
    Returns:
        list: 2D list of N*N dimension
    '''
    a = [[s[N*(i-1)+j-1] for j in range(1,N+1)] for i in range(1,N+1)]
    return a

t = int(input())
for _ in range(t):
    N = int(input())
    s = list(map(int, input().split()))
    dist = line_to_matrix(N,s)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    result = [dist[i][j] for i in range(N) for j in range(N)]
    print(' '.join(map(str, result)))

# Dynamic Programming | Set 17 Palindromic patitioning
t = int(input())
for _ in range(t):    
    a = input()
    N = len(a)
    f = [[0 for _ in range(N)] for _ in range(N)]
    is_palindrome = [[True for _ in range(N)] for _ in range(N)]    
    for length in range(1,N):
        for m in range(N-length):
            if a[m] == a[m+length] and is_palindrome[m+1][m+length-1]:
                is_palindrome[m][m+length] = True
            else:
                is_palindrome[m][m+length] = False
            
            if is_palindrome[m][m+length]:
                f[m][m+length] = 0
            else:
                f[m][m+length] = min(f[m][k] + f[k+1][m+length] + 1 for k in range(m,m+length))
    print(f[0][N-1])

# Breadth First Search: Shortest Reach
import sys
import queue
def bfs(adj, s):
    '''
    initialize all vertex v.color = white, v.d = infinite, v.parent = null
    s.color = gray, s.d = 0
    initialize an empty FIFO queue Q
    add s to Q
    while Q not empty:
        u = Q.dequeue() to examine u
        for each vertex v that u connects to:
            if v.color == white: meaning that v has never been found before
                v.color = gray
                v.d = u.d + 1
                v.parent = u
                Q.enqueue(v) to explore v later
        u.color = black because finishing exploring u
    '''
    n = len(adj)
    colors = [0 for _ in range(n)]
    dist = [-1 for _ in range(n)]
    colors[s] = 1
    dist[s] = 0
    q = queue.Queue(n)
    q.put(s)
    while not q.empty():
        u = q.get()
        for v in adj[u]:
            if colors[v] == 0:
                colors[v] = 1
                dist[v] = dist[u] + 6
                q.put(v)
        colors[u] = 2
    return dist

t = int(input())
for _ in range(t):
    V, E = map(int, input().split())
    adj = [list() for _ in range(V+1)]
    for _ in range(E):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    s = int(input())
    dist = bfs(adj, s)
    print(' '.join(str(dist[j]) for j in range(1,V+1) if j != s))
    pass

# Dynamic Programming | Set 20 (Maximum Length Chain of Pairs)
def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0  
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K
def compare(x,y):
    if x[1] <= y[0]:
        return -1
    if x[0] >= y[1]:
        return 1
def maxChainLen(lis, n):
    if n < 2:
        return n
    lis = sorted(lis, key=cmp_to_key(compare))
    f = [1 for _ in range(n)]    
    for k in range(1,n):
        for i in range(k):
            if lis[i][1] < lis[k][0]:
                if f[i] + 1 > f[k]:
                    f[k] = f[i] + 1
    return max(f)

# Dijkstra: Shortest Reach 2
import sys
import queue

class Node(object):
    def __init__(self, name):
        self.name = name
        self.d = sys.maxsize
        self.parent = None

def dij(adj, s):
    '''
    initialize for all v in V: v.d = inf, v.parent = null
    set s.d = 0
    initialize an empty set or priority minQ of unprocessed vertexes, add all v to Q
    while Q not empty and Q.delmin has d != inf:
        u = Q.delmin
        for each v that u connects to:
            if u.d + weight(u,v) < v.d:
                v.d = u.d + weight(u,v)
                v.parent = u
        mark u as processed
    '''
    n = len(adj) # n = V + 1
    Q = queue.PriorityQueue(n)
    
    for name in range(1,n):
        x = Node(name)
        if name == s:
            x.d = 0
        Q.put((x.d, x))
    while not Q.empty() and Q.get()[0] != sys.maxsize:
        _, u = Q.get()
        adj[u.name]
    pass

t = int(input())
for _ in range(t):
    V, E = map(int, input().split())
    adj = [list() for _ in range(V+1)]
    for _ in range(E):
        u, v, r = map(int, input().split())
        adj[u].append((v,r))
        adj[v].append((u,r))
        s = int(input())
        pass
    pass

# Dynamic Programming | Set 30 (Dice Throw)
# HR - Bricks Game 
def bricks(a):
    N = len(a)
    if N <= 3:
        return sum(a)
    f = [0 for _ in range(N+1)]
    a.reverse()
    for i in range(1,4):
        f[i] = sum(a[:i])
    for j in range(4,N+1):
        x1 = a[j-1] + min(f[j-2], f[j-3], f[j-4])
        if j >= 6:
            x2 = a[j-1] + a[j-2] + min(f[j-3], f[j-4], f[j-5])
            x3 = a[j-1] + a[j-2] + a[j-3] + min(f[j-4], f[j-5], f[j-6])
        elif j >= 5:
            x2 = a[j-1] + a[j-2] + min(f[j-3], f[j-4], f[j-5])
            x3 = a[j-1] + a[j-2] + a[j-3] + min(f[j-4], f[j-5])
        else:
            x2 = a[j-1] + a[j-2] 
            x3 = a[j-1] + a[j-2] + a[j-3] 
        
        f[j] = max(x1,x2,x3)    
    return f[N]

t = int(input())
for _ in range(t):
    N = int(input())
    a = list(map(int, input().split()))
    print(bricks(a))        

# KnightL on a Chessboard

# def bfs(s, G):
#     for vertex v in G:
#         v.color = white
#         v.dist = inf
#         v.parent = null 
#     s.dist = 0 
#     s.color = gray
#     q = queue.Queue()
#     q.enqueue(s)
#     while q.not_empty():
#         u = q.dequeue()
#         for each v that u connects to:
#             if v.color == white:
#                 q.enqueue(v)
#                 v.color = gray
#                 v.dist = u.dist + 1
#                 v.parent = u
#         u.color = black
def check_cell_on_board(n,i,j):
    # check if cell (i,j) is on board n x n
    if i < 1 or i > n:
        return False
    if j < 1 or j > n:
        return False
    return True

def knight(n,a,b):
    # go from (1,1) to (n,n) using knight(a,b) moves
    colors = [[0 for j in range(n+1)] for i in range(n+1)]
    dist = [[sys.maxsize for j in range(n+1)] for i in range(n+1)]
    dist[1][1] = 0
    colors[1][1] = 1
    q = queue.Queue()
    q.put((1,1))
    while not q.empty():
        u = q.get()
        # cell (i,j) is connected to (i+-a,j+-b) and (i+-b,j+-a)
        i, j = u
        connected = []
        for x in [i+a, i-a]:
            for y in [j+b, j-b]:
                if check_cell_on_board(n,x,y):
                    connected.append((x,y))
        for x in [i+b,i-b]:
            for y in [j+a,j-a]:
                if check_cell_on_board(n,x,y):
                    connected.append((x,y))
        for v in connected:
            x,y = v
            if colors[x][y] == 0:
                q.put(v)
                colors[x][y] = 1 
                dist[x][y] = dist[i][j] + 1
                # if destination found
                if x ==n and y == n:
                    return dist[x][y]
        colors[i][j] = 2
    # if cell (n,n) is never discovered then it is not reachable from (1,1)
    return -1

n = int(input())
res = [[None for i in range(n)] for j in range(n)]
for a in range(1,n):
    for b in range(1,n):
        if b < a:
            res[a][b] = res[b][a]
        else:
            res[a][b] = knight(n,a,b)
        print(res[a][b],end=" ")
        if b == n-1:
            print()

# 64. Minimum Path Sum
def minPathSum(self, grid):
    # minpath(i,j) = grid[i][j] + smaller of minpath(i+1,j) and minpath(i,j+1)
    m, n = len(grid), len(grid[0])
    cost = [[None for j in range(n)] for i in range(m)]
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            cost[i][j] = grid[i][j]
            if i+1 < m and j+1 < n:
                cost[i][j] += min(cost[i+1][j], cost[i][j+1])
            elif i+1 < m:
                cost[i][j] += cost[i+1][j]
            elif j+1 < n:
                cost[i][j] += cost[i][j+1]
    return cost[0][0]