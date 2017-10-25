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
