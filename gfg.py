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

# 1. Two Sum
def twoSum(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    # if you want to check if target can be written as a sum of e in nums
    # f(n,s) = f(n-1,s) or f(n-1,s-a[n])
    # n = len(nums)
    # mat = [[False for j in range(target+1)] for i in range(n+1)]
    # for i in range(n+1):
    #     mat[i][0] = True
    # for i in range(1,n+1):
    #     for j in range(1,target+1):
    #         if j - nums[i-1] >= 0:
    #             mat[i][j] = mat[i-1][j] or mat[i-1][j - nums[i-1]]
    # return mat[n][target]
    # now the two sum problem
    # map a key := element to its index in nums
    d = dict()
    for j in range(len(nums)):
        e = nums[j]
        if target-e in d:
            return [d[target-e], j]
        d[e] = j
    return [] # if nothing found

# 3. Longest Substring Without Repeating Characters
def lengthOfLongestSubstring(self, s):
    n = len(s)
    if n < 1:
        return 0
    # i,j := start and end point of substring under examination
    i, j = 0, 0    
    maxlen = 1
    d = {s[0]: 0}
    # only loop until before j hits the last index n-1, once j hits that and i is somewhere,
    # maxlen cannot increase further by moving i forward
    while (j+1 < n):
        # we already have tested that s(i,j) has no duplicate here 
        # now try increasing j, the end point
        # test if s[j+1] is in s(i,j)
        if s[j+1] in d:
            # k is the index of the duplicate 
            k = d[s[j+1]]
            if k >= i:
                # this means the duplicate is in s(i,j)
                # there is a duplicate, maxlen doesn't change
                i = k + 1                
            else:
                # here the dup is not in s(i,j)
                # eliminates the need to drop the elements from s[i:k] from d
                # s(i,j) can also be extended
                maxlen = max(maxlen, j+1-i+1)
        else:            
            # now your s(i,j) can be extended
            maxlen = max(maxlen, j+1-i+1)
        d[s[j+1]] = j+1
        j += 1
    return maxlen

# 4. Median of Two Sorted Arrays
import math
def findMedianSortedArrays(self, nums1, nums2):
    if len(nums1) <= len(nums2):
        A, B, m, n = nums1, nums2, len(nums1), len(nums2)
    else:
        A, B, m, n = nums2, nums1, len(nums2), len(nums1)    
    odd = ((m+n) % 2 != 0)
    begin, end = -1, m-1
    while begin <= end:
        i = math.ceil((begin+end)/2)
        if odd:
            j = (m+n+1) // 2 - 2 - i
        else:
            j = (m+n) // 2 - 2 - i

        # corner case
        if i == m-1: 
            # there is no A[i+1]
            min_right = B[j+1]
            if j == -1:
                max_left = A[i]
            else:
                max_left = max(A[i], B[j])            
            if odd: 
                return max_left
            else:
                return (max_left + min_right) / 2
        
        # corner case
        if i == -1:
            max_left = B[j]
            if j == n-1:
                min_right = A[i+1]
            else:
                min_right = min(A[i+1], B[j+1])
            if odd: 
                return min_right
            else:
                return (max_left + min_right) / 2

        # found the median
        if A[i] <= B[j+1] and B[j] <= A[i+1]:
            max_left = max(A[i], B[j])
            min_right = min(A[i+1], B[j+1])
            return (max_left + min_right) / 2
        
        if A[i] > B[j+1]:
            # i is too big
            end = i - 1
        else:
            # i is too small
            begin = i + 1

# 5. Longest Palindromic Substring
def longest_palindromic_subsequence(self, s):
    n = len(s)
    if n == 0:
        return
    f = [[None for j in range(n)] for i in range(n)]
    for k in range(n):
        f[k][k] = s[k]
        if k != n-1:
            f[k+1][k] = ''
    for length in range(1,n):
        for i in range(n-length):
            j = i + length
            if s[i] == s[j]:
                f[i][j] = s[i] + f[i+1][j-1] + s[j]
            else:
                if len(f[i][j-1]) >= len(f[i+1][j]):
                    f[i][j] = f[i][j-1]
                else:
                    f[i][j] = f[i+1][j]
    return f[0][n-1]

def longest_palindromic_substring(self, s):
    # DP method
    n = len(s)
    if n < 2:
        return s
    # f[i][j] is True if s[i:j+1] is Palindromic
    f = [[False for j in range(n)] for i in range(n)]
    for k in range(n):
        f[k][k] = True
        if k != n-1:
            f[k+1][k] = True
    maxlen, maxstr = 1, s[0]
    for length in range(1,n):
        for i in range(n-length):
            j = i + length
            if s[i] == s[j]:
                f[i][j] = f[i+1][j-1]
            else:
                f[i][j] = False
            # if you found a new Palindromic Substring
            if f[i][j]:
                if j-i+1 > maxlen:
                    maxlen = j-i+1
                    maxstr = s[i:j+1]
    return maxstr

def longest_palindromic_substring(self, s):
    # start by searching for the middle point
    n = len(s)
    if n < 2:
        return s
    maxlen, maxstr = 1, s[0]
    for i in range(n):
        # case when s[i] is the middle
        begin, end = i-1, i+1
        while begin >= 0 and end <= n-1:
            if s[begin] == s[end]:
                if end - begin + 1 > maxlen:
                    maxlen = end - begin + 1
                    maxstr = s[begin:end+1]
                begin -= 1
                end += 1
            else:
                break
        # case when s[i] == s[i+1] and they are the two middle
        if i != n-1 and s[i] == s[i+1]:
            # have got a length 2 Palindromic Substring here
            if maxlen < 2:
                maxlen == 2
                maxstr = s[i:i+2]
            begin, end = i-1, i+2
            while begin >= 0 and end <= n-1:
                if s[begin] == s[end]:
                    if end - begin + 1 > maxlen:
                        maxlen = end - begin + 1
                        maxstr = s[begin:end+1]
                    begin -= 1
                    end += 1
                else:
                    break
    return maxstr

# 11. Container With Most Water
def maxArea(self, height):
    n = len(height)
    begin, end = 0, n-1
    maxarea = 0
    while begin < end:
        if height[begin] < height[end]:
            current = height[begin] * (end - begin)
            begin += 1
        else:
            current = height[end] * (end - begin)
            end -= 1
        if current > maxarea:
            maxarea = current
    return maxarea

# 15. 3Sum
def two_sum(self, a, target):
    # there can be more than one solutions
    # return the two numbers not the two indexes
    ans = []
    n = len(a)
    if n < 2:
        return []
    d = {}
    for i in range(n):
        if target - a[i] in d:
            ans.append((target - a[i], a[i]))
        d[a[i]] = i
    return set(ans)

def threeSum(self, nums):
    nums.sort() # to avoid duplicate combos
    n = len(nums)
    ans = []
    for i in range(n-2):
        if i > 0 and nums[i] == nums[i-1]:
            # to avoid duplicate combos; there are duplicates in nums
            continue
        pairs = self.two_sum(nums[i+1:], 0 - nums[i])
        for left, right in pairs:
            ans.append([nums[i], left, right])
    return ans

# 42. Trapping Rain Water
def trap(self, height):
    n = len(height)
    if n < 3:
        return 0
    maxleft, maxright = 0, 0
    for i in range(1,n-1):
