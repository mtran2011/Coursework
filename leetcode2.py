# 690. Employee Importance
class Solution:
    def getImportance(self, employees, id):
        d = {e.id: e for e in employees} 
        from queue import LifoQueue
        stack = LifoQueue()
        root = d[id]
        total = 0
        stack.put(root)
        while not stack.empty():
            current = stack.get()
            total += current.importance
            for sub in current.subordinates:
                stack.put(d[sub])
        return total

# 104. Maximum Depth of Binary Tree
class Solution:
    def maxDepth(self, root):
        if root is None:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return 1 + max(left, right)

# 100. Same Tree
class Solution:
    def isSameTree(self, p, q):
        if p is q:
            return True
        if p is not None and q is not None:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False

# 101. Symmetric Tree
class Solution:
    # does not work using in order traversal
    def inorder(self, root, lst):
        if root is None:
            lst.append('None')
            return
        self.inorder(root.left,lst)
        lst.append(root.val)
        self.inorder(root.right,lst)

    def isSymmetric(self, root):
        if root is None:
            return True
        left, right = [], []
        self.inorder(root.left, left)
        self.inorder(root.right, right)
        return left == right[::-1]

class Solution:
    def helper(self, p, q):
        # check if two subtrees rooted at p, q are mirrors of each other
        if p is None or q is None:
            return p is q
        if p.val != q.val:
            return False
        return self.helper(p.right, q.left) and self.helper(p.left, q.right)

    def isSymmetric(self, root):
        if root is None:
            return True
        return self.helper(root.left, root.right)

# 110. Balanced Binary Tree
class Solution:
    def helper(self, root):
        # modified from maxdepth
        # return maxdepth and true if depth of left and right not differ by more than 1
        if root is None:
            return 0, True
        left_depth, left_is_balanced = self.helper(root.left)
        right_depth, right_is_balanced = self.helper(root.right)
        depth = 1 + max(left_depth, right_depth)
        is_balanced = abs(left_depth-right_depth) <= 1 and left_is_balanced and right_is_balanced
        return depth, is_balanced

    def isBalanced(self, root):
        return self.helper(root)[1]

# 257. Binary Tree Paths
class Solution:
    def getpaths(self, root):
        # should never be called on null root
        # if you do allow calling on null and return [], imagine a leaf node
        # from one leaf, there are 2 paths, both are [leaf]
        # so cannot call this on null node
        if root.left is None and root.right is None:
            # recursive base case
            return [[root.val]]
        if root.left is None:
            return [([root.val] + path) for path in self.getpaths(root.right)]
        elif root.right is None:
            return [([root.val] + path) for path in self.getpaths(root.left)]
        else:
            right = [([root.val] + path) for path in self.getpaths(root.right)]
            left = [([root.val] + path) for path in self.getpaths(root.left)]
            return left + right

    def binaryTreePaths(self, root):
        # paths from root = each path from root.left + each path from root.right, append root in front
        if root is None:
            return []
        paths = ['->'.join(map(str,path)) for path in self.getpaths(root)]
        return paths

# 144. Binary Tree Preorder Traversal
class Solution:
    def preorderTraversal(self, root):
        # print(root.val)
        # self.preorderTraversal(root.left)
        # self.preorderTraversal(root.right)
        res = []
        if root is None:
            return res
        from queue import LifoQueue
        stack = LifoQueue()
        stack.put(root)
        while not stack.empty():
            root = stack.get()            
            res.append(root.val)
            if root.right is not None:
                stack.put(root.right)
            if root.left is not None:
                stack.put(root.left)            
        return res

# 122. Best Time to Buy and Sell Stock II
class Solution:
    def maxProfit(self, prices):
        # if buy on day i, wanna sell on max(i+1,...,n-1)
        # say buy on day i, sell on day j is the best if you can transact only once
        # not sure if this can partition arr into 3 parts
        # if price tomorrow is higher than today, buy, else sell to zero today
        pass

# 189. Rotate Array
class Solution:
    def reverse(self, nums, begin, end):
        # reverse in place
        while begin < end:
            nums[begin], nums[end] = nums[end], nums[begin]
            begin += 1
            end -= 1
            
    def rotate(self, nums, k):
        # if you want to do it in place
        # first reverse the whole list
        # then reverse back the first k nums
        # then reverse back from (k+1,n)
        n = len(nums)
        k = k % n
        if k == 0:
            return
        self.reverse(nums, 0, n-1)
        self.reverse(nums, 0, k-1)
        self.reverse(nums, k, n-1)

# 142. Linked List Cycle II
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None:
            return None
        slow, fast = head, head
        met = False
        # part 1 find the meeting point
        while not met:
            if fast.next is None:
                return None
            if fast.next.next is None:
                return None
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                met = True
        # part 2 find entry point of cycle
        newslow = head
        while newslow is not slow:
            newslow = newslow.next
            slow = slow.next
        return slow

# 287. Find the Duplicate Number
class Solution(object):
    def findDuplicate(self, nums):
        # think of each cell in nums as a node 
        # the value of the cell is a pointer to the next node
        if len(nums) < 2:
            return -1
        slow, fast = 0, 0
        met = False
        while not met:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                met = True
        newslow = 0
        while newslow != slow:
            newslow = nums[newslow]
            slow = nums[slow]
        return slow

# 454. 4Sum II
class Solution(object):
    def fourSumCount(self, A, B, C, D):
        # get all possible sum from A and B and 
        # map sum to count how many times it appears
        mydict = {}
        for a in A:
            for b in B:
                s = a + b
                mydict[s] = mydict.get(s,0) + 1
        
        res = 0
        for c in C:
            for d in D:
                s = c + d
                if -s in mydict:
                    res += mydict[-s]
        return res

# 384. Shuffle an Array
import random
class Solution:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """        
        self.orig = nums
        self.nums = [x for x in nums]        

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        return self.orig        

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """        
        n = len(self.nums)
        for i in range(n):
            u = random.randint(i,n-1)
            self.nums[i], self.nums[u] = self.nums[u], self.nums[i]
        return self.nums

# 46. Permutations
class Solution:
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # recursive base case
        n = len(nums)
        if n < 2:
            return [nums]
        res = []
        for i in range(n):
            for lst in self.permute(nums[:i] + nums[i+1:]):
                res.append([nums[i]] + lst)
        return res

# 515. Find Largest Value in Each Tree Row
class Solution:
    def largestValues(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root is None:
            return []
        from queue import Queue
        q = Queue()
        # never put a null node into this q
        q.put(root)
        res = []
        while not q.empty():
            # this is before exploring a new level
            maxlevel = None
            for _ in range(q.qsize()):
                # use q.size to look at nodes only on that same level
                node = q.get()

                if node.left is not None:
                    q.put(node.left)
                if node.right is not None:
                    q.put(node.right)

                if maxlevel is None:
                    maxlevel = node.val
                else:
                    maxlevel = max(maxlevel, node.val)
            # now you have finished exploring this level
            res.append(maxlevel)
        return res

# 513. Find Bottom Left Tree Value
class Solution:
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        from queue import Queue
        q = Queue()
        # never put a null node into this q
        q.put(root)
        leftmost = None
        while not q.empty():
            for i in range(q.qsize()):
                # use q.size to look at nodes only on that same level
                node = q.get()
                if node.left is not None:
                    q.put(node.left)
                if node.right is not None:
                    q.put(node.right)
                
                if i == 0:
                    leftmost = node.val
        return leftmost

# 103. Binary Tree Zigzag Level Order Traversal
class Solution:
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root is None:
            return []
        from queue import Queue
        q = Queue()
        # never put a null node into this q
        q.put(root)
        res = []
        level = 0
        while not q.empty():
            thislevel = []
            for _ in range(q.qsize()):
                node = q.get()
                if node.left is not None:
                    q.put(node.left)
                if node.right is not None:
                    q.put(node.right)
                thislevel.append(node.val)
            # if level is even, go left to right
            # if level is odd, go right to left
            if level % 2 != 0:
                thislevel = thislevel[::-1]
            res.append(thislevel)
            level += 1 # prepare for next level
        return res

# 417. Pacific Atlantic Water Flow
class Solution:
    def pacificAtlantic(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        if len(matrix) < 1:
            return []
        m, n = len(matrix), len(matrix[0])
        if m < 1 or n < 1:
            return []
        atlantic = [[False for _ in range(n)] for _ in range(m)]
        pacific = [[False for _ in range(n)] for _ in range(m)]        
        for j in range(n):
            atlantic[-1][j] = True
            pacific[0][j] = True
        for i in range(m):
            atlantic[i][-1] = True
            pacific[i][0] = True
        # fill in pacific
        # this is wrong because water can also flow [i,j] to [i,j+1] to pacific
        # how about do the reverse? instead of asking if [i,j] can flow to some True cell
        # from each True cell, look for a higher cell that can flow down to it
        for i in range(1,m):
            for j in range(1,n):
                if pacific[i-1][j] is True and matrix[i][j] >= matrix[i-1][j]:
                    pacific[i][j] = True
                if pacific[i][j-1] is True and matrix[i][j] >= matrix[i][j-1]:
                    pacific[i][j] = True
        # fill in atlantic
        for i in range(m-2,-1,-1):
            for j in range(n-2,-1,-1):
                if atlantic[i+1][j] is True and matrix[i][j] >= matrix[i+1][j]:
                    atlantic[i][j] = True
                if atlantic[i][j+1] is True and matrix[i][j] >= matrix[i][j+1]:
                    atlantic[i][j] = True
        # fill in combined
        res = []
        for i in range(m):
            for j in range(n):
                if pacific[i][j] and atlantic[i][j]:
                    res.append([i,j])
        return res

class Solution:
    def pacificAtlantic(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        if len(matrix) < 1:
            return []
        m, n = len(matrix), len(matrix[0])
        if m < 1 or n < 1:
            return []
        from queue import Queue
        atlantic = [[False for _ in range(n)] for _ in range(m)]
        pacific = [[False for _ in range(n)] for _ in range(m)]
        atlantic_q, pacific_q = Queue(), Queue()
        for j in range(n):
            atlantic[m-1][j] = True
            atlantic_q.put((m-1,j))
            pacific[0][j] = True
            pacific_q.put((0,j))
        for i in range(m):
            atlantic[i][n-1] = True
            atlantic_q.put((i,n-1))
            pacific[i][0] = True
            pacific_q.put((i,0))
        
        # fill in connection to atlantic
        while not atlantic_q.empty():
            row, col = atlantic_q.get()
            for i, j in [(row,col+1), (row,col-1), (row+1,col), (row-1,col)]:
                if -1 < i < m and -1 < j < n:
                    if matrix[i][j] >= matrix[row][col]:
                        if atlantic[i][j] is False:
                            atlantic[i][j] = True
                            atlantic_q.put((i,j))
        
        while not pacific_q.empty():
            row, col = pacific_q.get()
            for i, j in [(row,col+1), (row,col-1), (row+1,col), (row-1,col)]:
                if -1 < i < m and -1 < j < n:
                    if matrix[i][j] >= matrix[row][col]:
                        if pacific[i][j] is False:
                            pacific[i][j] = True
                            pacific_q.put((i,j))
        
        # fill in combined
        res = []
        for i in range(m):
            for j in range(n):
                if pacific[i][j] and atlantic[i][j]:
                    res.append([i,j])
        return res

# 329. Longest Increasing Path in a Matrix
# def dfs(G):
#     for v in G.vertices:
#         v.color = white
#         v.parent = null
#     for v in G.vertices:
#         if v.color == white:
#             visit(G,v)

# def visit(G,v):
#     # this method should be called only on white vertex
#     v.color = gray
#     v.start_time = time
#     time += 1
#     for u in v.connections:
#         if u.color = white:
#             u.parent = v
#             visit(G,u)
#     v.color = black
#     v.end_time = time
#     time += 1

class Solution:
    def visit(self, row, col, matrix, memo):
        m, n = len(matrix), len(matrix[0])
        # the smallest len of an increasing path is 1, aka the cell itself        
        memo[row][col] = 1
        # for each cell in the 4 direction
        for i, j in [(row,col+1), (row,col-1), (row+1,col), (row-1,col)]:
            # if this is a cell inside the matrix boundary
            if -1 < i < m and -1 < j < n:                
                if matrix[row][col] < matrix[i][j]:
                    # visit [i,j] only if [i,j] is still a white vertex
                    if memo[i][j] == 0:
                        self.visit(i, j, matrix, memo)
                    # if [i,j] was visited before via another dfs, then its memo value is already correct
                    # if [i,j] was white, then you visit it above
                    # so now memo[i,j] should be correct
                    # now that you are done processing [i,j]
                    if memo[i][j] + 1 > memo[row][col]:
                        memo[row][col] = memo[i][j] + 1        
    
    def longestIncreasingPath(self, matrix):
        if len(matrix) < 1:
            return 0
        m, n = len(matrix), len(matrix[0])
        if n < 1:
            return 0
        # the memo matrix memoizes the len of longest path starting from each position
        # it also marks visited or not, memo = 0 means not visited
        memo = [[0 for _ in range(n)] for _ in range(m)]
        maxpath = 1
        for row in range(m):
            for col in range(n):
                if memo[row][col] == 0:
                    # if not visited yet
                    self.visit(row, col, matrix, memo)
                # if [row,col] was visited before via another dfs, then its memo value is already correct
                # if [row,col] was white, then you visit it above
                # so now memo[row,col] should be correct
                if memo[row][col] > maxpath:
                    maxpath = memo[row][col]
        return maxpath

# 542. 01 Matrix
class Solution:
    # dfs with memoization
    # def dfs(G):
    #     for v in G.V:
    #         v.parent = None
    #         v.color = white
    #     global time = 0
    #     for v in G.V:
    #         if v.color == white:
    #             visit(G,v)
    # def visit(G,u):
        # this visit method is only called on white vertex
        # time += 1
        # u.start_time = time
        # u.color = gray
        # for v in u.connected:
        #     if v.color == white:
        #         v.parent = u
        #         visit(G,v)
        # time += 1
        # u.end_time = time
    def visit(self, i, j, matrix, memo):
        m, n = len(matrix), len(matrix[0])        
        # recursive base case
        if matrix[i][j] == 0:
            memo[i][j] = 0
        else:
            memo[i][j] = 1e6 # mark it as gray
            for row, col in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
                if -1 < row < m and -1 < col < n:
                    if memo[row][col] is None:
                        self.visit(row, col, matrix, memo)
                    # if [row,col] has never been visited, then you visit it above
                    # if [row,col] was visited by another dfs before, then memo[row,col] is already correct
                    # but hold on this is not really true
                    # if you are exploring [i,j] by first moving 1 up, then now moving 1 to the right, you explore [i,j+1]
                    # but when exploring [i,j+1], you also check out [i,j] again which now has a temp value from the previous up path
                    # so the 2 lines below being here does not work
                    # if memo[i][j] > 1 + memo[row][col]:
                    #     memo[i][j] = 1 + memo[row][col]
            for row, col in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
                if -1 < row < m and -1 < col < n:
                    if memo[i][j] > 1 + memo[row][col]:
                        memo[i][j] = 1 + memo[row][col]
            # why this method doesn't work
            # if you move right (i,j) to (i,j+1), at (i,j+1) you only explore 3 paths 
            # so excluding the path from (i,j+1) back to (i,j)
            # the key for dfs to work easily is that if you go parent to child, there's no need to go child back to parent
            # but here you need (i,j+1) back to (i,j)
    
    def updateMatrix(self, matrix):
        # dfs with memoization
        m, n = len(matrix), len(matrix[0])
        memo = [[None for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if memo[i][j] is None:
                    self.visit(i, j, matrix, memo)
                # now memo[i][j] holds correct value either by the visit above
                # or via another dfs starting at a previous cell
        return memo
# in this problem you cannot do bfs starting from a non-zero cell
# instead from each zero-cell, use it as root and start a bfs tree
# recall that bfs gives the shortest path from this 0-cell to the 1-cell 
# when you encounter a 1-cell, if the distance found by this bfs < current dist, then set current dist
class Solution:
    def explore(self, row, col, matrix, dist):
        m, n = len(matrix), len(matrix[0])
        from queue import Queue
        q = Queue()
        q.put((row,col))
        while not q.empty():
            i, j = q.get()
            for r, c in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
                if -1 < r < m and -1 < c < n:
                    # (i,j) is like a parent of (r,c)
                    if dist[i][j] + 1 < dist[r][c]:
                        dist[r][c] = dist[i][j] + 1
                        q.put((r,c))

    def updateMatrix(self, matrix):
        import sys
        m, n = len(matrix), len(matrix[0])
        dist = [[None for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    dist[i][j] = 0
                else:
                    dist[i][j] = sys.maxsize
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    # explore a bfs tree rooted at (i,j)
                    self.explore(i, j, matrix, dist)
        return dist
# condense the above to 1 queue
class Solution:
    def updateMatrix(self, matrix):
        import sys
        from queue import Queue
        m, n = len(matrix), len(matrix[0])
        q = Queue()
        dist = [[None for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    dist[i][j] = 0
                    # explore this 0-node later
                    q.put((i,j))
                else:
                    dist[i][j] = sys.maxsize
        while not q.empty():
            i, j = q.get()
            for r, c in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
                if -1 < r < m and -1 < c < n:
                    # (i,j) is like a parent of (r,c) if the dist works out
                    if dist[i][j] + 1 < dist[r][c]:
                        dist[r][c] = dist[i][j] + 1
                        q.put((r,c))
        return dist

# 310. Minimum Height Trees
# there are N nodes, you can check each node as the source, do a bfs from this source
# bfs is O(V+E) and for a tree, V = N and E = N -1 so O(N) and repeat N times so O(N^2) in total
class Solution:
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        # remember, this is a tree graph so it has to be all connected
        # any connected graph without simple cycles is a tree
        # if you trim down to 2 leafs connected to each other, 
        # then either of them could serve as root of minimum height tree
        # build a graph adjacency list
        if len(edges) < 1:
            return [0]
        adj = {vertex : [] for vertex in range(n)}
        for u, v in edges:
            # the edge is (u,v) and (v,u)
            adj[u].append(v)
            adj[v].append(u)
        leafs = []
        for vertex in range(n):
            if len(adj[vertex]) == 1:
                leafs.append(vertex)
        while len(adj) > 2:
            nextlevel_leafs = []
            for leaf in leafs:
                potential_next_leaf = adj[leaf][0]
                # severe the link between leaf and potential_next_leaf
                adj[potential_next_leaf].remove(leaf)
                adj.pop(leaf)
                # check if this potential can be a leaf in next trimming round
                if len(adj[potential_next_leaf]) == 1:
                    nextlevel_leafs.append(potential_next_leaf)
            leafs = nextlevel_leafs
        return leafs

# 127. Word Ladder
# imagine the beginword is the source in a bfs
# the endword is the node you want to reach from the source
# bfs gives shortest path from source 
# the word u connects to v if v has 1 letter changed from u
class Solution:
    def word_to_dict(self, word):
        mydict = {}
        for s in word:
            mydict[s] = mydict.get(s,0) + 1
        return mydict

    def wrong_connected(self, word1, word2):
        # this method is wrong in checking if word1 connects to word2
        if len(word1) != len(word2):
            return False
        d1 = self.word_to_dict(word1)
        d2 = self.word_to_dict(word2)
        diff = 0
        for s in set(list(d1.keys()) + list(d2.keys())):
            diff += abs(d1.get(s,0) - d2.get(s,0))
        return diff < 3

    def connected(self, word1, word2):
        # word2 is 1 letter changed from word1
        if len(word1) != len(word2):
            return False
        diff = 0
        for i in range(len(word1)):
            if word1[i] != word2[i]:
                diff += 1
                if diff >= 2:
                    return False
        return diff == 1

    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        # build the adj list for the graph
        if endWord not in wordList:
            return 0
        vertexes = list(set([beginWord] + wordList))
        # each vertex is mapped to its adj list and color = 0 and distance = None
        adj = {word: [[], 0, None] for word in vertexes}
        # check if each word connects to the other word 
        # if i connects to i+k, when you are at i+k, no need to check for (i+k,i) again
        for i in range(len(vertexes)-1):
            for j in range(i+1, len(vertexes)):
                # check if (i,j) connects
                if self.connected(vertexes[i], vertexes[j]):
                    adj[vertexes[i]][0].append(vertexes[j])
                    adj[vertexes[j]][0].append(vertexes[i])
        # now that you have the adj list, use bfs where beginWord is the source
        from queue import Queue 
        q = Queue()
        q.put(beginWord)
        # source.color = gray, source.distance = 0
        adj[beginWord][1] = 1
        adj[beginWord][2] = 0
        reached_end_word = False
        while not reached_end_word and not q.empty():
            u = q.get()
            # for each v that u connects to
            # for v in adj[u][0]:
            #     if v.color == white:
            #         v.color = gray
            #         v.parent = u
            #         v.distance = u.distance + 1
            #         q.enqueue(v)
            # u.color = black
            for v in adj[u][0]:
                if adj[v][1] == 0:
                    adj[v][1] = 1
                    adj[v][2] = adj[u][2] + 1
                    q.put(v)
                if v == endWord:
                    reached_end_word = True
            adj[u][1] = 2
        if not reached_end_word:
            return 0
        return adj[endWord][2] + 1
