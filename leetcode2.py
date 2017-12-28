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
