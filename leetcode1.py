# 79. Word Search
# Given a 2D board and a word, find if the word exists in the grid.
# The word can be constructed from letters of sequentially adjacent cell, 
# where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.
class Solution:
    def dfs(self, board, word, i, j, visited):
        # check if word exists starting exactly from position [i,j]
        # DFS using recursive method
        # base case first
        m, n = len(board), len(board[0])
        if len(word) == 0:
            return True
        if i < 0 or i >= m or j < 0 or j >= n:
            return False
        # if [i,j] has been used before
        if visited[i][j]:
            return False
        # now prep for recursive case
        if word[0] != board[i][j]:
            return False
        # now that word[0] == board[i][j]:
        visited[i][j] = True
        word = word[1:]
        if self.dfs(board, word, i+1, j, visited):
            return True
        if self.dfs(board, word, i-1, j, visited):
            return True
        if self.dfs(board, word, i, j+1, visited):
            return True
        if self.dfs(board, word, i, j-1, visited):
            return True
        # this will be WRONG without this last line to adjust visited. 
        # you paint this [i,j] cell visited only when you are in the subtree of one of its adjacent cells
        # when you are no longer in that tree where [i,j] is a root, you need to return its status
        visited[i][j] = False
        return False
    
    def exist(self, board, word):
        # loop every i, j, check if word exists starting at i, j
        m, n = len(board), len(board[0])
        if m < 1 or n < 1:
            return False
        visited = [[False for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                if self.dfs(board, word, i, j, visited):
                    return True
        return False

# 78. Subsets
# Given a set of distinct integers, nums, return all possible subsets (the power set).
class Solution:
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        n = len(nums)
        # recursive base case        
        if n < 1:
            return [list()]
        not_included = self.subsets(nums[:-1])
        included = [s + [nums[-1]] for s in not_included]
        return not_included + included

# 94. Binary Tree Inorder Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # recursively
        # recursive base case
        if root is None:
            return []
        l = self.inorderTraversal(root.left) 
        m = [root.val] 
        r = self.inorderTraversal(root.right)
        return l + m + r

class Solution:
    def inorderTraversal(self, root):
        from queue import LifoQueue
        stack = LifoQueue()
        current = root
        while current is not None:
            stack.put(current)
            current = current.left
        # now that current is none
        mid = stack.pop()
        print(mid)
        current = mid.right
        # repeat the while loop

class Solution:
    def inorderTraversal(self, root):
        from queue import LifoQueue
        res = []
        stack = LifoQueue()
        current = root
        while current is not None or not stack.empty():
            while current is not None:
                stack.put(current)
                current = current.left
            # now that current is none
            mid = stack.get()
            res.append(mid.val)
            current = mid.right
        return res

# 111. Minimum Depth of Binary Tree
# The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
class Solution:
    def minDepth(self, root):
        # recursive method
        # recursive base case
        if root is None:
            # only for the weird case 
            # otherwise usually mindepth is not called on a null node
            return 0
        if root.left is None and root.right is None:
            return 1
        # now that root has at least 1 child
        if root.left is None:
            return 1 + self.minDepth(root.right)
        if root.right is None:
            return 1 + self.minDepth(root.left)
        # now that root has 2 children
        return 1 + min(self.minDepth(root.right), self.minDepth(root.left))

class Solution:
    def minDepth(self, root):
        # level order traversal, more efficient for mindepth
        # assume each node has an attribute node.level
        if root is None:
            return 0
        from queue import Queue
        root.level = 1
        reached_leaf = False
        q = Queue()
        q.enqueue(root)
        # never enqueue a None node
        while not reached_leaf:
            current = q.dequeue()
            # if current is a leaf
            if current.left is None and current.right is None:
                reached_leaf = True
                return current.level
            # now that current has at least 1 child
            if current.left is not None:
                current.left.level = current.level + 1
                q.enqueue(current.left)
            if current.right is not None:
                current.right.level = current.level + 1
                q.enqueue(current.level)

# 94. Binary Tree Inorder Traversal
class Solution:
    def inorderTraversal(self, root):
        res = []
        from queue import LifoQueue
        stack = LifoQueue()
        current = root
        while current is not None:
            stack.put(current)
            current = current.left
        # now that current is none
        x = stack.get()
        res.append(x)
        current = x.right
        # now go back to the above while loop, to do this see below
class Solution:
    def inorderTraversal(self, root):
        res = []
        from queue import LifoQueue
        stack = LifoQueue()
        current = root
        while current is not None or not stack.empty():
            while current is not None:
                stack.put(current)
                current = current.left
            # now that current has become none
            x = stack.get()
            res.append(x.val)
            current = x.right
        return res

# 62. Unique Paths
class Solution:
    def uniquePaths(self, m, n):
        f = [[1 for j in range(n)] for i in range(m)]
        for i in range(m-2,-1,-1):
            for j in range(n-2,-1,-1):
                f[i][j] = f[i+1][j] + f[i][j+1]
        return f[0][0]

# 55. Jump Game
# Given an array of non-negative integers, you are initially positioned at the first index of the array.
# Each element in the array represents your maximum jump length at that position.
# Determine if you are able to reach the last index.
# A = [2,3,1,1,4], return true.
# A = [3,2,1,0,4], return false.
class Solution:
    def canJump(self, nums):
        n = len(nums)
        if n < 1:
            return True
        f = [False for _ in range(n)]
        f[-1] = True
        for i in range(n-2,-1,-1):
            # try to determine f[i] here
            # the max number of steps you can take from here is either a[i] or n-1-i
            max_steps = min(nums[i], n-1-i)
            for step_size in range(1,max_steps+1):
                if f[i + step_size] is True:
                    f[i] = True
        # this is not O(N)
        return f[0]
# the above is O(N**2); the key to O(N) is if you can find an index j where reachable is true, 
# then reaching n is equivalent to reaching j as reaching j takes fewer steps so if smt can reach n, definitely can reach j
class Solution:
    def canJump(self, nums):
        n = len(nums)
        if n < 1:
            return True
        f = [False for _ in range(n)]
        f[-1] = True
        smallest_starting_position = n-1
        for i in range(n-2,-1,-1):
            if smallest_starting_position - i <= nums[i]:
                f[i] = True
                smallest_starting_position = i
        return f[0]
class Solution:
    def canJump(self, nums):
        n = len(nums)
        if n < 1:
            return True
        smallest_starting_position = n-1
        for i in range(n-2,-1,-1):
            if smallest_starting_position - i <= nums[i]:                
                smallest_starting_position = i
        return smallest_starting_position == 0

# 98. Validate Binary Search Tree
class Solution:
    def traversal(self, root):
        ''' return is_bst, min, max of this whole tree starting at root
        '''
        # this method should never be called on a null root
        # recursive base case
        if root.left is None and root.right is None:
            return True, root.val, root.val
        
        # now that root has at least 1 child
        # left_min should be min of this whole tree starting at root 
        # also need to check left_max < root.val
        if root.left is None:
            left_is_bst = True
            left_min = root.val
            left_is_smaller = True
            # in this case root.right is not none
            right_is_bst, right_min, right_max = self.traversal(root.right)
            right_is_bigger = root.val < right_min
        elif root.right is None:
            right_is_bst = True
            right_max = root.val
            right_is_bigger = True
            # here root.left is not none            
            left_is_bst, left_min, left_max = self.traversal(root.left)
            left_is_smaller = left_max < root.val
        else:
            right_is_bst, right_min, right_max = self.traversal(root.right)
            left_is_bst, left_min, left_max = self.traversal(root.left)
            left_is_smaller = left_max < root.val
            right_is_bigger = root.val < right_min

        is_bst = left_is_bst and right_is_bst and left_is_smaller and right_is_bigger
        return is_bst, left_min, right_max

    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        # valid BST: everything on the left subtree is less than mid
        # this is a valid bst if root.left is a valid bst, root.right is a valid bst, and max of left subtree < this < min of right subtree
        # think of the relationship between this and in order tree traversal
        # how can you return the right most (should be the max) and left most (should be the min) of a subtree
        if root is None:
            return True
        return self.traversal(root)[0]

# 98. Validate Binary Search Tree
class Solution:
    def isValidBST(self, root):
        # use in order traversal iterative
        # the key is, when you print the list from in order traversal, you get a sorted ascending list
        if root is None:
            return True
        from queue import LifoQueue
        stack = LifoQueue()
        current = root
        pre = None
        while current is not None or not stack.empty():
            while current is not None:
                stack.put(current)
                current = current.left
            # now that current has become none
            x = stack.get()
            if pre is not None:
                if pre.val >= x.val:
                    return False
            pre = x
            current = x.right
        return True

# 230. Kth Smallest Element in a BST
class Solution:
    def kthSmallest(self, root, k):
        count = 0
        from queue import LifoQueue
        stack = LifoQueue()
        current = root
        while current is not None or not stack.empty():
            while current is not None:
                stack.put(current)
                current = current.left
            # now that current has become none
            x = stack.get()
            count += 1
            if count == k:
                return x.val
            current = x.right

# 198. House Robber
class Solution:
    def rob(self, nums):
        # f(n-1) = max gain until house at position n-1, inclusive
        # f(n-1) = a[n-1] + f(n-3) if robbing house at index n-1, or f(n-2) if not robbing at index n-1
        # f(0) = a[0]
        n = len(nums)
        if n < 1:
            return 0
        f = [None for _ in range(n)]
        f[0] = nums[0]
        for i in range(1,n):
            if i-2 >= 0:
                f[i] = max(f[i-1], nums[i] + f[i-2])
            else:
                f[i] = max(f[i-1], nums[i])
        return f[-1]

# 121. Best Time to Buy and Sell Stock
class Solution:
    def maxProfit(self, prices):
        n = len(prices)
        if n < 2:
            return 0
        profit_if_buy_here = [None for _ in range(n)]
        profit_if_buy_here[-1] = 0
        local_max_price = prices[-1]
        for i in range(n-1,0,-1):
            if prices[i] > local_max_price:
                local_max_price = prices[i]
            profit_if_buy_here[i-1] = local_max_price - prices[i-1]
        return max(profit_if_buy_here)

# further optimize
class Solution:
    def maxProfit(self, prices):
        n = len(prices)
        if n < 2:
            return 0
        global_max_profit = 0
        # to hold max of prices[i:]
        local_max_price = prices[-1]
        for i in range(n-1,0,-1):
            if prices[i] > local_max_price:
                local_max_price = prices[i]
            local_max_profit = local_max_price - prices[i-1]
            if local_max_profit > global_max_profit:
                global_max_profit = local_max_profit
        return global_max_profit

# 494. Target Sum
class Solution:
    def findTargetSumWays(self, nums, S):
        if (S + sum(nums)) % 2 != 0:
            return 0
        n = len(nums)
        if n < 1:
            return 0
        target = (S + sum(nums)) // 2
        f = [[None for j in range(target+1)] for i in range(n+1)]
        for i in range(n+1):
            # if target is j=0, there is one way
            f[i][0] = 1 
        for j in range(1,target+1):
            # if no num is used, there is zero way
            f[0][j] = 0
        # i means using up to position i in nums, j is the target sum
        for i in range(1,n+1):
            for j in range(1,target+1):
                f[i][j] = f[i-1][j]
                if j - nums[i-1] >= 0:
                    f[i][j] += f[i-1][j - nums[i-1]]
        return f[n][target]

# further optimize
class Solution:
    def findTargetSumWays(self, nums, S):
        if (S + sum(nums)) % 2 != 0:
            return 0
        n = len(nums)
        if n < 1:
            return 0
        target = (S + sum(nums)) // 2
        f = [[None for j in range(target+1)] for i in range(n+1)]
        for i in range(n+1):
            # if target is j=0, there is one way
            f[i][0] = 1 
        for j in range(1,target+1):
            # if no num is used, there is zero way
            f[0][j] = 0
        # i means using up to position i in nums, j is the target sum
        for i in range(1,n+1):
            for j in range(target, nums[i-1]-1, -1):
                f[i][j] = f[i-1][j] + f[i-1][j - nums[i-1]]
        return f[n][target]

# further optimize
class Solution:
    def count_subsets_sum(self, nums, target):
        n = len(nums)
        if n < 1:
            return 0
        # this initialization is equivalent to when i=0, using 0 num
        f = [0 for j in range(target+1)]
        f[0] = 1 # when the target is 0, there is 1 way
        for i in range(1,n+1):
            for j in range(target, nums[i-1]-1, -1):
                f[j] += f[j - nums[i-1]]
        return f[target]

    def findTargetSumWays(self, nums, S):
        if (S + sum(nums)) % 2 != 0:
            return 0        
        target = (S + sum(nums)) // 2
        return self.count_subsets_sum(nums, target)
    
# 102. Binary Tree Level Order Traversal
class Solution:
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        q = Queue()
        res = []
        q.enqueue(root)
        while not q.empty():
            node = q.dequeue()
            res.append(node.val)
            if node.left is not None:
                q.enqueue(node.left)
            if node.right is not None:
                q.enqueue(node.right)
        # this cannot return a list of list

class Solution:
    def levelOrder(self, root):
        from queue import Queue
        q1, q2 = Queue(), Queue()
        res = []
        if root is None:
            return []
        q1.put(root)
        while not q1.empty() or not q2.empty():
            # at start of each loop, either q1 must be empty or q2 must be empty
            # alternating between 2 queues
            level_list = []
            if not q1.empty():
                while not q1.empty():
                    node = q1.get()
                    level_list.append(node.val)
                    if node.left is not None:
                        q2.put(node.left)
                    if node.right is not None:
                        q2.put(node.right)
            else:
                while not q2.empty():
                    node = q2.get()
                    level_list.append(node.val)
                    if node.left is not None:
                        q1.put(node.left)
                    if node.right is not None:
                        q1.put(node.right)
            if len(level_list) > 0:
                res.append(level_list)
        return res
class Solution:
    def levelOrder(self, root):
        # how can you combine q1 and q2 into 1 queue
        # do not use an inner while loop; instead check length of level and use for loop
        from queue import Queue
        q = Queue()
        res = []
        if root is None:
            return []
        q.put(root)
        while not q.empty():
            # len of the level you are about to process
            level_length = q.qsize()
            level_list = []
            for _ in range(level_length):
                node = q.get()
                level_list.append(node.val)
                if node.left is not None:
                    q.put(node.left)
                if node.right is not None:
                    q.put(node.right)
            if len(level_list) > 0:
                res.append(level_list)
        return res

# 91. Decode Ways
class Solution:
    def numDecodings(self, s):
        n = len(s)
        if n < 1:
            return 0
        valid = list(map(str, range(1,27)))
        g = [None for _ in range(n+1)]
        g[0] = 1
        for j in range(1,n+1):
            if int(s[j-1]) > 0:
                g[j] = g[j-1]                
            else:
                g[j] = 0
            if j-2 >= 0:
                if s[j-2:j] in valid:
                    g[j] += g[j-2]
        return g[-1]
