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

# 98. Validate Binary Search Tree
class Solution:
    def traversal(self, root):
        ''' return is_bst, min, max of this whole tree starting at root
        '''
        # this method should never be called on a null root
        # recursive base case
        if root.left is None and root.right is None:
            return True, root.val, root.val
        
        # left_min should be min of this whole tree starting at root 
        # also need to check left_max <= root.val
        if root.left is not None:
            left_is_bst, left_min, left_max = traversal(root.left)
        else:
            left_is_bst, left_min, left_max = True, root.val, root.val            
        
        # right_max should be max of this whole tree starting at root
        # need to check if root.val <= right_min
        if root.right is not None:
            right_is_bst, right_min, right_max = traversal(root.right)
        else:
            right_is_bst, right_min, right_max = True, root.val, root.val

    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        # valid BST: everything on the left subtree is less than mid
        # this is a valid bst if root.left is a valid bst, root.right is a valid bst, and max of left subtree < this < min of right subtree
        # think of the relationship between this and in order tree traversal
        # how can you return the right most (should be the max) and left most (should be the min) of a subtree
