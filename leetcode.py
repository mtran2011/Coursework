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
