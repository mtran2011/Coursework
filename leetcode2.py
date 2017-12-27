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