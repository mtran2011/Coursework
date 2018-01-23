# 647. Palindromic Substrings
class Solution:
    # dp method, O(n^2)
    def countSubstrings(self, s):
        n = len(s)
        if n < 1:
            return 0
        # f[start][end] is True if s[start:end+1] is palindrome
        f = [[False for _ in range(n)] for _ in range(n)]
        for k in range(n):
            f[k][k] = True
            # for convenience later
            if k+1 < n:
                f[k+1][k] = True
        count = n
        for length in range(1,n):
            for start in range(n-length):
                end = start + length
                if s[start] == s[end]:
                    f[start][end] = f[start+1][end-1]
                else:
                    f[start][end] = False
                if f[start][end] is True:
                    count += 1
        return count

class Solution:
    # look for the mid point of a palindrome and expand
    def countSubstrings(self, s):
        n = len(s)
        if n < 1:
            return 0
        count = n
        for mid in range(n-1):
            # case when s[mid] is the only mid point 
            start, end = mid-1, mid+1
            while start >= 0 and end < n:
                if s[start] == s[end]:
                    count += 1
                    start -= 1
                    end += 1
                else:
                    break
            # case when s[mid] and s[mid+1] are two mid points
            if s[mid] == s[mid+1]:
                count += 1
                start, end = mid-1, mid+2
                while start >= 0 and end < n:
                    if s[start] == s[end]:
                        count += 1
                        start -= 1
                        end += 1
                    else:
                        break
        return count

# 516. Longest Palindromic Subsequence
class Solution:
    def longestPalindromeSubseq(self, s):
        n = len(s)
        if n < 1:
            return 0
        # define f[i,j] = length of longest palindromic subseq between i, j
        f = [[0 for _ in range(n)] for _ in range(n)]
        for k in range(n):
            f[k][k] = 1
        for length in range(1,n):
            for begin in range(n-length):
                end = begin + length
                if s[begin] == s[end]:
                    f[begin][end] = 2 + f[begin+1][end-1]
                else:
                    f[begin][end] = max(f[begin+1][end], f[begin][end-1])
                
        return f[0][n-1]

# 120. Triangle
class Solution:
    def minimumTotal(self, triangle):
        n = len(triangle)
        if n < 1:
            return 0
        f = [[None for _ in range(len(triangle[i]))] for i in range(n)]
        # f[x] = min sum of path starting at position x
        # the bottom of f is the same as bottom of triangle
        for j in range(len(triangle[-1])):
            f[-1][j] = triangle[-1][j]
        # build f from the bottom up
        for row in range(n-2,-1,-1):
            for col in range(len(triangle[row])):
                f[row][col] = triangle[row][col] + min(f[row+1][col], f[row+1][col+1])
        return f[0][0]

# 139. Word Break
class Solution:
    # DP method
    def wordBreak(self, s, wordDict):
        n = len(s)
        wordDict = set(wordDict)
        # f[i] is True if s[:i] is breakable
        f = [False for _ in range(n+1)]
        f[0] = True
        for i in range(1,n+1):
            # determine f[i] and already know f[0] to f[i-1]
            # f[i] is when you add the new element s[i-1]
            for length in range(1,i+1):
                start = i-length
                if s[start:i] in wordDict and f[start] is True:
                    f[i] = True
                    break
        return f[-1]

class Solution:
    # DFS method
    def dfs(self, s, wordDict):
        # recursive base case
        if s in wordDict:
            return True        
        for word in wordDict:
            k = len(word)
            if k <= len(s):
                if s[:k] == word:
                    if self.dfs(s[k:], wordDict):
                        # this step is wasteful, for example if all word in dict has k=3
                        # then repeated check for nothing
                        return True
        return False

    def wordBreak(self, s, wordDict):
        # need to see that if a string u is breakable
        # then it's equivalent to u being in wordDict
        wordDict = set(wordDict)
        return self.dfs(s, wordDict)

# 152. Maximum Product Subarray
class Solution:
    def maxProduct(self, nums):
        # max(k) and min(k) = max and min product of subarray ending at exactly k
        # as you move forward, to get max(k+1) and min(k+1) you only need the previous max min
        # so you don't need to store the whole historical max, min
        n = len(nums)
        if n < 1:
            return 0
        localmax, localmin, globalmax = nums[0], nums[0], nums[0]
        for i in range(1,n):
            a, b = localmax * nums[i], localmin * nums[i]
            localmax = max(a, b, nums[i])
            localmin = min(a, b, nums[i])
            globalmax = max(globalmax, localmax)
        return globalmax

# 213. House Robber II
class Solution:
    def rob_straight_row(self, nums, begin, end):
        f = [0 for _ in range(len(nums))]
        f[begin] = nums[begin]
        for k in range(begin+1, end+1):
            if k-2 >= 0:
                f[k] = max(f[k-1], f[k-2] + nums[k])
            else:
                f[k] = max(f[k-1], nums[k])
        return f[end]

    def rob(self, nums):
        n = len(nums)
        if n < 1:
            return 0
        if n < 2:
            return nums[0]
        # using two passes
        x = self.rob_straight_row(nums, 1, n-1)
        y = self.rob_straight_row(nums, 0, n-2)
        print('x={}, y={}'.format(x,y))
        return max(x,y)

# 123. Best Time to Buy and Sell Stock III
class Solution:
    def profit_ktrades(self, prices, ntrades):
        # f[k][i] = max profit from up to k trades on the subarray prices from 0 to i inclusive
        # if not using prices[i] then f[k,i] = f[k, i-1]
        # if using prices[i] as the sale date of the last trade, then you make k-1 trades on prices 0 to j, and buy on date j
        # then f[k,i] = f[k-1,j] + a[i] - a[j] for some j between 0,i
        # f[0,i] = 0 for all i and f[k,0] = 0 for all k
        n = len(prices)
        if n < 2:
            return 0
        f = [[0 for _ in range(n)] for _ in range(ntrades+1)]
        for k in range(1,ntrades+1):
            # start out with j=0 and f[k-1][0] = 0 always
            # the trailingmax of f[k-1][j] - prices[j]
            trailingmax = 0 - prices[0]
            for i in range(1,n):
                trailingmax = max(trailingmax, f[k-1][i] - prices[i])
                f[k][i] = max(f[k][i-1], prices[i] + trailingmax)
        return f[ntrades][-1]

    def maxProfit(self, prices):
        return self.profit_ktrades(prices, 2)
# another method
class Solution:
    def maxProfit(self, prices):
        # if you make 2 trades, there must be 1 or more separate points i
        # you make 1 trade on 0 to i, inclusive, then make 1 trade from i to n-1
        # so you need maxleft of i and maxright of i
        # maxright[i] = max profit to gain on subarray i to n-1
        # construct maxright first, then as you move to get maxleft, keep track of global max
        n = len(prices)
        if n < 2:
            return 0
        maxright = [0 for _ in range(n)]
        maxleft = [0 for _ in range(n)]

        max_price_right = prices[-1]
        max_profit_right = 0
        for i in range(n-2,-1,-1):            
            max_price_right = max(prices[i], max_price_right)
            max_profit_right = max(max_profit_right, max_price_right - prices[i])
            maxright[i] = max_profit_right
        
        min_price_left = prices[0]
        max_profit_left = 0
        global_max = 0
        for i in range(1,n):
            min_price_left = min(min_price_left, prices[i])
            max_profit_left = max(max_profit_left, prices[i] - min_price_left)
            maxleft[i] = max_profit_left
            global_max = max(global_max, maxleft[i] + maxright[i])
        return global_max

# 20. Valid Parentheses
class Solution:
    def isValid(self, s):
        opening = {'(','{','['}
        closing = {')','}',']'}
        matched = {'(':')','{':'}','[':']'}
        if len(s) < 1:
            return True
        if s[0] not in opening or len(s) % 2 != 0:
            return False
        from queue import LifoQueue
        stack = LifoQueue()
        stack.put(s[0])
        for i in range(1,len(s)):
            if s[i] in opening:
                stack.put(s[i])
            else:
                open_sign = stack.get()
                if s[i] != matched[open_sign]:
                    return False
        # now you have processed all closing signs, there should be no opening signs left
        return stack.empty()

# 34. Search for a Range
class Solution:
    def binary(self, nums, target):
        n = len(nums)
        if n < 1:
            return [-1,-1]
        from math import floor
        begin, end = 0, n-1
        while begin <= end:
            mid = floor((begin + end) / 2)
            if target == nums[mid]:
                return mid
            if target < nums[mid]:
                end = mid - 1
            else:
                begin = mid + 1
        return -1

    def searchRange(self, nums, target):
