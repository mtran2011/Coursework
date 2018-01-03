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