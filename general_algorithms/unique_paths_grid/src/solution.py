'''
# Unique paths in a grid

A small example of using dynamic programming techniques for finding out the total number of unique paths between two points in a 2D grid.

Author:
Goran Trlin

Find more tutorials and code samples on:
https://playandlearntocode.com
'''


class Solution:
    # camelCase is used at the LeetCode website
    # def uniquePaths(self, m: int, n: int) -> int:
    def unique_paths(self, m: int, n: int) -> int:
        '''
        Dynamic programming solution
        On paper, it's just about summation of left and above at each new element
        An alternative would be backtracking.
        '''
        if m == 0 or n == 0:
            return 0

        if m == 1 and n == 1:
            return 1
        if m == 2 and n == 1:
            return 1
        if m == 1 and n == 2:
            return 1
        if m == 2 and n == 2:
            return 2

        cur_m = 1
        cur_n = 1

        if cur_m > m - 1:
            cur_m = m - 1

        if cur_n > n - 1:
            cur_n = n - 1

        table = [[0 for x in range(n)] for y in range(m)]  # create table
        done_m = False
        done_n = False

        while cur_m < m and cur_n < n:
            temp_m = 0
            temp_n = 0

            # go right:
            if done_m == False:
                while temp_n <= cur_n:
                    if temp_n == 0:
                        table[cur_m][temp_n] = 1
                    else:
                        table[cur_m][temp_n] = table[cur_m][temp_n - 1] + table[cur_m - 1][temp_n]
                    temp_n += 1

            # go down:
            if done_n == False:
                while temp_m <= cur_m:
                    if temp_m == 0:
                        table[temp_m][cur_n] = 1
                    else:
                        table[temp_m][cur_n] = table[temp_m][cur_n - 1] + table[temp_m - 1][cur_n]
                    temp_m += 1

            if cur_m == m - 1 and cur_n == n - 1:
                return table[cur_m][cur_n]  # done

            cur_m += 1
            cur_n += 1

            if cur_m > m - 1:
                done_m = True
                cur_m = m - 1

            if cur_n > n - 1:
                done_n = True
                cur_n = n - 1

        return 0
