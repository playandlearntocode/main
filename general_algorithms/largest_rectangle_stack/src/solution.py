'''
Stack based O(N) Python solution for the algorithmic problem of finding the largest rectangle in a histogram

Author:
Goran Trlin

Find more tutorials and code samples on:
https://playandlearntocode.com

'''


class Solution:
    # LeetCode camelCase declaration format:
    # def largestRectangleArea(self, heights: List[int]) -> int:
    def largest_rectangle_area(self, heights):
        '''
        Stack based O(n) solution

        1st phase
        Compressing with stack. If the element gets positioned as the last remaining in the stack,
        it means that all heights left from its position are larger than it

        2nd phase
        Remaining elements after phase 1 can be only in rising slope
        '''
        stack = []
        max_area = 0
        i = 0

        while i < len(heights):
            if len(stack) == 0 or heights[i] >= heights[stack[-1]]:
                # if its larger or equal than the current stack pointer, it gets added to the stack
                stack.append(i)
                i += 1

            else:
                # smaller bar found, start the unwinding
                while len(stack) > 0 and heights[stack[-1]] > heights[i]:
                    # areaWithBar = (i + 1 - stack[-1]) * heights[i]
                    area = max_area
                    if len(stack) == 1:
                        area = max(area, heights[stack[-1]] * i)
                    else:
                        area = (i - (stack[-2] + 1)) * heights[stack[-1]]

                    if area > max_area:
                        max_area = area
                    stack.pop()

                stack.append(i)
                i += 1

        # smaller bar, start the unwinding
        if len(stack) > 0:
            while len(stack) > 0:
                if len(stack) == 1:
                    # smallest element in stack:
                    area = len(heights) * heights[stack[-1]]
                else:
                    # not the smallest one:
                    # cover the area from the end till +1 from the left el. in stack
                    area = max(heights[stack[-1]], heights[stack[-1]] * (len(heights) - stack[-2] - 1))
                if area > max_area:
                    max_area = area
                stack.pop()

        return max_area
