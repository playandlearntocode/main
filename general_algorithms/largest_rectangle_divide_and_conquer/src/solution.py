class Solution:
    '''
    Divide and Conquer solution using a segment tree (O(N logN))
    Build a segment tree (O(N logN)), and use it to find minimums of bar height.
    Then, choose the max out of the tuple (central area, left to min bar area, right to min bar area)

    Note the use of two arrays to build segment trees (one for values, one for indexes).
    '''
    def build_segment_tree(self, node_index, heights, tree_arr, tree_arr_indexed, start, end):
        '''
        Build segment tree so we can quickly (logN) find the min.value in any given range
        '''
        if (start == end):
            tree_arr[node_index] = heights[start]
            tree_arr_indexed[node_index] = start
            return start
        else:
            mid = (end + start) // 2

            left_index = self.build_segment_tree(2 * node_index, heights, tree_arr, tree_arr_indexed, start, mid)
            right_index = self.build_segment_tree(2 * node_index + 1, heights, tree_arr, tree_arr_indexed, mid + 1, end)

            if (heights[left_index] <= heights[right_index]):
                tree_arr[node_index] = heights[left_index]
                tree_arr_indexed[node_index] = left_index
                return left_index
            else:
                tree_arr[node_index] = heights[right_index]
                tree_arr_indexed[node_index] = right_index
                return right_index

    def query_segment_tree(self, node_index, tree_arr, tree_arr_indexed, start, end, r1, r2):
        # if cur. space contains no extra numbers, return the min (full overlap)
        # if no overlap , return MAX_INT
        # if partial overlap , focus on the overlapping section ?
        if (r1 < start and r2 < start) or (r1 > end and r2 > end):
            # no overlap
            return {'value': 99999, 'index': -1}
        elif (start >= r1 and end <= r2):
            # full overlap:
            return {'value': tree_arr[node_index], 'index': tree_arr_indexed[node_index]}
        else:
            # partial, just go further down:
            mid = (end + start) // 2

            # index += 1
            m_left = self.query_segment_tree(2 * node_index, tree_arr, tree_arr_indexed, start, mid, r1, r2)
            m_right = self.query_segment_tree(2 * node_index + 1, tree_arr, tree_arr_indexed, mid + 1, end, r1, r2)

            if m_left['value'] <= m_right['value']:
                return m_left
            else:
                return m_right

    def recursive_largest_rect(self, heights, tree_arr, tree_arr_indexed, start, end):
        # main Divide and Conquer algorithm function
        # divide by min height bar and then select max out of 1) whole section , 2) left from the smallest one, 3) right from the smallest one
        if start > end:
            return 0

        if end > len(heights) - 1:
            return 0

        if start < 0:
            return 0

        if start == end:
            return heights[start]

        min_height = self.query_segment_tree(1, tree_arr, tree_arr_indexed, 0, len(heights) - 1, start, end)

        central_area = (end - start + 1) * min_height['value']
        left_area = self.recursive_largest_rect(heights, tree_arr, tree_arr_indexed, start, min_height['index'] - 1)
        right_area = self.recursive_largest_rect(heights, tree_arr, tree_arr_indexed, min_height['index'] + 1, end)

        return max(central_area, left_area, right_area)

    # LeetCode camelCase declaration:
    # def largestRectangleArea(self, heights: List[int]) -> int:
    def largest_rectangle_area(self, heights):
        '''
        Lets now build a Divide and Conquer O(nlogn) solution

        We need to. be able to find min. height for any range quickly. In order to do that,
        we will build a segment tree. Building that tree is O(nlogn) but querying is only
        O(logN)

        DnC algorithms are usually recursive. We will split the whole heights array into
        ranges between smallest elements. Then, each on these sections will be divided, and so on,
        recursively, until the base level is reached.
        '''
        tree_arr = [0] * (len(heights) * 4)
        tree_arr_indexed = [-1] * (len(heights) * 4)

        self.build_segment_tree(1, heights, tree_arr, tree_arr_indexed, 0, len(heights) - 1)

        max_area = self.recursive_largest_rect(heights, tree_arr, tree_arr_indexed, 0, len(heights) - 1)
        return max_area
