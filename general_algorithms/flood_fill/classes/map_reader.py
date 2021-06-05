class MapReader:
    '''
    Reads the reduced map (1s and 0s), where 1 - island and 0 - sea
    '''

    def recursive_flood_fill(self, grid, row: int, col: int):
        '''
        Recursive flood fill can get stopped by stack limits (maximum recursion depth error). Iterative version doesn't have this problem.
        '''
        if (row < 0 or row > len(grid) - 1):
            return

        if (col < 0 or col > len(grid[0]) - 1):
            return

        if (grid[row][col] != 1):
            return

        if (grid[row][col] == 1):
            grid[row][col] = -1

        self.flood_fill(grid, row - 1, col)
        self.flood_fill(grid, row + 1, col)
        self.flood_fill(grid, row, col - 1)
        self.flood_fill(grid, row, col + 1)

    def is_one(self, grid, row, col):
        '''
        Helper method for checking whether the pixel belongs to an island or not
        '''
        if (row < 0 or row > len(grid) - 1):
            return False

        if (col < 0 or col > len(grid[0]) - 1):
            return False

        if grid[row][col] == 1:
            return True
        else:
            return False

    def iterative_flood_fill(self, grid, row, col):
        '''
        Iterative version of flood fill algorithm. Works better for larger maps.
        '''
        if (row < 0 or row > len(grid) - 1):
            return

        if (col < 0 or col > len(grid[0]) - 1):
            return

        if (grid[row][col] != 1):
            return

        q = []  # init empty queue (FIFO)
        grid[row][col] = -1  # mark as visited
        q.append([row, col])  # add to queue

        while len(q) > 0:
            [cur_row, cur_col] = q[0]
            del q[0]

            if (self.is_one(grid, cur_row - 1, cur_col) == True):
                grid[cur_row - 1][cur_col] = -1
                q.append([cur_row - 1, cur_col])

            if (self.is_one(grid, cur_row + 1, cur_col) == True):
                grid[cur_row + 1][cur_col] = -1
                q.append([cur_row + 1, cur_col])

            if (self.is_one(grid, cur_row, cur_col - 1) == True):
                grid[cur_row][cur_col - 1] = -1
                q.append([cur_row, cur_col - 1])

            if (self.is_one(grid, cur_row, cur_col + 1) == True):
                grid[cur_row][cur_col + 1] = -1
                q.append([cur_row, cur_col + 1])

    def count_islands(self, reduced_map):
        '''
        Main method for counting islands on the map
        '''
        islands = 0

        for row in range(len(reduced_map)):
            for col in range(len(reduced_map[0])):
                if (reduced_map[row][col] == 1):
                    islands += 1
                    self.iterative_flood_fill(reduced_map, row, col)
        return islands
