import numpy as np


class LucasKanadeTracker:
    def get_intensity_changes_for_box_x(self, grid, row, col, box_size):
        output = np.zeros((box_size, box_size))

        for r in range(box_size):
            for c in range(box_size):
                next_value = 0
                prev_value = 0

                if (c + 1 > box_size - 1):
                    next_value = grid[row + r][col + c]
                else:
                    next_value = grid[row + r][col + c + 1]

                if (c - 1 < 0):
                    prev_value = grid[row + r][col + c]
                else:
                    prev_value = grid[row + r][col + c - 1]

                # output[r][c] = next_value - grid[row + r][col + c]
                output[r][c] = (next_value * 1.00 - prev_value) / 2.0

        return output

    def get_intensity_changes_for_box_y(self, grid, row, col, box_size):
        output = np.zeros((box_size, box_size))
        for r in range(box_size):
            for c in range(box_size):
                next_value = 0
                prev_value = 0
                cur_value = grid[row + r][col + c]

                if (r + 1 > box_size - 1):
                    next_value = cur_value
                else:
                    next_value = grid[row + r + 1][col + c]

                if (r - 1 < 0):
                    prev_value = cur_value
                else:
                    prev_value = grid[row + r - 1][col + c]

                # output[r][c] = next_value - grid[row + r][col + c]
                output[r][c] = (next_value - prev_value) / 2.0
        return output

    def get_intensity_changes_for_box_time(self, grid_t1, grid_t2, row, col, box_size):
        output = np.zeros((box_size, box_size))
        for r in range(box_size):
            for c in range(box_size):
                output[r][c] = -(grid_t2[row + r][col + c] - grid_t1[row + r][col + c])
        return output
