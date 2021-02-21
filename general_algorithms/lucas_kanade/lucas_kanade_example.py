'''
Lucas-Kanade Tracker example in Python

Author: Goran Trlin
Find more tutorials and code samples on:
https://playandlearntocode.com
'''

import numpy as np
from classes.image_helper import ImageHelper
from classes.lucas_kanade_tracker import LucasKanadeTracker


def main():
    image_width, image_height = 400, 400

    # tracked regions (features):
    # starting point (top left coordinate)
    x0_arr = [178, 295, 250]
    y0_arr = [96, 175, 310]

    original_x0_arr, original_y0_arr = x0_arr[:], y0_arr[:]
    # tracked box width / height in pixels:
    box_size = 30

    # image dataset size:
    start_image_index = 17
    end_image_index = 38

    image_name_base = "sample1-"
    velocity_vector_arr = [None] * len(x0_arr)
    resulting_images = []

    # for any image:
    for img_index in range(start_image_index, end_image_index + 1):

        img_helper = ImageHelper()
        # full color images:
        (loaded_image_1, pixels_1) = img_helper.load_image("images/sample1-" + str(img_index) + ".png")
        (loaded_image_2, pixels_2) = img_helper.load_image("images/sample1-" + str(img_index + 1) + ".png")

        reduced_map_t1 = []  # map of grey pixels for image t1
        for i in range(image_height):
            reduced_map_t1.append([])
            for j in range(image_width):
                reduced_map_t1[i].append(None)

        for x in range(image_width):
            for y in range(image_height):
                avg = (pixels_1[x, y][0] + pixels_1[x, y][1] + pixels_1[x, y][2]) / 3.0
                reduced_map_t1[y][x] = avg

        reduced_map_t2 = []  # map of grey pixels for image t2
        for i in range(image_height):
            reduced_map_t2.append([])
            for j in range(image_width):
                reduced_map_t2[i].append(None)

        for x in range(image_width):
            for y in range(image_height):
                avg = (pixels_2[x, y][0] + pixels_2[x, y][1] + pixels_2[x, y][2]) / 3.0
                reduced_map_t2[y][x] = avg

        # for any tracked rectange:
        for point_index in range(len(x0_arr)):
            if velocity_vector_arr[point_index] is not None:
                x0_arr[point_index] += velocity_vector_arr[point_index][0][0]
                y0_arr[point_index] += velocity_vector_arr[point_index][1][0]

                x0_arr[point_index] = round(x0_arr[point_index])
                y0_arr[point_index] = round(y0_arr[point_index])
                x0_arr[point_index] = int(x0_arr[point_index])
                y0_arr[point_index] = int(y0_arr[point_index])

            lucas_kanade = LucasKanadeTracker()
            changes_x = lucas_kanade.get_intensity_changes_for_box_x(reduced_map_t1,
                                                                     y0_arr[point_index], x0_arr[point_index], box_size)
            changes_y = lucas_kanade.get_intensity_changes_for_box_y(reduced_map_t1, y0_arr[point_index],
                                                                     x0_arr[point_index],
                                                                     box_size)
            changes_t = lucas_kanade.get_intensity_changes_for_box_time(reduced_map_t1, reduced_map_t2,
                                                                        y0_arr[point_index], x0_arr[point_index],
                                                                        box_size)

            cropped_t1 = loaded_image_1.crop((x0_arr[point_index], y0_arr[point_index], x0_arr[point_index] + box_size,
                                              y0_arr[point_index] + box_size))
            cropped_t2 = loaded_image_2.crop((x0_arr[point_index], y0_arr[point_index], x0_arr[point_index] + box_size,
                                              y0_arr[point_index] + box_size))

            # if we want to display the tracked region only:
            # cropped_t1.show()
            # cropped_t2.show()

            # compute velocity vector:

            # form S matrix now:
            flatten_x = changes_x.flatten()
            transpose_changes_x = np.array([flatten_x]).transpose()

            flatten_y = changes_y.flatten()
            transpose_changes_y = np.array([flatten_y]).transpose()

            flatten_t = changes_t.flatten()
            transpose_changes_t = np.array([flatten_t]).transpose()

            s_matrix = np.concatenate(np.array([transpose_changes_x, transpose_changes_y]), axis=1)
            s_matrix_transpose = s_matrix.transpose()
            t_matrix = transpose_changes_t
            st_s = np.matmul(s_matrix_transpose, s_matrix)

            w, v = np.linalg.eig(st_s)
            cond = abs(w[0]) / abs(w[1])
            print("Condition number:")
            print(cond)

            st_s_inv = np.linalg.inv(st_s)
            temp_matrix = np.matmul(st_s_inv, s_matrix_transpose)
            velocity_vector_arr[point_index] = np.matmul(temp_matrix, t_matrix)

            print('velocity_vector:')
            print(velocity_vector_arr)

            # draw  scaled velocity vector:
            line_scale_factor = 30

            line_x0 = x0_arr[point_index] + box_size / 2
            line_y0 = y0_arr[point_index] + box_size / 2

            img_helper.draw_line(loaded_image_1,
                                 (
                                     line_x0, line_y0,
                                     line_x0 + velocity_vector_arr[point_index][0][0] * line_scale_factor,
                                     line_y0 + velocity_vector_arr[point_index][1][0] * line_scale_factor),
                                 (255, 0, 0))

            # draw the original position:
            img_helper.draw_rectangle(loaded_image_1,
                                      [original_x0_arr[point_index], original_y0_arr[point_index],
                                       original_x0_arr[point_index] + box_size,
                                       original_y0_arr[point_index] + box_size], fill=None,
                                      outline=(100, 100, 100))

            # draw the tracked box:
            img_helper.draw_rectangle(loaded_image_1,
                                      [x0_arr[point_index], y0_arr[point_index], x0_arr[point_index] + box_size,
                                       y0_arr[point_index] + box_size], fill=None,
                                      outline=(255, 255, 255))

        loaded_image_1.show()
        resulting_images.append(loaded_image_1)


# RUN:
main()
