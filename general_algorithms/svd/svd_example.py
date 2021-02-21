'''
Objective:
SVD for image compression example
Author:
Goran Trlin

Prerequisites:
pip install numpy
pip install pillow

Video tutorial URL:
https://www.youtube.com/watch?v=SU851ljMIZ8

Find more tutorials and code samples on:
https://playandlearntocode.com
'''

import numpy
from PIL import Image


# FUNCTION DEFINTIONS:

# open the image and return 3 matrices, each corresponding to one channel (R, G and B channels)
def open_image(image_path):
    im_orig = Image.open(image_path)
    im = numpy.array(im_orig)

    a_red = im[:, :, 0]
    a_green = im[:, :, 1]
    a_blue_ = im[:, :, 2]

    return [a_red, a_green, a_blue_, im_orig]


# compress the matrix of a single channel
def compress_single_channel(channel_data_matrix, singular_values_limit):
    u_channel, s_channel, vh_channel = numpy.linalg.svd(channel_data_matrix)
    a_channel_compressed = numpy.zeros((channel_data_matrix.shape[0], channel_data_matrix.shape[1]))
    k = singular_values_limit

    left_side = numpy.matmul(u_channel[:, 0:k], numpy.diag(s_channel)[0:k, 0:k])
    a_channel_compressed_inner = numpy.matmul(left_side, vh_channel[0:k, :])
    a_channel_compressed = a_channel_compressed_inner.astype('uint8')
    return a_channel_compressed


# MAIN PROGRAM:
print('*** Image Compression using SVD - a demo')
a_red, a_green, a_blue_, original_image = open_image('lena.png')

# image width and height:
img_width = 512
img_height = 512

# number of singular values to use for reconstructing the compressed image
singular_values_limit = 160

a_red_compressed = compress_single_channel(a_red, singular_values_limit)
a_green_compressed = compress_single_channel(a_green, singular_values_limit)
a_blue_compressed = compress_single_channel(a_blue_, singular_values_limit)

imr = Image.fromarray(a_red_compressed, mode=None)
img = Image.fromarray(a_green_compressed, mode=None)
imb = Image.fromarray(a_blue_compressed, mode=None)

new_image = Image.merge("RGB", (imr, img, imb))

original_image.show()
new_image.show()

# CALCULATE AND DISPLAY THE COMPRESSION RATIO
mr = img_height
mc = img_width

original_size = mr * mc * 3
compressed_size = singular_values_limit * (1 + mr + mc) * 3

print('original size:')
print(original_size)

print('compressed size:')
print(compressed_size)

print('Ratio compressed size / original size:')
ratio = compressed_size * 1.0 / original_size
print(ratio)

print('Compressed image size is ' + str(round(ratio * 100, 2)) + '% of the original image ')
print('DONE - Compressed the image! Over and out!')
