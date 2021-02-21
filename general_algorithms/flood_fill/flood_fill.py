'''
Flood Fill algorithm example in Python
Counting islands on a green/white map (image) using flood fill algorithm.
Green areas represent islands and white areas represent sea.
Also, it is assumed that the whole map is surrounded by sea.

Author: Goran Trlin
Find more tutorials and code samples on:
https://playandlearntocode.com
'''

from classes.image_loader import ImageLoader
from classes.map_reader import MapReader

def main():
    image_width, image_height = 400,400
    color_threshold = 254 # sea = white area, islands = green areas

    img_loader = ImageLoader()
    (loaded_image, pixels) = img_loader.load_image('images/map3.png')

    reduced_map = [] # map of 1s and 0s. 1 - island area, 0 - sea area
    for i in range(image_height):
        reduced_map.append([])
        for j in range(image_width):
            reduced_map[i].append(None)

    for x in range( image_width ):
        for y in range(image_height ):
            avg = (pixels[x,y][0] + pixels[x,y][1] + pixels[x,y][2]) / 3.0
            if (avg < color_threshold):
                reduced_map[y][x] = 1
            else:
                reduced_map[y][x] = 0

    map_reader = MapReader()
    islands = map_reader.count_islands(reduced_map)

    print('Number of islands on this map is:' + str(islands))

# RUN:
main()
