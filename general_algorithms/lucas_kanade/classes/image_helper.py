from PIL import Image, ImageDraw

class ImageHelper:
    def load_image(self, file_path):
        '''
        Load image from a file path into memory. Get the pixel array.
        '''
        im = Image.open(file_path)
        pixels = im.load()
        return (im, pixels)

    def draw_rectangle(self, img, xy,fill, outline=None):
        d = ImageDraw.Draw(img)
        d.rectangle(xy,fill,outline)
        return img

    def draw_line(self, img, coord, fill_color):
        draw = ImageDraw.Draw(img)
        draw.line(coord, fill=fill_color)