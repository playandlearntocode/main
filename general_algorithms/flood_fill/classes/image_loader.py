from PIL import Image


class ImageLoader:
    def load_image(self, file_path):
        '''
        Load image from a file path into memory. Get the pixel array.
        '''
        im = Image.open(file_path)
        pixels = im.load()
        return (im, pixels)
