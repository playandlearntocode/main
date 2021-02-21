from PIL import Image
#
# Load an image and compute its features:
# Feature 1 - total percentage of black pixels in the image
# Feature 2 - percentage of black pixels around image center
# Feature 3 - percentage of black pixels near the image borders

class ImageFeatureExtractor:

    image_width = 16
    image_height = 16

    color_threshold = 128

    # load image from a file path into memory. Get the pixel array.
    def load_image(self, file_path):
        im = Image.open(file_path)
        pixels = im.load()
        return (im,pixels)

    # take the loaded image and extract feature values.
    def extract_features(self, image_object, pixels):
        print('Extracting features...')
        feature1 = round(self.compute_feature_1(pixels),2)
        feature2 = round(self.compute_feature_2(pixels),2)
        feature3 = round(self.compute_feature_3(pixels),2)

        #boost feature values:
        # feature1 *= 4
        # feature2 *= 4
        # feature3 *= 4

        return (feature1,feature2,feature3)

    # total percentage of black pixels
    def compute_feature_1(self, pixels):
        pixel_worth = 1.0 / (self.image_height * self.image_width)
        val = 0
        for i in range(0, self.image_height -1):
            for j in range(0, self.image_width - 1):
                avg = ( pixels[i,j][0] + pixels[i,j][1] + pixels[i,j][2] ) / 3.0

                if(avg < self.color_threshold ):
                    val = val + pixel_worth
        return val

    # returns True if the pixel is in the central part of the image:
    def __is_in_cental_part(self,x,y):

        x_start = self.image_width / 2 - 2
        x_end = self.image_width / 2  +  2

        y_start = self.image_height / 2  -  2
        y_end = self.image_height / 2  +  2

        if x >= x_start and x <= x_end and y >= y_start and y <= y_end:
            return True
        return False

    # percentage of black central_pixels
    def compute_feature_2(self, pixels):
        val = 0
        central_pixel_worth = 1.0 / (4*4)
        for i in range(0, self.image_height -1):
            for j in range(0, self.image_width - 1):
                avg = ( pixels[i,j][0] + pixels[i,j][1] + pixels[i,j][2] ) / 3.0

                if(avg < self.color_threshold ):
                    if(self.__is_in_cental_part(i,j) == True):
                        val += central_pixel_worth

        return val

    # returns True if the pixel is near the border:
    def __is_near_border(self,x,y):

        x_left = 3
        x_right = self.image_width -3

        y_top = 3
        y_bottom = self.image_height -3

        if (x <= x_left or x >= x_right) and (y <= y_top or y >= y_bottom):
            return True
        return False

    # percentage of near-border black pixels:
    def compute_feature_3(self, pixels):
        val = 0
        border_pixel_worth = 1.0 / (4 * 48 - 4*9)
        for i in range(0, self.image_height -1):
            for j in range(0, self.image_width - 1):
                avg = ( pixels[i,j][0] + pixels[i,j][1] + pixels[i,j][2] ) / 3.0

                if(avg < self.color_threshold ):
                    if(self.__is_near_border(i,j) == True):
                        val += border_pixel_worth

        return val
