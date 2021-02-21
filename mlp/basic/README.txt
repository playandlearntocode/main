A tutorial on building, training and using Multilayer Perceptron neural network capable of distinguishing between circle and line images provided as input.

How to use:
images/ folder contains the training images. All the images are black and white, 16x16 pixels.
The script extract-image-features.py extracts the features from all the training images and stores them to file csv/input-file.txt

The script mlp.py trains two MLP neural networks:
MLP[0] - for recognizing circles
MLP[1] - for recognizing lines

When the training is completed, the script asks for a filename of the test image.
Valid test image names are:
tc-1.png (circle)
tc-2.png (circle)
tc-3.png (circle)
tc-4.png (circle)
tc-5.png (circle)
tl-1.png (line)
tl-2.png (line)
tl-3.png (line)
tl-4.png (line)
tl-5.png (line)

You can also create your own test images and add them to test-images/  folder.

After you enter a test image filename, the program will tell you whether it is a circle or a line!

For more information and the full tutorial, visit https://playandlearntocode.com.

Author: Goran Trlin

Find more tutorials and code samples on:
https://playandlearntocode.com