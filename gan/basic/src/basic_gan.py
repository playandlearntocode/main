'''
A Generative Adversarial Network (GAN) example in Keras (Python)
The program performs generation of new hand-drawn circles based on the provided small input dataset
Author: Goran Trlin
Find more tutorials and code samples on:
https://playandlearntocode.com
'''

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # switch to CPU processing
import keras
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, LeakyReLU, Reshape
from keras import backend
from collections import namedtuple
from matplotlib import pyplot
from classes.extraction.folder_helper import FolderHelper

# image properties:
image_height = 16
image_width = 16
number_of_channels = 1  # grayscale input images

# network properties:
batch_size = 40
image_input_object = (image_height, image_width, number_of_channels)  # (16,16,1) - 1 channel / grayscale
latent_space_dimensions = 2  # latent space dimensionality
iterations_discriminator = 500  # Discriminator epochs count
iterations_gan = 25  # GAN epochs count
pixel_val_threshold = 0.0  # dynamically computed, used for post processing / filtering
pause_every_n_iterations = 600  # not really used, only for long runs
plot_every_nth_iteration = 1  # if 1, a test image will be drawn after every iteration


def post_process_image(img):
    '''
    Sharpen the resulting image, filter based on pixel value
    :param img:
    :return:
    '''
    pixel_val_threshold = np.average(img)
    pixel_val_threshold -= 0.05

    after_repaint = np.where(img > pixel_val_threshold, 1.0, img)
    return after_repaint


def fetch_latent_space_points(latent_space_dim, count):
    '''
    Samples from latent space
    :param latent_space_dim:
    :param count:
    :return:
    '''
    x = np.random.rand(latent_space_dim * count)
    return x.reshape((count, latent_space_dim))


def fetch_real_images(count):
    '''
    Returns <count> real images
    :param count:
    :return:
    '''
    fh = FolderHelper()
    input_images_grayscale = fh.process_images_folder('../images_circles', 16,
                                                      16)  # returns grayscale images, 1 channel (16,16,1)
    x_training = input_images_grayscale[:count]
    y_training = np.ones(count)

    ret_obj = namedtuple('inputs_outputs', ['inputs', 'outputs'])
    return ret_obj(x_training[:count], y_training[:count])


def fetch_random_images(image_width, image_height, count):
    '''
    Creates random / fake images without using the generator model, just plan noise:
    :param image_width:
    :param image_height:
    :param count:
    :return:
    '''
    randoms = np.random.rand(count * image_width * image_height)

    x_random = randoms.reshape((count, image_height, image_width, 1))
    y_random = np.zeros(count)
    ret_obj = namedtuple('inputs_outputs', ['inputs', 'outputs'])
    return ret_obj(x_random[:count], y_random[:count])


def fetch_fake_images(model_generator, count, latent_space_dim):
    '''
    Uses the generator to create new images
    At the beginning, these are fake images, but as the generator model gets trained , these will become more realistic
    :param model_generator:
    :param count:
    :param latent_space_dim:
    :return:
    '''
    randoms = np.random.rand(count, latent_space_dim)
    x_fake = randoms
    y_fake = model_generator.predict(x_fake)

    ret_obj = namedtuple('inputs_outputs', ['inputs', 'outputs'])
    return ret_obj(x_fake[:count], y_fake[:count])


def make_discriminator(input_object_shape):
    '''
    Creates the discrimintor model, which is a standard CNN network
    :param input_object_shape:
    :return:
    '''
    model = Sequential()
    layers = [
        # 1st convolutional layer
        Conv2D(64, kernel_size=(3, 3), strides=(2, 2), input_shape=input_object_shape),
        LeakyReLU(alpha=0.25),
        Flatten(),
        Dense(1, activation='sigmoid'),
    ]

    # attach layers to the model:
    for layer in layers:
        model.add(layer)

    # compile the model:
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model


def make_generator(latent_space_dim):
    '''
    Creates the generator model.
    Conv2Transpose performs upscaling. With strides = (2,2) it doubles both width and height of the image
    :param latent_space_dim:
    :return:
    '''
    model = Sequential()
    layers = [
        Dense(32 * 16 * 16, input_dim=latent_space_dim),
        Reshape((16, 16, 32)),
        Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same'),
        Conv2D(1, (4, 4), strides=(2, 2), padding='same')
    ]

    # attach layers to the model:
    for layer in layers:
        model.add(layer)

    return model


def make_gan(model_discriminator, model_generator):
    '''
    GAN model is created by combining the generator and discriminator. Discriminator needs to be marked untrainable ( at this point, it is already trained )
    :param model_discriminator:
    :param model_generator:
    :return:
    '''
    model = Sequential()
    model_discriminator.trainable = False
    model.add(model_generator)
    model.add(model_discriminator)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())
    return model


def train_discriminator(model_discriminator, batch_size, iterations):
    '''
    Training the discriminator CNN to be able to distinguish between real and fake images.
    :param model_discriminator:
    :param batch_size:
    :param iterations:
    :return:
    '''
    for i in range(iterations):
        fake_data = fetch_random_images(16, 16, int(batch_size / 2))
        real_data = fetch_real_images(int(batch_size / 2))

        loss_fake = model_discriminator.train_on_batch(fake_data.inputs, fake_data.outputs, return_dict=True)
        loss_real = model_discriminator.train_on_batch(real_data.inputs, real_data.outputs, return_dict=True)

        print('Discriminator loss on real images:')
        print(loss_real)
        print('Discriminator loss on fake images:')
        print(loss_fake)


def train_gan(model_gan, model_discriminator, model_generator, latent_space_dim, batch_size, iterations):
    '''
    Since the discriminator model is already trained, only generator gets trained here.
    Note that we use ones as the output.

    :param model_discriminator:
    :param batch_size:
    :param iterations:
    :return:
    '''
    for i in range(iterations):
        fake_batch = int(batch_size / 2)
        real_batch = batch_size - fake_batch
        fake_data = fetch_fake_images(model_generator, fake_batch, latent_space_dim)
        real_data = fetch_real_images(real_batch)

        # combine the fake and real data:
        x_combined = np.vstack((fake_data.outputs, real_data.inputs))
        y_combined = np.hstack((np.zeros(fake_batch), np.ones(real_batch)))

        # double the input dataset:
        x_combined = np.vstack((x_combined, x_combined))
        y_combined = np.hstack((y_combined, y_combined))

        disc_loss = model_discriminator.train_on_batch(x_combined, y_combined)
        print('Discriminator loss:')
        print(disc_loss)

        y_output = np.hstack((np.ones(fake_batch), np.ones(real_batch)))

        latent_points = fetch_latent_space_points(latent_space_dim, batch_size)
        gan_loss = model_gan.train_on_batch(latent_points, y_output)

        print('GAN loss:')
        print(gan_loss)

        if (i % plot_every_nth_iteration == 0):
            test_images = model_generator.predict(np.array([latent_points[0]]))  # returns an array with only one image
            first_test_image = test_images[0]  # extract the image from the array
            after_repaint = post_process_image(first_test_image)

            pyplot.imshow(after_repaint, cmap='gray')
            pyplot.show()

        # optional pausing after N epochs / iterations:
        if (i % pause_every_n_iterations == 0 and i > 0):
            print('Pausing for a moment...')
            time.sleep(10)


def main():
    print('GAN example starting....')
    if backend.image_data_format() != 'channels_last':
        raise Exception('This example expects a channels_last backend,i.e. TensorFlow')

    model_discriminator = make_discriminator(image_input_object)
    model_generator = make_generator(latent_space_dimensions)

    print('*** STARTING THE TRAINING OF THE DISCRIMINATOR MODEL ***')
    train_discriminator(model_discriminator, batch_size, iterations_discriminator)
    print('*** TRAINING OF THE DISCRIMINATOR MODEL COMPLETED ***')

    random_image_objs = fetch_random_images(16, 16, 5).inputs

    real_image_objs = fetch_real_images(5).inputs
    # print('Real image shape:')
    print(real_image_objs.shape)

    print('Summary of the discriminator model before the GAN was created')
    model_discriminator.summary()

    prediction = model_discriminator.predict(np.array(random_image_objs))
    print('Discriminator model output for random images:')
    print(prediction)

    prediction = model_discriminator.predict(np.array(real_image_objs))
    print('Discriminator model output for random images:')
    print(prediction)

    print('*** ENTERING THE GAN PHASE ***')

    model_gan = make_gan(model_discriminator, model_generator)

    print('Summary of the discriminator model after the GAN was created')
    model_discriminator.summary()
    train_gan(model_gan, model_discriminator, model_generator, latent_space_dimensions, batch_size, iterations_gan)

    print('*** GAN TRAINING COMPLETED ***')

    test_image_count = 5  # total number of test images to draw from the trained generator model

    test_latent_points = fetch_latent_space_points(latent_space_dimensions, test_image_count)
    test_images = model_generator.predict(np.array(test_latent_points))

    for i in range(test_image_count):
        print('Plotting ' + str(i + 1) + '. test image...')
        # pyplot.imshow(test_images[i], cmap='gray') # original, non-processed version
        pyplot.imshow(post_process_image(test_images[i]), cmap='gray')
        pyplot.show()

    print('GAN example completed!')


main()
