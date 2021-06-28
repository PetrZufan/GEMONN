from matplotlib import pyplot as plt
import numpy as np

def to_image(input):
    image = np.array(input, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()



#    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
#    first_image = mnist.test.images[0]
#    first_image = np.array(first_image, dtype='float')
#    pixels = first_image.reshape((28, 28))
#    plt.imshow(pixels, cmap='gray')
#    plt.show()