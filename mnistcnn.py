import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def readMnist(num_images):
    
    #Images
    f = gzip.open('./mnistdata/train-images-idx3-ubyte.gz','r')

    image_size = 28

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)

    images = [np.asarray(data[x]).squeeze() for x in range(num_images)]

    #Labels
    f = gzip.open('./mnistdata/train-labels-idx1-ubyte.gz','r')
    f.read(8)
    buf = f.read(num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return images, labels

images, labels = readMnist(60000)

class ConvLayer:
    def __init__(self, depth, input_depth, input_shape, kernel_size):
        self.depth = depth
        self.input_depth = input_depth
        self.input_shape = input_shape
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)
        self.output_shape = (depth, input_shape[1] - kernel_size + 1, input_shape[2] - kernel_size + 1)
        self.kernels = np.random.randn(self.kernel_shape)
        self.biases = np.random.randn(self.output_shape)

    def forwards_prop(self, input):
        self.input = input
        self.output = self.biases

        for i in range(self.input_depth):
            for j in range(self.depth):
                self.output[i] += signal.convolve2d(input[j], self.kernels[i, j], "valid")

        return self.output
    

    def backwards_prop(self, output_gradients, learning_rate):

        input_gradients = self.input
        for i in range(self.input_depth):
            for j in range(self.depth):
                self.kernel[i, j] -= learning_rate * signal.correlate2d(self.input[j], output_gradients[i], "valid")
                input_gradients[i] = signal.convolve2d(output_gradients[i], self.kernel[i, j], "full")
        self.biases[i] -= learning_rate * output_gradients

        return -learning_rate * signal.convolve2d()