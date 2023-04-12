import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def readMnist(num_images, traintest):
    
    #Images
    f = gzip.open('./mnistdata/{}-images-idx3-ubyte.gz'.format(traintest),'r')

    image_size = 28

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)

    images = [np.asarray(data[x] / 255.0).squeeze() for x in range(num_images)]

    #Labels
    f = gzip.open('./mnistdata/{}-labels-idx1-ubyte.gz'.format(traintest),'r')
    f.read(8)
    buf = f.read(num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return images, labels


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
        self.input = input.copy()
        self.output = self.biases.copy()

        for i in range(self.input_depth):
            for j in range(self.depth):
                self.output[i] += signal.correlate2d(input[j], self.kernels[i, j], "valid")

        return self.output
    

    def backwards_prop(self, output_gradients, learning_rate):

        input_gradients = self.input.copy()
      
        for i in range(self.input_depth):
            for j in range(self.depth):
                self.kernel[i, j] -= learning_rate * signal.correlate2d(self.input[j], output_gradients[i], "valid")
                input_gradients[i] = signal.convolve2d(output_gradients[i], self.kernel[i, j], "full")
      
        self.biases[i] -= learning_rate * output_gradients

        return -learning_rate * signal.convolve2d()

class MaxPooling:
    def __init__(self, shape, stride):
        self.shape = shape
        self.stride = stride 

    def forward(self, input, input_shape):
        output_shape = (input_shape[0], int(input_shape[1] / self.shape[0]), int(input_shape[2] / self.shape[1]))
        
        outputs = np.zeros(output_shape)

        for m in range(input_shape[0]): #For each matrix
            for i in range(0, input_shape[1], self.shape[0]): #Height of the matrix
                for j in range(0, input_shape[2], self.shape[1]): #Width of the matrix
                    outputs[m, int(i / self.shape[0]), int(j / self.shape[1])] = max(
                        input[m, i, j], input[m, i + 1, j], input[m, i, j + 1], input[m, i + 1, j + 1]
                    )

        return outputs
    
    def backward(output_gradients):
        pass

class DenseLayer:
    def __init__(self, input_shape, output_shape, activation_func):

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.activations = np.random.randn(output_shape)
        self.biases = np.random.randn(output_shape)
        self.weights_shape = (output_shape, input_shape)
        self.weights = np.random.randn(self.weights_shape)
        self.activation_func = activation_func

        self.sums = np.zeros(output_shape)
    

    def forward(self, input):
        sums = np.matmul(input, self.weights) + self.biases
        self.activations = self.activation_func(sums, True)
        return self.activations
    
    def backward(self, output_gradients, learning_rate):
        
        next_output_gradients = np.zeros(output_gradients.shape)

        
            
                
    
def ReLU(inp, no_der):
    output = np.zeros(inp.shape)
    if no_der:
        #Standard function not the derivative
        output = np.array([(x > 0) * x for x in inp])
    else:
        #Derivative
        output = np.array([x > 0  for x in inp])

    return 0

def small_softmax(inp, sums):
    
    output = np.exp(inp) / sums
    return output

def softmax(inp, no_der):
    output = np.zeros(inp.shape)
    exps = np.exp(inp)
    sums = np.sum(exps)

    if no_der:
        #Standard function not the derivative
        output = exps / sums
    else:
        #Derivative
        for i in range(len(inp)):
            for j in range(len(inp)):
                if i == j:
                    #DjSi = Sj(1 - Si)
                    #Although i = j therefore Sj = Si
                    output[i] = exps[j] / sums * (1.0 - exps[i] / sums)
                else:
                    #DjSi = -SiSj
                    output[i] = -1.0 * ((exps[i] / sums) * (exps[j] / sums))
    
    return output

'''
Try network structure of:
Input layer 28x28 grayscale 0 - 1
Conv layer
Max pooling 2x2 stride 2
Conv layer
Conv layer
flatten layer
dense layer
dense layer
output 0 - 9
'''

#images, labels = readMnist(60000, "train")

''' Testing for max pooling
test_input = np.array([[[5, 9, 8, 7],
                       [3, 1, 5, 7],
                       [9, 0, 8, 7],
                       [3, 1, 2, 1]]])
print(test_input)
layer = MaxPooling((2, 2), 2)
print(layer.forward(test_input, (1, 4, 4)))
'''