# MNIST-CNN-Python

Small neural network library which can dynamically initialise different network structures, including conv networks. Currently implemented is an MNIST solution.
Neural network implementations are not given by any other API or library and are fully defined here.

Results max out at around 90% accuracy(Terrible) - an increase from 60% until implenting He Initialization using the solution provided in "On weight initialization in deep neural networks"(https://arxiv.org/pdf/1704.08863.pdf)
Solution uses ReLu -> Softmax with cross entropy loss, still maxing out at 90%
