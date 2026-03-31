from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from loss import softmax_cross_entropy_forward_backward, safe_softmax

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

def build_mnist_model():
    return Sequential([
        Conv2D(1, 8, 3, 3, stride=1, pad=1),     # [N,1,28,28] -> [N,8,28,28]
        ReLU(),                                     # [N,8,28,28]
        MaxPool2D(2, 2, stride=2),                  # [N,8,14,14]
        Conv2D(8, 16, 3, 3, stride=1, pad=1),      # [N,16,14,14]
        ReLU(),                                     # [N,16,14,14]
        MaxPool2D(2, 2, stride=2),                  # [N,16,7,7]
        Flatten(),                                  # [N,784]
        Dense(784, 10),                             # [N,10]
    ])
