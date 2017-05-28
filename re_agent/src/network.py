import numpy as np
import cupy as cp
import chainer
from chainer import functions as F
from chainer import links as L

class NatureDQNHead(chainer.ChainList):
    """DQN's head (Nature version)"""

    def __init__(self, n_input_channels=4, n_output_channels=512,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 32, 8, stride=4, bias=bias),
            L.Convolution2D(32, 64, 4, stride=2, bias=bias),
            L.Convolution2D(64, 64, 3, stride=1, bias=bias),
            L.Linear(3136, n_output_channels, bias=bias),
        ]

        super(NatureDQNHead, self).__init__(*layers)

    def __call__(self, state):
        # if type(state) == cp.core.core.ndarray:
        #     state = cp.asanyarray(state, dtype=cp.float32)
        # else:
        state = np.asanyarray(state, dtype=np.float32)
        h = state / 255.0
        for layer in self:
            h = self.activation(layer(h))
        return h
