import torch.nn as nn


class _AnchorLayer(nn.Module):

    def __init__(self):
        super(_AnchorLayer, self).__init__()

    def forward(self, x):
        pass

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
