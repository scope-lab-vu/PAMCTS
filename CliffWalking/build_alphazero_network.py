import torch
import torch.nn as nn
import torch.nn.functional as F


ENV_WIDTH = 12
ENV_HEIGHT = 4


class Alphazero_Network(nn.Module):
    def __init__(self):
        super(Alphazero_Network, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear((ENV_WIDTH * ENV_HEIGHT,), 64)  # The input size is 1 and output to next layer is 64. This is arbitrary and can be adjusted.
        self.fc2 = nn.Linear(64, 64) # Second hidden layer
        self.fc3 = nn.Linear(64, 64) # Third hidden layer
        self.fc4 = nn.Linear(64, 64) # Third hidden layer
        self.fc5 = nn.Linear(64, 64) # Third hidden layer

        self.output1 = nn.Linear(64, 3) # Output layer with 3 neurons
        self.output2 = nn.Linear(64, 3) # Output layer with 1 neuron
    
    def forward(self, x):
        # convert x to index tensor
        # x = F.one_hot(x, num_classes=ENV_WIDTH*ENV_HEIGHT).float()

        # Flatten the tensor if it has more than 2 dimensions
        # if len(x.shape) > 2:
        #     x = x.view(x.size(0), -1)

        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        
        # Output layers
        out1 = nn.Softmax(dim=1)(self.output1(x))
        out2 = self.output2(x)
        
        return out1, out2
    