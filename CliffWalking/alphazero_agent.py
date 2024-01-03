import torch.nn as nn

class Alphazero_Network(nn.Module):
    def __init__(self):
        super(Alphazero_Network, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(2, 64)  # The input size is 2 and output to next layer is 64. This is arbitrary and can be adjusted.
        self.fc2 = nn.Linear(64, 64) # Second hidden layer

        self.output1 = nn.Linear(64, 4) # Output layer with 4 neurons
        self.output2 = nn.Linear(64, 1) # Output layer with 1 neuron

    def forward(self, x):
        # Hidden layers with ReLU activation
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        
        # Output layers
        out1 = nn.Softmax(dim=1)(self.output1(x))
        out2 = self.output2(x)
        
        return out1, out2
