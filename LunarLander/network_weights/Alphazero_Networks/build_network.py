import torch.nn as nn

class Alphazero_Network(nn.Module):
    def __init__(self):
        super(Alphazero_Network, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(8, 128)  # The input size is 8 and output to next layer is 64. This is arbitrary and can be adjusted.
        self.fc2 = nn.Linear(128, 128) # Second hidden layer
        self.fc3 = nn.Linear(128, 128) # Third hidden layer
        self.fc4 = nn.Linear(128, 128) # Fourth hidden layer
        self.fc5 = nn.Linear(128, 128) # Fifth hidden layer

        self.output1 = nn.Linear(128, 4) # Output layer with 4 neurons
        self.output2 = nn.Linear(128, 1) # Output layer with 1 neuron

    def forward(self, x):
        # Hidden layers with ReLU activation
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        x = nn.ReLU()(self.fc4(x))
        x = nn.ReLU()(self.fc5(x))
        
        # Output layers
        out1 = nn.Softmax(dim=1)(self.output1(x))
        out2 = self.output2(x)
        
        return out1, out2
