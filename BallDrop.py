import torch.nn as nn
import torch.nn.functional as F


class BallDropNet(nn.Module):

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # 5 hidden layers
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, hidden_size)
        self.linear7 = nn.Linear(hidden_size, out_size)

    def forward(self, input_vector):
        # Apply the layer to the input vector
        # The activation function is ReLu
        out = self.linear1(input_vector)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        out = F.relu(out)
        out = self.linear6(out)
        out = F.relu(out)
        out = self.linear7(out)
        return out

    def training_step(self, inputs, outputs):
        loss_function = F.mse_loss  # Uses mean square error loss
        out = self(inputs)  # Generate predictions
        loss = loss_function(out, outputs)  # Calculate loss
        return loss
