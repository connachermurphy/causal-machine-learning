import numpy as np
import torch.nn as nn
from tqdm import tqdm  # ascii progress bar


# Create a class for a DNN with ReLU activation, dropout layers, and Kaiming initialization
class DeepNeuralNetworkReLU(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, dropout_rate):
        super(DeepNeuralNetworkReLU, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            linear = nn.Linear(prev_size, hidden_size)
            nn.init.kaiming_normal_(linear.weight, mode="fan_in")
            self.layers.append(linear)
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        self.layers.append(nn.Linear(prev_size, output_dim))

    # Define the forward pass
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# take model as input
def train_dnn(
    dnn_features,
    structural_features,
    outcomes,
    structural_layer,
    model,
    loss_function,
    optimizer,
    num_epochs,
    noisily = False,
):
    loss_history = []  # keep track of the loss at each epoch

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        # Forward pass
        structural_parameters = model(dnn_features)
        outcomes_est = structural_layer(structural_parameters, structural_features)

        loss = loss_function(outcomes_est, outcomes)
        if np.isnan(float(loss)):
            print("NaN loss detected :(")
            print(loss)
            break

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()

        # Store loss
        loss_history.append(float(loss))
        
        # add a noisily option:
        if noisily:
            # Print the loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model
