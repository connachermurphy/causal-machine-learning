import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.linalg as linalg
from tqdm import tqdm  # ascii progress bar


# Early stopping class: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    # returns True if we should stop training, False otherwise
    def early_stop(self, validation_loss):
        # If validation loss is lower than the minimum validation loss, reset the counter
        if validation_loss < self.min_validation_loss:
            self.min_val_loss = validation_loss
            self.counter = 0

        # If validation loss is higher than the minimum validation loss + the delta adjustment
        # increment the counter
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            # If counter is greater than patience, return True
            if self.counter >= self.patience:
                return True
        return False


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


# A somewhat flexible training function
def train_dnn(
    dnn_features,
    structural_features,
    outcomes,
    structural_layer,
    model,
    loss_function,
    optimizer,
    num_epochs,
    include_loss_history=False,
    noisily=False,
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
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return (model, loss_history) if include_loss_history else model


# We don't need a structural layer for Lambda, so we just use the identity function
def identity(structural_parameters, structural_features):
    return structural_parameters


# Estimate conditional expectation of the Hessian for a given split and structural parameter DNN
def estimate_expected_hessian(
    splits,
    split,
    models_structural_parameters,
    model_sp,
    loss_function,
    structural_layer,
    hidden_sizes,
    dropout_rate,
    learning_rate,
    weight_decay,
    num_epochs,
):
    print(f"Split {split + 1} with structural parameter DNN {model_sp + 1}")

    # Number of observations in the split
    N = splits[split]["dnn_features"].size(0)

    # Number of DNN features
    dnn_features_dim = splits[split]["dnn_features"].size(1)

    # Evaluate the structural parameters on <split>, using the DNN from <model>
    structural_parameters = models_structural_parameters[model_sp](
        splits[split]["dnn_features"]
    )
    structural_parameters_dim = structural_parameters.size(1)

    # Predict outcomes for <split>
    outcomes_est = structural_layer(
        structural_parameters, splits[split]["structural_features"]
    )

    # Calculate loss
    loss = loss_function(outcomes_est, splits[split]["outcomes"])

    # Take the gradient of the loss w.r.t. to the structural parameters
    loss_grad = autograd.grad(
        loss, structural_parameters, create_graph=True, retain_graph=True
    )[0]

    # Initialize Hessian
    loss_hessian = torch.zeros(
        [N, structural_parameters_dim, structural_parameters_dim]
    )

    # Iterate over the elements of the Hessian
    # We can just calculate the upper triangle and reflect to the lower triangle
    for k in range(structural_parameters_dim):
        loss_hessian_row = autograd.grad(
            loss_grad[:, k].sum(), structural_parameters, retain_graph=True
        )[0]

        for j in range(k, structural_parameters_dim):
            print(f"Element ({k}, {j})")

            # Extract Hessian element (k,j)
            loss_hessian_element = loss_hessian_row[:, j].view(N, 1)

            # Initialize neural network
            model = DeepNeuralNetworkReLU(
                input_dim=dnn_features_dim,
                hidden_sizes=hidden_sizes,
                output_dim=1,
                dropout_rate=dropout_rate,
            )

            # Initialize optimizer; we use stochastic gradient descent
            optimizer = optim.SGD(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

            loss_hessian_element_projection = train_dnn(
                splits[split]["dnn_features"],
                splits[split]["structural_features"],
                loss_hessian_element,
                identity,  # no structural layer for the hessian elements
                model,
                nn.MSELoss(reduction="mean"),
                optimizer,
                num_epochs,
            )

            # Store projection
            loss_hessian[:, k, j] = loss_hessian_element_projection(
                splits[split]["dnn_features"]
            ).view(N)

            if k != j:  # reflect upper triangle to lower triangle
                print(f"Reflecting to element ({j}, {k})")
                loss_hessian[:, j, k] = loss_hessian[:, k, j]

    print("\n")
    return loss_hessian


def estimate_influence_function(
    splits,
    split,
    models_structural_parameters,
    model_sp,
    loss_function,
    models_expected_hessians,
    model_eh,
    structural_layer,
    statistic,
):
    # Number of observations in the split
    N = splits[split]["dnn_features"].size(0)

    # Evaluate the structural parameters on <split>, using the DNN from <model>
    structural_parameters = models_structural_parameters[model_sp](
        splits[split]["dnn_features"]
    )
    structural_parameters_dim = structural_parameters.size(1)

    # Predict outcomes for <split>
    outcomes_est = structural_layer(
        structural_parameters, splits[split]["structural_features"]
    )

    # Calculate loss
    loss = loss_function(outcomes_est, splits[split]["outcomes"])

    # Take the gradient of the loss w.r.t. to the structural parameters
    loss_grad = autograd.grad(
        loss, structural_parameters, create_graph=True, retain_graph=True
    )[0].view(N, structural_parameters_dim)

    # Evaluate statistic
    statistic_est = statistic(
        splits[split]["dnn_features"],
        structural_parameters,
        splits[split]["structural_features"],
    )
    statistic_dim = statistic_est.size(1)

    # Calculate the gradient of the statistic w.r.t. the structural parameters
    # CM: I need to accommodate a multivariate statistic
    statistic_grad = autograd.grad(
        statistic_est.sum(), structural_parameters, create_graph=True
    )[0].view(N, structural_parameters_dim)

    # Calculate the influence function
    influence_function_est = statistic_est - torch.matmul(
        torch.matmul(
            statistic_grad.view(N, statistic_dim, structural_parameters_dim),
            linalg.pinv(models_expected_hessians[model_eh]),
        ).view(N, statistic_dim, structural_parameters_dim),
        loss_grad.view(N, structural_parameters_dim, statistic_dim),
    ).view(N, statistic_dim)

    return influence_function_est
