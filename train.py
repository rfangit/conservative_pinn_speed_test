import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Import functional module
from model import SimpleNet, GradNet
import matplotlib.pyplot as plt
import numpy as np
import time
import os

def train_network_with_test_loss(dimension, train_points, train_values, test_points, test_values, neurons, num_epochs=400, seed = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate model, loss function, and optimizer
    if seed is None:
        model = SimpleNet(dimension, neurons).to(device)  # Move model to GPU
    else:
        model = SimpleNet(dimension, neurons, seed).to(device)  # Move model to GPU
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Move data to GPU
    train_points = train_points.to(device)
    train_values = train_values.to(device)
    test_points = test_points.to(device)
    test_values = test_values.to(device)

    train_losses = []  # Store training loss at each epoch
    test_losses = []   # Store test loss at each epoch

    # Training loop
    for epoch in range(num_epochs):
        # Compute test loss before training loss
        with torch.no_grad():  # Disable gradient computation for test loss
            y_test_pred = model(test_points)
            test_loss = criterion(y_test_pred, test_values)
            test_losses.append(test_loss.item())
        
        # Training step
        optimizer.zero_grad()
        y_pred = model(train_points)
        train_loss = criterion(y_pred, train_values)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

    return train_losses, test_losses, model

def train_grad_network_with_test_loss(dimension, train_points, train_values, test_points, test_values, neurons, num_epochs=400, seed = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate model, loss function, and optimizer
    if seed is None:
        model = GradNet(dimension, neurons).to(device)  # Move model to GPU
    else:
        model = GradNet(dimension, neurons, seed).to(device)  # Move model to GPU
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Move data to GPU
    train_points = train_points.to(device)
    train_values = train_values.to(device)
    test_points = test_points.to(device)
    test_values = test_values.to(device)

    train_losses = []  # Store training loss at each epoch
    test_losses = []   # Store test loss at each epoch

    # Training loop
    for epoch in range(num_epochs):
        # Compute test loss
        optimizer.zero_grad()
        test_points.requires_grad = True
        y_test_pred = model(test_points)
        dydx_test = torch.autograd.grad(outputs=y_test_pred, inputs=test_points, grad_outputs=torch.ones_like(y_test_pred),
                                        create_graph=True)[0]
        test_loss = criterion(test_values, dydx_test)
        test_losses.append(test_loss.item())
        
        # Training step
        optimizer.zero_grad()
        train_points.requires_grad = True
        y_pred = model(train_points)
        dydx = torch.autograd.grad(outputs=y_pred, inputs=train_points, grad_outputs=torch.ones_like(y_pred),
                                   create_graph=True)[0]
        train_loss = criterion(train_values, dydx)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

    return train_losses, test_losses, model


def train_test_multiple_networks(train_network_function, dimension, train_points, neurons, num_epochs, train_values, test_points, test_values, num_trials):
    all_losses = []
    all_test_losses = []
    all_models = []

    start_time = time.time()
    for trial in range(num_trials):
        loss, test_loss, model = train_network_function(dimension, train_points, train_values, test_points, test_values, neurons, num_epochs, seed=trial)
        all_losses.append(loss)
        all_test_losses.append(test_loss)
        all_models.append(model)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"The network training took {elapsed_time:.2f} seconds.")

    all_losses = np.array(all_losses)
    mean_loss = np.mean(all_losses, axis=0)
    variance_loss = np.var(all_losses, axis=0)

    all_test_losses = np.array(all_test_losses)
    mean_test_loss = np.mean(all_test_losses, axis=0)
    variance_test_loss = np.var(all_test_losses, axis=0)
    return mean_loss, variance_loss, mean_test_loss, variance_test_loss

def save_plot_test_train(epochs, mean_loss, std_loss, mean_loss_grad, std_loss_grad,
                      mean_test_loss, std_test_loss, mean_test_loss_grad, std_test_loss_grad,
                      dimension, neurons, num_epochs, output_subfolder):
    """
    Saves a plot of training and test losses over epochs.

    Parameters:
        epochs (np.array): Array of epoch numbers.
        mean_loss (np.array): Mean training loss for the normal network.
        variance_loss (np.array): Variance of training loss for the normal network.
        mean_loss_grad (np.array): Mean training loss for the conservative network.
        variance_loss_grad (np.array): Variance of training loss for the conservative network.
        mean_test_loss (np.array): Mean test loss for the normal network.
        variance_test_loss (np.array): Variance of test loss for the normal network.
        mean_test_loss_grad (np.array): Mean test loss for the conservative network.
        variance_test_loss_grad (np.array): Variance of test loss for the conservative network.
        dimension (str): Dimension of the problem (e.g., "2D", "8D").
        neurons (int): Number of neurons in the layer.
        num_epochs (int): Total number of epochs.
        output_subfolder (str): Path to the folder where the plot will be saved.
    """
    plt.figure(figsize=(10, 5))

    # Plot training loss for normal network
    plt.plot(epochs, mean_loss, label="Training Loss (Standard NN)", color="blue", linewidth=2)
    plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color="blue", alpha=0.2)#, label="Variance (±1 std)")

    # Plot test loss for normal network
    plt.plot(epochs, mean_test_loss, label="Test Loss (Standard NN)", color="black", linewidth=2)
    plt.fill_between(epochs, mean_test_loss - std_test_loss, mean_test_loss + std_test_loss, color="black", alpha=0.2)#, label="Variance (±1 std)")

    # Plot training loss for conservative network
    plt.plot(epochs, mean_loss_grad, label="Training Loss (Hamiltonian NN)", color="red", linewidth=2)
    plt.fill_between(epochs, mean_loss_grad - std_loss_grad, mean_loss_grad + std_loss_grad, color="red", alpha=0.2)#, label="Variance (±1 std)")

    # Plot test loss for conservative network
    plt.plot(epochs, mean_test_loss_grad, label="Test Loss (Hamiltonian NN)", color="green", linewidth=2)
    plt.fill_between(epochs, mean_test_loss_grad - std_test_loss_grad, mean_test_loss_grad + std_test_loss_grad, color="green", alpha=0.2)#, label="Variance (±1 std)")

    # Set x and y limits
    plt.xlim([0, num_epochs])

    # Increase font sizes for labels and title
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title(f"MSE (±1 std), Network Hidden Layers {neurons}, {dimension} Dimensions", fontsize=16)

    # Make legend font size larger
    plt.legend(fontsize=12)

    # Remove whitespace around the plot
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(output_subfolder, f"{dimension}D_Conservative_NN.png")
    plt.savefig(plot_filename, bbox_inches='tight')  # Remove extra whitespace when saving
    plt.close()  # Close the plot to free memory
    print(f"Plot saved to {plot_filename}")