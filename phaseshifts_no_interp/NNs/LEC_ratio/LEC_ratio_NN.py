#import sys
#sys.path.append('/Users/pleazy/PycharmProjects/Proposal/venv/lib/python3.11/site-packages/torch')
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # choose the hardware accelerator

data = np.loadtxt('../../random_sampling/phaseshifts_SLLJT_00001_lambda_2.00_s5.dat')

LECS = data[:, 0:4]
ratios = data[:,4]


###returns 1 if phaseshift is within 68% boundary of EKM uncertainty band, 0 else
for ratio in ratios:
    if ratio > 0.68:
        ratio = 1
    else:
        ratio = 0

###normalizes all singular values to the first one
for u in range(len(LECS[:, 0])):
    for LEC in LECs[u,:]:
        LEC = LECs[u,:]/data[u,0]


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # Data samples, e.g., images
        self.labels = labels  # Labels for each sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],  # A data sample (e.g., an image)
            'label': self.labels[idx]  # Corresponding label for the sample
        }
        return sample


data = LECS
labels = ratios

training_data = CustomDataset(data=data[0:90], labels=labels[0:90])
test_data = CustomDataset(data=data[90:100], labels=labels[90:100])

batch_size = 100
trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i - 1], hidden_dims[i]) for i in range(1, len(hidden_dims))
        ])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = nn.functional.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        x = self.output_layer(x)
        return x



## Training loop

def train_model(model, trainloader, testloader, loss_fn, optimizer, num_epochs):

    """
    Trains the model for num_epochs epochs and returns the test and train losses.
    """

    # Define lists to store the training and testing losses
    train_losses = []
    test_losses = []

    # Train the model
    for epoch in range(num_epochs):
        # Train the model on the training set
        train_loss = 0.0
        for i, data in enumerate(trainloader):
            # Get the inputs and targets
            inputs, targets = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs.to(device))

            # Compute the loss
            loss = loss_fn(outputs, targets.to(device))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Add the batch loss to the epoch loss
            train_loss += loss.item()

        # Compute the average training loss for the epoch
        train_loss /= len(trainloader)
        train_losses.append(train_loss)

        # Test the model on the test set
        test_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                # Get the inputs and targets
                inputs, targets = data

                # Forward pass
                outputs = model(inputs.to(device))

                # Compute the loss
                loss = loss_fn(outputs, targets.to(device))

                # Add the batch loss to the epoch loss
                test_loss += loss.item()

        # Compute the average test loss for the epoch
        test_loss /= len(testloader)
        test_losses.append(test_loss)
        if epoch % 5 == 0:
        # Print the epoch number and loss
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return train_losses, test_losses




# Example I:

input_dim = 4  # Dimension of input (Collision data)
output_dim = 2  # Dimension of output (v'_1 and v'_2)
hidden_dims = [8]   # List of hidden layer widths

# Create MLP model and optimizer
model = MLP(input_dim, output_dim, hidden_dims).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define the loss function
loss_fn = nn.MSELoss()

# Set the number of epochs to train for
num_epochs = 1000

train_losses, test_losses = train_model(model, trainloader,
                                        testloader, loss_fn,
                                        optimizer, num_epochs)

plt.figure(figsize= (7,7))
plt.plot(np.arange(len(test_losses)),test_losses, label = f"layers({hidden_dims})")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.title("Evolution of model's training")