import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()  # Sigmoid ensures outputs are scaled between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(data, encoding_dim=3, epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train an autoencoder on the given data and return the encoded features.
    """
    # Preprocess the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Convert data to PyTorch tensors
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

    # Initialize the model, loss function, and optimizer
    input_dim = data.shape[1]
    model = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data_tensor)
        loss = criterion(outputs, data_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Extract encoded features
    model.eval()
    with torch.no_grad():
        encoded_features = model.encoder(data_tensor).numpy()

    # Return the encoded features as a DataFrame
    encoded_df = pd.DataFrame(encoded_features, index=data.index, columns=[f"Feature_{i+1}" for i in range(encoding_dim)])
    return encoded_df