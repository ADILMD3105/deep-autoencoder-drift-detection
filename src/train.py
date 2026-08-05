import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import custom modules for data handling and model
from src.data_loader import load_data, preprocess_data, split_baseline_stream
from src.autoencoder import Autoencoder


def train_autoencoder(
    data_path,
    epochs=20,
    batch_size=256,
    learning_rate=1e-3
):

    # STEP 1: Load and preprocess data

    # Load dataset from given path (CSV, TXT, XLSX)
    df = load_data(data_path)

    # Clean, normalize and convert data into numerical format
    data, scaler = preprocess_data(df)

    # Split dataset into baseline (training) and stream (future data)
    baseline_data, _ = split_baseline_stream(data)



    # STEP 2: Convert data to PyTorch tensors

    # Convert baseline data into tensor format for model training
    baseline_tensor = torch.tensor(baseline_data, dtype=torch.float32)

    # Create dataset and dataloader for batch processing
    dataset = TensorDataset(baseline_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



    # STEP 3: Model initialization

    # Input dimension = number of features
    input_dim = baseline_tensor.shape[1]

    # Initialize autoencoder model
    model = Autoencoder(input_dim)

    # Define loss function (Mean Squared Error for reconstruction)
    criterion = nn.MSELoss()

    # Define optimizer (Adam for efficient training)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    # STEP 4: Training loop

    # Set model to training mode
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        # Iterate over batches
        for batch in dataloader:
            x = batch[0]

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass: reconstruct input
            reconstructed = model(x)

            # Compute reconstruction error
            loss = criterion(reconstructed, x)

            # Backpropagation
            loss.backward()

            # Update model weights
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)

        # Print training progress
        print(f"Epoch [{epoch+1}/{epochs}] - Reconstruction Loss: {avg_loss:.6f}")



    # STEP 5: Save trained model

    # Save model weights for later use in drift detection
    torch.save(model.state_dict(), "../autoencoder.pth")

    print("\nModel saved successfully as autoencoder.pth")

    # Return model and scaler for reuse
    return model, scaler



# MAIN ENTRY POINT


if __name__ == "__main__":
    # Ensure dataset path is provided via command line
    if len(sys.argv) < 2:
        print("Usage: python train.py <path_to_csv>")
        sys.exit(1)

    # Read dataset path from command line argument
    data_path = sys.argv[1]

    # Train the autoencoder model
    train_autoencoder(data_path)