import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_data, preprocess_data, split_baseline_stream
from src.autoencoder import Autoencoder


def compute_reconstruction_error(model, data):
    # set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # convert data to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # reconstruct input
        reconstructed = model(data_tensor)

        # calculate MSE for each sample
        errors = torch.mean((reconstructed - data_tensor) ** 2, dim=1)

    return errors.numpy()


def main(data_path):
    # load and preprocess data
    df = load_data(data_path)
    data, _ = preprocess_data(df)

    # split into baseline and stream
    baseline, stream = split_baseline_stream(data)

    # load trained model
    input_dim = baseline.shape[1]
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load("../autoencoder.pth"))
    model.eval()

    # compute reconstruction errors
    baseline_errors = compute_reconstruction_error(model, baseline)
    stream_errors = compute_reconstruction_error(model, stream)

    # set threshold using baseline (95th percentile)
    threshold = np.percentile(baseline_errors, 95)

    # detect drift (if error > threshold)
    drift_flags = stream_errors > threshold
    drift_ratio = drift_flags.mean()

    print(f"Drift threshold: {threshold:.6f}")
    print(f"Percentage of stream flagged as drift: {drift_ratio * 100:.2f}%")

    # plot reconstruction error
    plt.figure(figsize=(10, 4))
    plt.plot(stream_errors, label="Stream Reconstruction Error", alpha=0.7)
    plt.axhline(threshold, color="red", linestyle="--", label="Drift Threshold")
    plt.xlabel("Time / Sample Index")
    plt.ylabel("Reconstruction Error")
    plt.title("Data Drift Detection via Reconstruction Error")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # check if dataset path is given
    if len(sys.argv) < 2:
        print("Usage: python detect_drift.py <path_to_csv>")
        sys.exit(1)

    data_path = sys.argv[1]
    main(data_path)