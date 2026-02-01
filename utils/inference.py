import os
from typing import List

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import DataLoader


def predict_and_save(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    tile_paths: List[str],
    save_dir: str = 'model/temp_pred',
    threshold: float = 0.5
) -> List[str]:
    """
    Run model predictions on tiles and save binary outputs as TIFF files.

    This function iterates through the test loader, runs inference with the model,
    applies a sigmoid activation and threshold to produce binary predictions,
    then saves each prediction as a TIFF file.

    Args:
        model: A PyTorch model for inference.
        test_loader: DataLoader containing the test dataset.
        device: The device to run inference on (e.g., 'cuda' or 'cpu').
        tile_paths: List of paths to the original tile files (used for naming outputs).
        save_dir: Directory where prediction TIFF files will be saved.
        threshold: Threshold value for binary classification (default: 0.5).

    Returns:
        List of paths to the saved prediction TIFF files.

    Example:
        >>> model = load_model("model.pth")
        >>> device = torch.device("cuda")
        >>> predictions = predict_and_save(model, test_loader, device, tile_paths)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Starting prediction on device: {device}")
    model.eval()
    model.to(device)
    predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, _, _ = batch  # Unpack the batch; labels and paths are not needed for predictions
            print(f"Predicting batch {batch_idx + 1}/{len(test_loader)}")
            print(f"Images type: {type(images)}, Images shape: {images.shape if isinstance(images, torch.Tensor) else 'unknown'}")

            images = images.to(device)  # Move images to the correct device
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.cpu().numpy()

            # Save the predictions for each image in the batch
            for i in range(images.shape[0]):
                output_image = outputs[i]
                binary_output = (output_image > threshold).astype(np.uint8)  # Apply threshold to create binary output
                original_tile_name = os.path.basename(tile_paths[batch_idx * test_loader.batch_size + i]).replace('.npy', '.tif')
                save_path = os.path.join(save_dir, original_tile_name)
                print(f"Saving prediction to {save_path}")
                tiff.imwrite(save_path, binary_output)
                predictions.append(save_path)

    return predictions
