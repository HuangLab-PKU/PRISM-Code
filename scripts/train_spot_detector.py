import os
import numpy as np
from sklearn.model_selection import train_test_split
from stardist.models import Config2D, StarDist2D
from csbdeep.utils import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.spot_detection.dataset import load_data, prepare_data
import argparse


def main(args):
    # Configuration
    Path(args.model_dir).mkdir(exist_ok=True)

    # Load and prepare data
    images, masks = load_data(args.data_dir)
    X, Y = prepare_data(images, masks)

    # Check if there is enough data for training and validation
    if len(X) < 2:
        print("\n" + "="*80)
        print("STOP: Not enough training data.")
        print("StarDist requires at least two images to create a training and a validation set.")
        print(f"Please provide at least 2 image/mask pairs in '{os.path.join(args.data_dir, 'training')}'.")
        print(f"Found: {len(X)} image/mask pair(s).")
        print("="*80 + "\n")
        return

    if not X:
        print("Error: No images found. Please check the data directory.")
        return
    # The number of channels is the last dimension after transposing
    n_channels = X[0].shape[-1]
    print(f"Number of channels: {n_channels}")

    # Configure the StarDist model
    grid = (2, 2)
    n_rays = 16
    
    conf = Config2D(
        n_rays=n_rays,
        grid=grid,
        use_gpu=args.use_gpu,
        n_channel_in=n_channels,
        train_patch_size=(args.patch_size, args.patch_size),
        train_epochs=args.epochs,
        train_batch_size=args.batch_size
    )
    print(conf)
    
    # Create a StarDist model
    model = StarDist2D(conf, name=args.model_name, basedir=args.model_dir)

    # Split data and train
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)
    print(f"Number of training images: {len(X_train)}")
    print(f"Number of validation images: {len(X_val)}")
    model.train(X_train, Y_train, validation_data=(X_val, Y_val), epochs=conf.train_epochs)
    model.optimize_thresholds(X_val, Y_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a StarDist model for spot detection.")
    parser.add_argument('--data-dir', type=str, default='data', help="Directory containing the training data.")
    parser.add_argument('--model-dir', type=str, default='models', help="Directory to save the trained model.")
    parser.add_argument('--model-name', type=str, default='stardist_spot_detector', help="Name of the model.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--patch-size', type=int, default=256, help="Patch size for training.")
    parser.add_argument('--use-gpu', action='store_true', help="Use GPU for training if available.")

    args = parser.parse_args()
    main(args) 