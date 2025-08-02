import os
import sys
import pandas as pd
import numpy as np
from tifffile import imread
from stardist.models import StarDist2D
from tqdm import tqdm
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spot_detection.predict import predict_spots, get_spot_centroids
from spot_detection.gaussian_fitting import fit_gaussian_2d, get_intensity_and_background

def main(args):
    # Load the trained model
    model = StarDist2D(None, name=args.model_name, basedir=args.model_dir)

    # Load images and stack them
    channel_files = {channel: os.path.join(args.input_dir, f) for channel, f in zip(args.channels, args.channel_files)}
    
    # Read images and ensure they are 2D
    images_2d = {channel: imread(path) for channel, path in channel_files.items()}
    
    # Stack images directly into (H, W, C) format
    image_stack = np.stack([images_2d[ch] for ch in args.channels], axis=-1)

    print(f"Predicting spots from {image_stack.shape[-1]}-channel image...")
    labels = predict_spots(image_stack, model, prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh)
    centroids = get_spot_centroids(labels)
    print(f"Found {len(centroids)} unique spots.")

    # Measure intensities for each spot in all channels
    results = []
    for y, x in tqdm(centroids, desc="Measuring spot intensities"):
        spot_data = {'Y': y, 'X': x}
        for channel, image in images_2d.items():
            popt, _ = fit_gaussian_2d(image, y, x, roi_size=args.roi_size)
            intensity, background = get_intensity_and_background(popt)
            spot_data[f'{channel}_intensity'] = intensity
            spot_data[f'{channel}_background'] = background
        results.append(spot_data)

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-channel spot readout using StarDist.")
    parser.add_argument('--input-dir', type=str, required=True, help="Directory containing the input TIFF images.")
    parser.add_argument('--output-csv', type=str, required=True, help="Path to save the output CSV file.")
    parser.add_argument('--model-dir', type=str, default='models', help="Directory where the trained model is saved.")
    parser.add_argument('--model-name', type=str, default='stardist_spot_detector', help="Name of the trained model.")
    parser.add_argument('--channels', nargs='+', default=['cy5', 'TxRed', 'cy3', 'FAM'], help="List of channel names.")
    parser.add_argument('--channel-files', nargs='+', required=True, help="List of TIFF filenames corresponding to the channels.")
    parser.add_argument('--prob-thresh', type=float, default=0.5, help="Probability threshold for StarDist.")
    parser.add_argument('--nms-thresh', type=float, default=0.3, help="NMS threshold for StarDist.")
    parser.add_argument('--roi-size', type=int, default=15, help="Size of the ROI for Gaussian fitting.")
    
    args = parser.parse_args()
    main(args) 