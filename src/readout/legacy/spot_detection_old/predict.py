import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from scipy.ndimage import center_of_mass

def predict_spots(image, model, prob_thresh=0.5, nms_thresh=0.3):
    """
    Predicts spot locations in an image using a trained StarDist model.

    Args:
        image (np.ndarray): The input image.
        model (StarDist2D): The trained StarDist model.
        prob_thresh (float): Probability threshold for object detection.
        nms_thresh (float): Non-maximum suppression threshold.

    Returns:
        np.ndarray: A label image containing the predicted spots.
    """
    img_normalized = normalize(image, 1, 99.8, axis=(0, 1))
    labels, _ = model.predict_instances_big(
        img_normalized,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh
    )
    return labels

def get_spot_centroids(labels):
    """
    Calculates the centroid coordinates of each labeled spot.

    Args:
        labels (np.ndarray): A label image where each spot has a unique integer ID.

    Returns:
        np.ndarray: An array of (y, x) coordinates for each spot centroid.
    """
    if labels.max() == 0:
        return np.empty((0, 2), dtype=int)
        
    centroids = center_of_mass(labels, labels, index=np.arange(1, labels.max() + 1))
    return np.array(centroids, dtype=int) 