import numpy as np
from scipy.optimize import curve_fit


def gaussian_2d(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    A 2D Gaussian function.
    """
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def fit_gaussian_2d(image, center_y, center_x, roi_size=15):
    """
    Fits a 2D Gaussian to a region of interest (ROI) in the image.

    Args:
        image (np.ndarray): The input image.
        center_y (int): The y-coordinate of the ROI center.
        center_x (int): The x-coordinate of the ROI center.
        roi_size (int): The size of the ROI (must be an odd number).

    Returns:
        tuple: A tuple containing the optimized parameters of the Gaussian fit
               and the covariance matrix. Returns (None, None) if the fit fails.
    """
    if roi_size % 2 == 0:
        raise ValueError("roi_size must be an odd number.")

    half_size = roi_size // 2
    y_start, y_end = center_y - half_size, center_y + half_size + 1
    x_start, x_end = center_x - half_size, center_x + half_size + 1

    # Ensure ROI is within image bounds
    y_start, y_end = max(0, y_start), min(image.shape[0], y_end)
    x_start, x_end = max(0, x_start), min(image.shape[1], x_end)

    roi = image[y_start:y_end, x_start:x_end]

    if roi.size == 0:
        return None, None

    y, x = np.indices(roi.shape)
    
    initial_guess = (
        roi.max() - roi.min(),  # amplitude
        half_size,              # xo
        half_size,              # yo
        2.0,                    # sigma_x
        2.0,                    # sigma_y
        0,                      # theta
        roi.min()               # offset
    )

    try:
        popt, pcov = curve_fit(
            gaussian_2d,
            (x, y),
            roi.ravel(),
            p0=initial_guess,
            bounds=(
                (0, 0, 0, 0.5, 0.5, -np.inf, 0),
                (np.inf, roi.shape[1], roi.shape[0], roi_size, roi_size, np.inf, np.inf)
            )
        )
        return popt, pcov
    except (RuntimeError, ValueError):
        return None, None

def get_intensity_and_background(popt):
    """
    Calculates the integrated intensity (volume) of a 2D Gaussian 
    and the background value from its parameters.

    The volume under a 2D Gaussian is 2 * pi * A * sigma_x * sigma_y.
    This represents the total signal above the fitted background.
    """
    if popt is None:
        return 0.0, 0.0  # intensity, background
    amplitude, _, _, sigma_x, sigma_y, _, offset = popt
    intensity = 2 * np.pi * amplitude * sigma_x * sigma_y
    return intensity, offset 