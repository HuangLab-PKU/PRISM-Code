import numpy as np
from scipy import ndimage


def find_isolated_maxima(alData, params, verbose=False):
    """
    Finds local maxima above the threshold in the smoothed image.

    alData: dict containing 'smooth', 'img', 'isMovie', and 'curFrame' keys.
    params: dict containing 'threshLevel', 'threshUnits', 'minDistBetweenSpots', 'numDim', and 'maxSpots'.
    verbose: boolean, prints additional information if True.

    Example usage:
        alData = {
            'smooth': your_smoothed_image_here,
            'img': your_original_image_here,
            'isMovie': False,  # or True if it's a movie
            'curFrame': 0  # current frame to process if it's a movie
        }
        params = {
            'threshLevel': your_threshold_level,
            'threshUnits': 'absolute',  # or 'SD' or 'legacySD'
            'minDistBetweenSpots': your_minimum_distance_between_spots,
            'numDim': 2,  # or 3 for 3D images
            'maxSpots': your_maximum_number_of_spots_allowed
        }
        maxima = find_isolated_maxima_clean3(alData, params, verbose=True)
    """
    
    # Retrieve threshold level
    if params['threshUnits'] == 'absolute':
        threshInt = params['threshLevel']
        
    elif params['threshUnits'] in ['SD', 'legacySD']:
        channel_name = alData.curFile.replace('.tif', '')
        threshLevel = params['threshLevel'][channel_name]
        image_to_use = alData.img if params['threshUnits'] == 'SD' else alData.smooth
        threshInt = np.mean(image_to_use[image_to_use > 0]) + threshLevel * np.std(image_to_use[image_to_use > 0])

    else: raise ValueError("Unsupported threshUnits")
    if verbose: print(f"Threshold value is {threshInt} in absolute units")

    # Find local maxima
    minDist = round(params['minDistBetweenSpots'])
    if minDist > 0:
        struct_size = [2 * minDist + 1] * params['numdim'] # create the structural element with the desired size
        struct = np.ones(struct_size, dtype=bool)
        center = tuple([minDist] * params['numdim'])
        struct[center] = False  # Setting the center to False
        image_to_process = alData.smooth
        local_max = image_to_process >= ndimage.grey_dilation(image_to_process, footprint=struct)       
    else:
        image_to_process = alData.smooth
        local_max = np.ones_like(image_to_process, dtype=bool)

    # Enforce that local maxima intensity is above threshold
    local_max &= image_to_process > threshInt

    # Store maxima as a list of coordinates / intensity
    indices = np.argwhere(local_max)
    intensities = image_to_process[local_max]
    maxima = np.column_stack((indices, intensities))
    
    # Ordering the maxima by descending intensity value
    maxima = maxima[maxima[:, -1].argsort()[::-1]]

    # Truncating the array if it has more spots than allowed
    if maxima.shape[0] > params['maxSpots']:
        maxima = maxima[:params['maxSpots'], :]
        if verbose: print(f"Predetected {maxima.shape[0]} spots; Truncated to {params['maxSpots']} spots;")
    elif verbose: print(f"Predetected {maxima.shape[0]} spots;")

    return maxima