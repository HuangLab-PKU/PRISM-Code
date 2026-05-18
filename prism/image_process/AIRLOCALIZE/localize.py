import numpy as np
from tqdm import tqdm

# self defined packages
from lib.AIRLOCALIZE.background_correction import gen_linear_interpol, subtract_median
from lib.AIRLOCALIZE.localize_method import gaussian_mask_small, gaussian_fit_local, local_maxproj


def run_gaussian_fit_on_all_spots_in_image(spotCandidates, alData, params, verbose=True):
    """
    Translated function to run Gaussian fit on all spots in an image.

    Args:
        spotCandidates (np.ndarray): Array of spot candidates.
        alData (dict): Dictionary containing image data and metadata.
        params (dict): Dictionary of parameters for fitting.

    Returns:
        tuple: A tuple containing the location of spots and their variables.
    """
    # Initialize arrays and set cutwidth
    nSpots = len(spotCandidates)
    loc = None
    locVars = None
    cutWidth = None

    if params['fitMethod'] == '3DGaussianFit':
        loc = np.zeros((nSpots, 8))
        locVars = ['x_in_pix', 'y_in_pix', 'z_in_pix', 'integratedIntensity', 'residuals', 'image_number']
        cutWidth = [np.ceil(params['fittedRegionSize'] * psf) for psf in params['psfSigma']]

    elif params['fitMethod'] in ['3DMaskFull', '2DMaskOnLocalMaxProj']:
        loc = np.zeros((nSpots, 6))
        locVars = ['x_in_pix', 'y_in_pix', 'z_in_pix', 'integratedIntensity', 'residuals', 'image_number']
        cutWidth = [np.ceil(params['fittedRegionSize'] * psf) for psf in params['psfSigma']]
        
    elif params['fitMethod'] in ['2DGaussianMask', '2DGaussianFit']:
        loc = np.zeros((nSpots, 4))
        locVars = ['x_in_pix', 'y_in_pix', 'integratedIntensity', 'residuals', 'image_number']
        cutWidth = np.ceil(params['fittedRegionSize'] * params['psfSigma'][0])
    
    else: raise ValueError("Unrecognized fit method")


    # Loop over each pre-detected spot
    with tqdm(total=nSpots, desc=f"Fit predetected spots in {alData.curFile}", position=0, leave=True, disable=not verbose) as pbar_sub:
        for j in range(nSpots):
            new_ctr = spotCandidates[j, :params['numdim']]
            ROIlimits = None

            # Background correction
            if params['bgCorrectionMode'] == 'localPlane':
                _, stack_bg_corr, new_ctr, ROIlimits, *_ = gen_linear_interpol(alData, new_ctr, cutWidth, 1, 'large')
            elif params['bgCorrectionMode'] == 'localMedian':
                stack_bg_corr, new_ctr, ROIlimits = subtract_median(alData, new_ctr, cutWidth, 1, mode='local')
            elif params['bgCorrectionMode'] == 'globalMedian':
                stack_bg_corr, new_ctr, ROIlimits = subtract_median(alData, new_ctr, cutWidth, 1, mode='global')
            
            # Localize spots
            if params['fitMethod'] == '3DMaskFull':
                # print(stack_bg_corr.shape)
                x0, y0, z0, N0, err0, *_ = gaussian_mask_small(stack_bg_corr, new_ctr, params)
                loc[j, 0:5] = [x0 + ROIlimits[0, 0] + 1, y0 + ROIlimits[0, 1] + 1, z0 + ROIlimits[0, 2], N0, err0]
            
            elif params['fitMethod'] == '2DGaussianMask':
                x0, y0, _, N0, err0, *_ = gaussian_mask_small(stack_bg_corr, new_ctr, params)
                loc[j, 0:4] = [x0 + ROIlimits[0, 0], y0 + ROIlimits[0, 1], N0, err0]

            elif params['fitMethod'] == '3DGaussianFit':
                Gaussout = gaussian_fit_local(stack_bg_corr, new_ctr, params, 1)
                loc[j, 0:5] = [Gaussout[4] + ROIlimits[0, 0], Gaussout[5] + ROIlimits[0, 1], Gaussout[6] + ROIlimits[0, 2], Gaussout[7], Gaussout[8]]

            elif params['fitMethod'] == '2DGaussianFit':
                Gaussout = gaussian_fit_local(stack_bg_corr, new_ctr, params, 1)
                loc[j, 0:4] = [Gaussout[3] + ROIlimits[0, 0], Gaussout[4] + ROIlimits[0, 1], Gaussout[5], Gaussout[6]]

            elif params['fitMethod'] == '2DMaskOnLocalMaxProj':
                img_bg_corr = local_maxproj(stack_bg_corr, new_ctr, params)
                x0, y0, _, N0, err0, *_ = gaussian_mask_small(img_bg_corr, new_ctr[:2], params)
                loc[j, 0:5] = [x0 + ROIlimits[0, 0], y0 + ROIlimits[0, 1], new_ctr[2] + ROIlimits[0, 2], N0, err0]

            else:
                print(f"Unrecognized fit method: {params['fitMethod']}")

            pbar_sub.update(1)


    return loc, locVars