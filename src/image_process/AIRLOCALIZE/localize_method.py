import numpy as np
from scipy.optimize import curve_fit

# self defined packages
from lib.AIRLOCALIZE.background_correction import gen_linear_interpol, compute_ROI_boundaries
from lib.AIRLOCALIZE.gaussian_intensity_calculation import *


def gaussian_mask_small(data, spotCenter, params):
    '''
    runs a 2D/3D gaussian mask algorithm to localize and quantify the intensity of a fluorescent spot
    data is the image/stack
    all units in pix with origin at the edge of pixel i.e. leftmost corner
    center corresponds to y = 0.5
    spot_ctr = [x y z] : the guess center coordinates (or [x,y] in 2D)
    params is an airlocalizeParams object holding the fit parameters
        params.psfType : 'gaussian' or 'integratedGaussian' or 'integratedGaussianStdZ' (last option only in 3D)
        params.psfSigma(1) : PSF width (in pixels)
        params.psfSigma(2) : PSF height (in pixels - ignored in 2D)
        params.fittedRegionSize : range (in PSF width units) of the region around the spot used for fitting. 
        params.maxIterations is the maximum number of iterations of the equations allowed before convergence
        params.tol is the tolerance on the convergence (in lateral pixel dimension units)
    '''
    # Convert input parameters
    xs, ys = spotCenter[:2]
    sxy = params['psfSigma'][0]
    tol = params['tol']
    maxcount = params['maxIterations']
    cutSize = params['fittedRegionSize']
    cutwidth = [cutSize * sxy]
    it = 0

    # Initialize arrays
    x0 = np.zeros(maxcount)
    y0 = np.zeros(maxcount)
    z0 = None
    N0 = np.zeros(maxcount)
    dist = np.zeros(maxcount)
    x0[0], y0[0] = xs, ys

    # Handle 3D data
    if data.ndim == 3:
        zs = spotCenter[2]
        sz = params['psfSigma'][1]
        cutwidth.append(cutSize * sz)
        z0 = np.zeros(maxcount)
        z0[0] = zs

    # Compute boundaries of the ROI
    # print(data.shape)
    _, ROIlimits = compute_ROI_boundaries(data.shape, [x0[it], y0[it]] + ([z0[it]] if data.ndim == 3 else []), cutwidth, 0, 'small')
    # Select sub-data within ROI
    if data.ndim == 3:
        # Compute ROI boundaries for 3D data
        xp_min, xp_max = int(ROIlimits[0, 0]), int(ROIlimits[1, 0])
        yp_min, yp_max = int(ROIlimits[0, 1]), int(ROIlimits[1, 1])
        zp_min, zp_max = int(ROIlimits[0, 2]), int(ROIlimits[1, 2])

        # Generating the grid within ROI and selecting sub-data
        xp, yp, zp = np.meshgrid(range(xp_min, xp_max), range(yp_min, yp_max), range(zp_min, zp_max), indexing='ij')
        sdata = data[xp_min:xp_max, yp_min:yp_max, zp_min:zp_max]
    else:
        # Compute ROI boundaries for 2D data
        xp_min, xp_max = int(ROIlimits[0, 0]), int(ROIlimits[1, 0])
        yp_min, yp_max = int(ROIlimits[0, 1]), int(ROIlimits[1, 1])
        # Generating the grid within ROI and selecting sub-data
        xp, yp = np.meshgrid(range(xp_min, xp_max), range(yp_min, yp_max), indexing='ij')
        sdata = data[xp_min:xp_max, yp_min:yp_max]

    # Loop through iterations
    it = 1
    dx, dy, dz = 1, 1, 1
    tol = tol * (dx + dy) / 2.0
    tmp = tol + 1
    while it < maxcount and tmp > tol:
        # Initializations for the iteration
        x, y = x0[it - 1], y0[it - 1]
        if data.ndim == 3: z = z0[it - 1]

        # Placeholder for the intensity calculation based on the PSF type
        # Assume intensity calculation functions are defined elsewhere
        if data.ndim == 3:
            if params['psfType'] == 'gaussian':
                intensity = intensity_gaussian3D(xp, yp, zp, x, y, z, sxy, sz, dx, dy, dz)
            elif params['psfType'] == 'integratedGaussian':
                intensity = intensity_integrated_gaussian3D(xp, yp, zp, x, y, z, sxy, sz, dx, dy, dz)
            elif params['psfType'] == 'integratedGaussianStdZ':
                intensity = intensity_integrated_gaussian3D_stdZ(xp, yp, zp, x, y, z, sxy, sz, dx, dy, dz)
        else:
            if params['psfType'] == 'gaussian':
                intensity = intensity_gaussian2D(xp, yp, x, y, sxy, dx, dy)
            elif params['psfType'] in ['integratedGaussian', 'integratedGaussianStdZ']:
                intensity = intensity_integrated_gaussian2D(xp, yp, x, y, sxy, dx, dy)
        
        intsum = np.sum(intensity * sdata)
        sumsum = np.sum(intensity**2)

        sumx = np.sum((xp - 0.5) * intensity * sdata)
        sumy = np.sum((yp - 0.5) * intensity * sdata)
        if data.ndim == 3: sumz = np.sum(zp * intensity * sdata)

        if intsum <= 0 or sumsum == 0:
            x0[it], y0[it], N0[it] = -1, -1, -1
            if data.ndim == 3: z0[it] = -1
        else:
            x0[it] = sumx / intsum
            y0[it] = sumy / intsum
            if data.ndim == 3: z0[it] = sumz / intsum

            N0[it] = intsum / sumsum

            # Location_out_of_ROI check
            is_outside = False  # Flag to indicate if the location is outside the ROI
            x0_current, y0_current = x0[it], y0[it]
            if x0_current / dx < xp_min - 1 or x0_current / dx >= xp_max: is_outside = True
            if y0_current / dy < yp_min - 1 or y0_current / dy >= yp_max: is_outside = True
            if data.ndim == 3:
                z0_current = z0[it]
                if z0_current / dz < zp_min - 1 or z0_current / dz >= zp_max: is_outside = True

            # If the location is determined to be outside the ROI, update the values to -1
            if is_outside:
                x0[it], y0[it], N0[it] = -1, -1, -1
                if data.ndim == 3: z0[it] = -1

        # Update distance and prepare for the next iteration
        if data.ndim == 3: dist[it] = np.sqrt((x - x0[it])**2 + (y - y0[it])**2 + (z - z0[it])**2)
        else: dist[it] = np.sqrt((x - x0[it])**2 + (y - y0[it])**2)

        tmp = dist[it]
        if x0[it] == -1: tmp = tol - 1  # Force exit if the location is out of ROI

        it += 1


    x0 = x0[it - 1]
    y0 = y0[it - 1]
    if data.ndim == 3: z0 = z0[it - 1]

    N0 = N0[it - 1]
    dist = dist[it - 1]
    
    # Select the relevant sub-array from the data based on the ROI boundaries for error computation
    err0 = np.sqrt(np.sum((N0 * intensity - sdata) ** 2))

    # intensity calculation
    if data.ndim == 3:
        if params['psfType'] == 'integratedGaussian':
            N0 *= 8
        elif params['psfType'] == 'integratedGaussianStdZ':
            # Define a grid around the estimated center
            x = np.arange(np.floor(x0 - 3 * sxy), np.ceil(x0 + 3 * sxy) + 1)
            y = np.arange(np.floor(y0 - 3 * sxy), np.ceil(y0 + 3 * sxy) + 1)
            z = np.arange(np.floor(z0 - 3 * sz), np.ceil(z0 + 3 * sz) + 1)
            xx, yy, zz = np.meshgrid(y, x, z, indexing='ij')
            xx = np.ceil(xx) - 0.5
            yy = np.ceil(yy) - 0.5
            zz = np.round(zz)
            # Calculate the integrated intensity
            Itot = intensity_integrated_gaussian3D_stdZ(xx, yy, zz, x0, y0, z0, sxy, sz, 1, 1, 1)
            N0 *= np.sum(Itot)
        elif params['psfType'] == 'gaussian':
            # Similar grid definition as above
            xx, yy, zz = np.meshgrid(y, x, z, indexing='ij')
            xx = np.ceil(xx) - 0.5
            yy = np.ceil(yy) - 0.5
            zz = np.round(zz)
            # Calculate the Gaussian intensity
            Itot = intensity_gaussian3D(xx, yy, zz, x0, y0, z0, sxy, sz, 1, 1, 1)
            N0 *= np.sum(Itot)
    else:
        if params['psfType'] == 'integratedGaussian':
            N0 *= 4
        elif params['psfType'] == 'integratedGaussianStdZ':
            N0 *= 4  # Assuming this is the same adjustment as for integratedGaussian in 2D
        elif params['psfType'] == 'gaussian':
            # Define a 2D grid around the estimated center
            x = np.arange(np.floor(x0 - 3 * sxy), np.ceil(x0 + 3 * sxy) + 1)
            y = np.arange(np.floor(y0 - 3 * sxy), np.ceil(y0 + 3 * sxy) + 1)
            xx, yy = np.meshgrid(y, x, indexing='ij')
            xx = np.ceil(xx) - 0.5
            yy = np.ceil(yy) - 0.5
            # Calculate the Gaussian intensity
            Itot = intensity_gaussian2D(xx, yy, x0, y0, sxy, 1, 1)
            N0 *= np.sum(Itot)

    return x0, y0, z0, N0, err0, dist, it


def gaussian_fit_local(alData, spotCenter, params, background_correction):
    """
    Performs Gaussian fitting on local image data.

    Parameters:
    - alData: Contains the image data and movie flag.
    - spotCenter: Guess center coordinates [x, y] or [x, y, z].
    - params: Fit parameters.
    - background_correction: Flag for background correction (1 or 0).

    Returns:
    - Gaussout: Array with the fitting results.
    """
    dx = dy = 1  # Assuming unity voxel dimensions for simplicity
    numDim = alData['img'].ndim
    xc, yc = spotCenter[0], spotCenter[1]
    zc = spotCenter[2] if numDim == 3 else None

    if alData['img'].ndim == 3: nx, ny, nz = alData['img'].shape
    else: nx, ny = alData['img'].shape[:2]  # Only extract the first two dimensions


    # Set default or use provided parameters
    sxy = params.get('psfSigma', [2.0])[0] / dx
    sz = params.get('psfSigma', [2.0, 2.0])[1] / dy if numDim == 3 else 2.0
    cutSize = params.get('fittedRegionSize', 3)
    tol = params.get('tol', 1e-6)
    maxcount = params.get('maxIterations', 200)
    psfType = params.get('psfType', 'integratedGaussianStdZ' if numDim == 3 else 'integratedGaussian')
    
    # Generate data and perform background correction
    _, data, new_ctr, ROIlimits, _, plfit, _ = gen_linear_interpol(alData, spotCenter, [cutSize * sxy, cutSize * sz] if numDim == 3 else cutSize * sxy, 1, 'large')

    if background_correction:
        if numDim == 2:
            xmin, ymin = ROIlimits[0, 0], ROIlimits[0, 1]
            xc2, yc2 = new_ctr[0], new_ctr[1]

            nrows, ncols = data.shape[:2]
            r, c = np.meshgrid(np.arange(1, ncols+1), np.arange(1, nrows+1), indexing='ij')
            c = np.ceil(c) - 0.5
            r = np.ceil(r) - 0.5
            pts = np.vstack([r.ravel(), c.ravel()]).T  # Transposing to get [r c] pairs
            xmax = xmin + data.shape[0] - 1
            ymax = ymin + data.shape[1] - 1

        elif numDim == 3:
            xmin, ymin, zmin = ROIlimits[0, 0], ROIlimits[0, 1], ROIlimits[0, 2]
            xc2, yc2, zc2 = new_ctr[0], new_ctr[1], new_ctr[2]

            nrows, ncols, nups = data.shape
            r, c, h = np.meshgrid(np.arange(1, ncols+1), np.arange(1, nrows+1), np.arange(1, nups+1), indexing='ij')
            c = np.ceil(c) - 0.5
            r = np.ceil(r) - 0.5
            h = np.round(h)
            pts = np.vstack([r.ravel(), c.ravel(), h.ravel()]).T  # Transposing to get [r c h] triplets
            xmax = xmin + data.shape[0] - 1
            ymax = ymin + data.shape[1] - 1
            zmax = zmin + data.shape[2] - 1
    else:
        npix = [np.ceil(cutSize * sxy), np.ceil(cutSize * sxy), np.ceil(cutSize * sz)] if numDim == 3 and not alData['isMovie'] else [np.ceil(cutSize * sxy), np.ceil(cutSize * sxy)]
        xmin = max(int(np.ceil(xc) - npix[0]), 0)
        xmax = min(int(np.ceil(xc) + npix[0]), nx - 1)  # Python indexing is 0-based
        ymin = max(int(np.ceil(yc) - npix[1]), 0)
        ymax = min(int(np.ceil(yc) + npix[1]), ny - 1)

        if numDim == 3:
            zmin = max(int(np.ceil(zc) - npix[2]), 0)
            zmax = min(int(np.ceil(zc) + npix[2]), nz - 1)
            data = alData['img'][xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
            r, c, h = np.meshgrid(range(xmin, xmax+1), range(ymin, ymax+1), range(zmin, zmax+1), indexing='ij')
            c = np.ceil(c) - 0.5
            r = np.ceil(r) - 0.5
            h = np.round(h)
            pts = np.vstack([r.ravel(), c.ravel(), h.ravel()]).T
            xc2 = xc - (xmin + 0.5)
            yc2 = yc - (ymin + 0.5)
            zc2 = zc - (zmin + 0.5)
        else:
            data = alData['img'][xmin:xmax+1, ymin:ymax+1]
            r, c = np.meshgrid(range(xmin, xmax+1), range(ymin, ymax+1), indexing='ij')
            pts = np.vstack([r.ravel(), c.ravel()]).T - 0.5
            xc2 = xc - (xmin + 0.5)
            yc2 = yc - (ymin + 0.5)

    # defining guess values of the fit parameters, their acceptable range, and other fit options
    bgmin = np.min(data)
    bg = np.median(data)
    Imax = np.max(data) - bg

    if numDim == 3:
        Coeffs = [Imax, bg, sxy, sz, xc2, yc2, zc2]
        lb = [0, bgmin, 0.1, 0.1, 0, 0, 0]
        ub = [5 * Imax, 0.5 * Imax + bg, (xmax - xmin) / 2, (zmax - zmin) / 2, xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1]
    else:
        Coeffs = [Imax, bg, sxy, xc2, yc2]
        lb = [0, bgmin, 0.1, 0, 0]
        ub = [5 * Imax, 0.5 * Imax + bg, (xmax - xmin) / 2, xmax - xmin + 1, ymax - ymin + 1]

    # Setting the appropriate fitting function
    if numDim == 3:
        if psfType == 'gaussian': hfun = gauss3D
        elif psfType == 'integratedGaussian': hfun = gauss_integrated3D
        elif psfType == 'integratedGaussianStdZ': hfun = gauss_integrated3D_stdZ
    elif numDim == 2:
        if psfType == 'gaussian': hfun = gauss2D
        elif psfType == 'integratedGaussian': hfun = gauss_integrated2D
        elif psfType == 'integratedGaussianStdZ': hfun = gauss_integrated2D


    # Actual fit
    Gaussout, cov = curve_fit(hfun, pts, data, p0=Coeffs, bounds=(lb, ub), kwargs={'maxfev': maxcount, 'xtol': tol})
    
    if numDim == 3 and not alData.isMovie:
        if psfType == 'gaussian':
            xc, yc, zc = Gaussout[4], Gaussout[5], Gaussout[6]
            sxy, sz = Gaussout[2], Gaussout[3]
            x = np.arange(np.floor(xc - 3*sxy), np.ceil(xc + 3*sxy) + 1)
            y = np.arange(np.floor(yc - 3*sxy), np.ceil(yc + 3*sxy) + 1)
            z = np.arange(np.floor(zc - 3*sz), np.ceil(zc + 3*sz) + 1)
            xx, yy, zz = np.meshgrid(y, x, z, indexing='ij')
            pts2 = np.stack([np.ceil(xx)-0.5, np.ceil(yy)-0.5, np.round(zz)], axis=-1).reshape(-1, 3)
            bg = Gaussout[1]
            Gaussout[1] = 0
            Itot = np.sum(gauss3D(Gaussout, pts2))
            Gaussout[1] = bg
        elif psfType == 'integratedGaussian':
            I0 = intensity_integrated_gaussian3D(1, 1, 1, 0.5, 0.5, 1, Gaussout[2], Gaussout[3], 1, 1, 1)
            Itot = 8*Gaussout[0]/I0
        elif psfType == 'integratedGaussianStdZ':
            xc, yc, zc = Gaussout[4], Gaussout[5], Gaussout[6]
            sxy, sz = Gaussout[2], Gaussout[3]
            x = np.arange(np.floor(xc - 3*sxy), np.ceil(xc + 3*sxy) + 1)
            y = np.arange(np.floor(yc - 3*sxy), np.ceil(yc + 3*sxy) + 1)
            z = np.arange(np.floor(zc - 3*sz), np.ceil(zc + 3*sz) + 1)
            xx, yy, zz = np.meshgrid(y, x, z, indexing='ij')
            pts2 = np.stack([np.ceil(xx)-0.5, np.ceil(yy)-0.5, np.round(zz)], axis=-1).reshape(-1, 3)
            bg = Gaussout[1]
            Gaussout[1] = 0
            Itot = np.sum(gauss_integrated3D_stdZ(Gaussout, pts2))
            Gaussout[1] = bg

        Gaussout = np.append(Gaussout, [Itot, np.sqrt(cov)])

    elif numDim == 2:
        if psfType == 'gaussian':
            xc, yc, sxy = Gaussout[3], Gaussout[4], Gaussout[2]
            x = np.arange(np.floor(xc - 3*sxy), np.ceil(xc + 3*sxy) + 1)
            y = np.arange(np.floor(yc - 3*sxy), np.ceil(yc + 3*sxy) + 1)
            xx, yy = np.meshgrid(y, x, indexing='ij')
            pts2 = np.stack([np.ceil(xx)-0.5, np.ceil(yy)-0.5], axis=-1).reshape(-1, 2)
            bg = Gaussout[1]
            Gaussout[1] = 0
            Itot = np.sum(gauss2D(Gaussout, pts2))
            Gaussout[1] = bg
        elif psfType == 'integratedGaussian':
            I0 = intensity_integrated_gaussian2D(1, 1, 0.5, 0.5, Gaussout[2], 1, 1)
            Itot = 4*Gaussout[0]/I0

        # Update Gaussout with the total integrated intensity and the square root of the residual norm.
        Gaussout = np.append(Gaussout, [Itot, np.sqrt(cov)])


    # Correcting the center position for the offset of the substack used for the fit
    if numDim == 3:
        Gaussout[4] = Gaussout[4] + xmin  # Adjust x center
        Gaussout[5] = Gaussout[5] + ymin  # Adjust y center
        Gaussout[6] = Gaussout[6] + zmin  # Adjust z center
    else:
        Gaussout[3] = Gaussout[3] + xmin  # Adjust x center for 2D
        Gaussout[4] = Gaussout[4] + ymin  # Adjust y center for 2D

    # Adding the extra parameters from the background equation to the output
    if background_correction == 1:
        if numDim == 3: Gaussout = np.append(Gaussout, plfit[:4])  # Assuming plfit has at least 4 elements
        else: Gaussout = np.append(Gaussout, plfit[:3])  # Assuming plfit has at least 3 elements

    return Gaussout


def local_maxproj(alData, spot_ctr, params):
    """
    Compute a localized maximum intensity projection around a specified spot center.

    Args:
    - alData: An object or dictionary containing the imaging data in 'img' key or attribute.
    - spot_ctr: The center of the spot as a tuple or list, [x, y, z].
    - params: A dictionary or object with attributes 'psfSigma' and 'fittedRegionSize', 
              specifying the standard deviation of the PSF and the size of the region to fit, respectively.

    Returns:
    - img: A 2D numpy array representing the localized maximum intensity projection.
    """
    zc = spot_ctr[2]
    cutwidth_z = np.ceil(params['psfSigma'][1] * params['fittedRegionSize'])
    nz = alData['img'].shape[2]
    
    # Calculate the range of planes for the max projection
    ROIlimits_z = np.ceil(zc) - np.floor(cutwidth_z)
    ROIlimits_z = max(1, ROIlimits_z)  # Ensure the lower bound is within the image
    zmax = np.ceil(zc) + np.floor(cutwidth_z)
    zmax = min(nz, zmax)  # Ensure the upper bound is within the image
    
    # Perform the max projection
    img = np.max(alData['img'][:, :, int(ROIlimits_z)-1:int(zmax)], axis=2)  # Adjusted for Python's 0-based indexing

    return img


