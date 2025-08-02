import numpy as np


def compute_ROI_boundaries(stack_size, spot_ctr, cutwidth, thickness, ROI_size):
    """
    Compute the boundaries of an ROI around a spot center in an image stack.

    Parameters:
    - stack_size: Tuple of integers representing the size of the stack (nx, ny, [nz]).
    - spot_ctr: Tuple of floats representing the center of the spot (xc, yc, [zc]).
    - cutwidth: Tuple of integers specifying the cut width in pixels.
    - thickness: Integer specifying the additional thickness for 'large' ROI.
    - ROI_size: String specifying the ROI size ('small' or 'large').

    Returns:
    - new_ctr: Tuple of new center coordinates within the ROI.
    - ROIlimits: Numpy array specifying the min and max bounds of the ROI ([xmin, ymin, (zmin)], [xmax, ymax, (zmax)]).
    """

    # Check input consistency
    # print('spot_ctr:', spot_ctr)
    if len(stack_size) == 3 and len(spot_ctr) < 3:
        print('Cannot compute ROI boundaries, incorrect ROI center coordinates.')
        return None, None

    # Compute half length of ROI square/cube
    if ROI_size == 'small':
        halflength = [cutwidth[0]]
    elif ROI_size == 'large':
        halflength = [cutwidth[0] + thickness]

    if len(stack_size) == 3:
        halflength.append(cutwidth[1] if ROI_size == 'small' else cutwidth[1] + thickness)

    # Compute ROI limits
    nx, ny = stack_size[:2]
    xc, yc = spot_ctr[:2]
    xmin, xmax = np.clip([np.ceil(xc - halflength[0]), np.ceil(xc + halflength[0])], 0, nx-1)
    ymin, ymax = np.clip([np.ceil(yc - halflength[0]), np.ceil(yc + halflength[0])], 0, ny-1)
    ROIlimits = np.array([[xmin, ymin], [xmax, ymax]])

    # Adjust for 3D
    if len(stack_size) == 3:
        nz = stack_size[2]
        zc = spot_ctr[2]
        zmin, zmax = np.clip([np.round(zc - halflength[1]), np.round(zc + halflength[1])], 0, nz-1)
        ROIlimits = np.hstack((ROIlimits, [[zmin], [zmax]]))

    # Compute coordinates of spot center in new region
    new_ctr = (xc - xmin + 1, yc - ymin + 1) + ((zc - zmin + 1,) if len(stack_size) == 3 else ())
    # print('ROIlimits_in:', ROIlimits.astype(int))
    return new_ctr, ROIlimits


def generate_bg_region_3D(alData, xcenter, ycenter, zcenter, cutwidth_xy, cutwidth_z, thickness):
    """
    Generate the background region for a 3D dataset.
    
    Parameters:
    - alData: A data structure or array containing the image data.
    - xcenter, ycenter, zcenter: Center coordinates of the spot.
    - cutwidth_xy, cutwidth_z: Cutwidth parameters for defining the ROI.
    - thickness: Defines the thickness of the region around the ROI used for background calculation.
    
    Returns:
    - bg: An array of points in the format [x, y, z, intensity], where x, y, z
      are pixel positions and intensity is the value from the original image.
    """
    # Ensuring integer values for calculations
    thickness = int(abs(thickness))
    cutwidth_xy = int(abs(cutwidth_xy))
    cutwidth_z = int(abs(cutwidth_z))
    nx, ny, nz = alData.img.shape

    # Define the ROI based on provided center and cutwidth
    xc, yc, zc = np.ceil([xcenter, ycenter, zcenter]).astype(int)
    xc = np.clip(xc, 1, nx)
    yc = np.clip(yc, 1, ny)
    zc = np.clip(zc, 1, nz)
    
    # Adjusting ranges to Python's 0-based indexing by subtracting 1
    x2, y2, z2 = np.clip([xc - cutwidth_xy, yc - cutwidth_xy, zc - cutwidth_z], 0, [nx, ny, nz])
    x3, y3, z3 = np.clip([xc + cutwidth_xy, yc + cutwidth_xy, zc + cutwidth_z], 0, [nx, ny, nz])
    
    # Expand the ROI by 'thickness' pixels to define the region for background calculation
    x1, y1, z1 = np.clip([x2 - thickness, y2 - thickness, z2 - thickness], 0, [nx, ny, nz])
    x4, y4, z4 = np.clip([x3 + thickness, y3 + thickness, z3 + thickness], 0, [nx, ny, nz])
    
    # Initialize the background array
    bg = []

    # Loop over the expanded ROI and collect background pixels
    # Including logic to select points within 'thickness' of the ROI
    # Similar to the MATLAB loops for collecting bg points
    for zpix in range(z1, z4):
        for ypix in range(y1, y4):
            for xpix in range(x1, x4):
                # Check if the current pixel is within the 'thickness' boundary of the ROI
                if (xpix < x2 or xpix >= x3) or (ypix < y2 or ypix >= y3) or (zpix < z2 or zpix >= z3):
                    bg.append([xpix + 0.5, ypix + 0.5, zpix, alData.img[xpix, ypix, zpix]])

    bg = np.array(bg)  # Convert list to numpy array for easier handling
    # print(bg.shape, bg.size)
    return bg


def generate_bg_region_2D(alData, xcenter, ycenter, cutwidth_xy, thickness):
    # Adjust for Python's zero-based indexing
    thickness = int(abs(thickness))
    cutwidth_xy = int(abs(cutwidth_xy))
    
    # Determine if alData is handling movie data and adjust accordingly
    img = alData.img
    
    nx, ny = img.shape[:2]
    npts = 2 * thickness * (2 * (thickness + cutwidth_xy) + 1)
    bg = np.zeros((npts+1, 3))
    
    # Adjust for Python's zero-based indexing
    xc = int(np.ceil(xcenter) - 1)
    yc = int(np.ceil(ycenter) - 1)
    xc = max(0, min(xc, nx-1))
    yc = max(0, min(yc, ny-1))
    
    x2, y2 = max(xc - cutwidth_xy, 0), max(yc - cutwidth_xy, 0)
    x3, y3 = min(xc + cutwidth_xy, nx-1), min(yc + cutwidth_xy, ny-1)
    
    x1, y1 = max(xc - cutwidth_xy - thickness, 0), max(yc - cutwidth_xy - thickness, 0)
    x4, y4 = min(xc + cutwidth_xy + thickness, nx-1), min(yc + cutwidth_xy + thickness, ny-1)
    
    # Collect background points
    k = 0
    for xpix in range(x1, x4):
        for ypix in range(y1, y4):
            if (xpix < x2 or xpix > x3) or (ypix < y2 or ypix > y3):
                bg[k, 0] = xpix + 0.5  # Adjusting index for human-readable format (origin at 1,1)
                bg[k, 1] = ypix + 0.5
                bg[k, 2] = img[xpix, ypix]
                k += 1

    bg = bg[:k, :]  # Trim the unused part of the array
    
    # Adjust for spatial coordinates (origin at 0,0)
    bg[:, 0] -= 0.5
    bg[:, 1] -= 0.5
    
    return bg


def fit_to_3D_plane(data):
    """
    Fits the intensity I vs. (x,y) to a 3D-hyperplane with equation I = ax + by + c.
    Data should be formatted as 3 columns: x, y, intensity, where x, y refer to the
    physical coordinates (in pixel units) of the center of the pixel (origin at 0,0
    at the corner of the image).
    
    Parameters:
    - data: A numpy array of shape (n, 3) where n is the number of points.
    
    Returns:
    - x: A numpy array containing the coefficients [a, b, c] of the plane.
    """
    # Calculate sums required for the matrix equation
    sx = np.sum(data[:, 0])
    sy = np.sum(data[:, 1])
    sI = np.sum(data[:, 2])
    sxx = np.sum(data[:, 0]**2)
    syy = np.sum(data[:, 1]**2)
    sxy = np.sum(data[:, 0]*data[:, 1])
    sIx = np.sum(data[:, 2]*data[:, 0])
    sIy = np.sum(data[:, 2]*data[:, 1])
    npts = data.shape[0]

    # Construct the matrix equation Ax = v for the least squares solution
    fitmat = np.array([[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, npts]])
    v = np.array([sIx, sIy, sI])

    # Solve the matrix equation, handling singular matrix case
    try:
        x = np.linalg.solve(fitmat, v)
    except np.linalg.LinAlgError:  # If matrix inversion is impossible
        x = np.array([0, 0, np.mean(data[:, 2])])

    return x


def fit_to_4D_plane(data):
    """
    Fits the intensity I vs. (x,y,z) to a 4D hyperplane with the equation I = ax + by + cz + d.
    Data should be formatted as 4 columns: x, y, z, intensity, where x, y, z refer to the
    physical coordinates (in pixel units) of the center of the pixel (origin at 0,0,0).
    
    Parameters:
    - data: A numpy array of shape (n, 4) where n is the number of points.
    
    Returns:
    - x: A numpy array containing the coefficients [a, b, c, d] of the plane.
    """
    # Ensure data is in double precision
    data = np.array(data, dtype=float)

    # Summations required for the matrix equation Ax = v
    sx = np.sum(data[:, 0])
    sy = np.sum(data[:, 1])
    sz = np.sum(data[:, 2])
    sI = np.sum(data[:, 3])
    
    # Summations for squared terms
    sxx = np.sum(data[:, 0]**2)
    syy = np.sum(data[:, 1]**2)
    szz = np.sum(data[:, 2]**2)
    
    # Summations for product terms
    sxy = np.sum(data[:, 0] * data[:, 1])
    sxz = np.sum(data[:, 0] * data[:, 2])
    syz = np.sum(data[:, 1] * data[:, 2])
    
    # Summations for product of intensity and coordinates
    sIx = np.sum(data[:, 3] * data[:, 0])
    sIy = np.sum(data[:, 3] * data[:, 1])
    sIz = np.sum(data[:, 3] * data[:, 2])
    
    # Matrix and vector for the linear system
    fitmat = np.array([
        [sxx, sxy, sxz, sx],
        [sxy, syy, syz, sy],
        [sxz, syz, szz, sz],
        [sx, sy, sz, data.shape[0]]
    ])
    v = np.array([sIx, sIy, sIz, sI])

    # Solve the system, handling the case of a singular matrix
    try:
        x = np.linalg.solve(fitmat, v)
    except np.linalg.LinAlgError:  # If matrix inversion is impossible
        x = np.array([0, 0, 0, np.mean(data[:, 3])])

    return x


def final_3D_plane_small(alData, xc, yc, cutwidth_xy, thickness, plfit, ROIsize):
    """
    Generates a 2D image (pl) where pixel values within a specified ROI around
    (xc, yc) are set to values computed by planar interpolation of the background.
    
    Parameters:
    - alData: An object or structure with .img attribute or key containing the image data.
              It can be a 2D array or a 3D array for movie data, with an additional .curFrame attribute or key.
    - xc, yc: Center coordinates of the ROI in the original image.
    - cutwidth_xy: Defines the half-size of the ROI around the center point.
    - thickness: Specifies the thickness for ROI boundary calculation.
    - plfit: Coefficients [a, b, c] of the plane fitting the background.
    - ROIsize: 'small' or 'large', affects the size of the computed ROI.

    Returns:
    - pl: Interpolated plane values within the ROI.
    - stack_bg_corr: The original image corrected by subtracting the interpolated plane.
    - new_ctr: New center of the ROI in the corrected image (adjusted for possible image boundary effects).
    - ROIlimits: The boundaries of the computed ROI.
    """
    if hasattr(alData, 'isMovie') and alData.isMovie:
        img = alData.img[:, :, alData.curFrame]
    else:
        img = alData.img
    
    new_ctr, ROIlimits = compute_ROI_boundaries(img.shape[:2], [xc, yc], cutwidth_xy, thickness, ROIsize)
    
    xmin, xmax = ROIlimits[0, 0], ROIlimits[1, 0]
    ymin, ymax = ROIlimits[0, 1], ROIlimits[1, 1]
    
    pl = np.zeros((xmax-xmin+1, ymax-ymin+1))
    stack_bg_corr = np.zeros_like(pl)
    
    for xpix in range(xmin, xmax+1):
        for ypix in range(ymin, ymax+1):
            interpolated_value = plfit[0]*(xpix-0.5) + plfit[1]*(ypix-0.5) + plfit[2]
            pl[xpix-xmin, ypix-ymin] = interpolated_value
            stack_bg_corr[xpix-xmin, ypix-ymin] = img[xpix, ypix] - interpolated_value
    
    return pl, stack_bg_corr, new_ctr, ROIlimits


def final_4D_plane_small(alData, xc, yc, zc, cutwidth_xy, cutwidth_z, thickness, plfit, ROIsize):
    """
    Generates an image pl within an ROI defined around a central spot (xc, yc, zc)
    in a 3D image. Pixels within the ROI are set to the values computed by a planar
    interpolation of the background surrounding the ROI.
    
    Parameters:
    - alData: Object or dictionary with .img attribute as a 3D numpy array representing the original image data.
    - xc, yc, zc: Coordinates of the central spot.
    - cutwidth_xy, cutwidth_z: Cutwidth parameters defining the size of the ROI.
    - thickness: Not directly used in this function but passed to compute_ROI_boundaries.
    - plfit: Coefficients [a, b, c, d] of the plane fitting the background.
    - ROIsize: Option ('small' or 'large') that affects ROI boundaries computation.
    
    Returns:
    - pl: Interpolated plane values within the ROI.
    - stack_bg_corr: The original image corrected by subtracting the interpolated plane.
    - new_ctr: New center of the ROI in the corrected image.
    - ROIlimits: Boundaries of the ROI.
    """
    new_ctr, ROIlimits = compute_ROI_boundaries(alData.img.shape, [xc, yc, zc], [cutwidth_xy, cutwidth_z], thickness, ROIsize)

    xmin, xmax = int(ROIlimits[0, 0]), int(ROIlimits[1, 0])
    ymin, ymax = int(ROIlimits[0, 1]), int(ROIlimits[1, 1])
    zmin, zmax = int(ROIlimits[0, 2]), int(ROIlimits[1, 2])

    # Initialize the output arrays
    # print(xmax, xmin, ymax, ymin, zmax, zmin)
    pl = np.zeros((xmax-xmin, ymax-ymin, zmax-zmin))
    stack_bg_corr = np.zeros_like(pl)
    

    xgrid, ygrid, zgrid = np.meshgrid(range(xmin, xmax), range(ymin, ymax), range(zmin, zmax), indexing='ij')

    # Calculate interpolated values using vectorized operations
    interpolated_values = plfit[0] * (xgrid+0.5) + plfit[1] * (ygrid+0.5) + plfit[2] * zgrid + plfit[3]
    # Update 'pl' and 'stack_bg_corr' arrays
    pl[:] = interpolated_values
    # print(xmin, xmax)
    # print(alData.img[xmin:xmax, ymin:ymax, zmin:zmax].shape, interpolated_values.shape, stack_bg_corr.shape)
    stack_bg_corr[:] = alData.img[xmin:xmax, ymin:ymax, zmin:zmax] - interpolated_values

    # Compute the interpolated plane and corrected image
    # for xpix in range(xmin, xmax):
    #     for ypix in range(ymin, ymax):
    #         for zpix in range(zmin, zmax):
    #             interpolated_value = (plfit[0] * (xpix-0.5) + plfit[1] * (ypix-0.5) + plfit[2] * zpix + plfit[3])
    #             pl[xpix-xmin, ypix-ymin, zpix-zmin] = interpolated_value
    #             stack_bg_corr[xpix-xmin, ypix-ymin, zpix-zmin] = (alData.img[xpix, ypix, zpix] - interpolated_value)

    return pl, stack_bg_corr, new_ctr, ROIlimits


def gen_linear_interpol(alData, spotCenter, cutwidth, thickness, ROIsize='small'):
    numDim = 3 if len(alData.img.shape) == 3 else 2
    cutwidth_xy = cutwidth[0]
    xc, yc = spotCenter[:2]

    if numDim == 3:
        if len(cutwidth) < 2:
            print("3D stack selected but cutwidth parameter has only 1 element; ensure that fit entry is compatible with 3D data.")
            return [np.array([])] * 7  # Return empty arrays for all outputs
        cutwidth_xy, cutwidth_z = cutwidth[:2]
        
        if len(spotCenter) < 3:
            print("3D stack selected but spot center has only 2 coordinates; ensure that numdim is set to 3.")
            return [np.array([])] * 7  # Return empty arrays for all outputs
        
        zc = spotCenter[2]
        bg = generate_bg_region_3D(alData, spotCenter[0], spotCenter[1], zc, cutwidth_xy, cutwidth_z, thickness=thickness)
    else:
        # Assuming cutwidth and spotCenter are correctly formatted for 2D
        bg = generate_bg_region_2D(alData, spotCenter[0], spotCenter[1], cutwidth[0], thickness)

    if bg.size == 0:
        if numDim == 3:
            plfit = np.zeros(4)  # For 3D data, plfit has 4 elements
            new_ctr = np.array([xc, yc, zc])
            ROIlimits = np.array([1, 1, 1])
        else:
            plfit = np.zeros(3)  # For 2D data, plfit has 3 elements
            new_ctr = np.array([xc, yc])
            ROIlimits = np.array([1, 1])
        
        pl = np.array([])  # Initialize as an empty array
        stack_bg_corr = np.array([])  # Initialize as an empty array
        plmean = 0
        return pl, stack_bg_corr, new_ctr, ROIlimits, plmean, plfit, bg
    

    # Fit to a plane based on dimensionality
    if numDim == 3: plfit = fit_to_4D_plane(bg)  # bg should be prepared beforehand
    else: plfit = fit_to_3D_plane(bg)  # Implement these functions in Python

    # Generate corrected images and ROI based on the fitting
    if numDim == 3: pl, stack_bg_corr, new_ctr, ROIlimits = final_4D_plane_small(alData, spotCenter[0], spotCenter[1], spotCenter[2], cutwidth[0], cutwidth[1], thickness, plfit, ROIsize)
    else: pl, stack_bg_corr, new_ctr, ROIlimits = final_3D_plane_small(alData, spotCenter[0], spotCenter[1], cutwidth[0], thickness, plfit, ROIsize)

    # Calculate mean intensity in the ROI
    plmean = np.mean(pl)
    return pl, stack_bg_corr, new_ctr, ROIlimits, plmean, plfit, bg


def subtract_median(alData, spotCenter, cutWidth, thickness, ROIsize, median_type):
    """
    Subtracts the median (local or global) from a specified ROI in an image or movie frame.
    
    Parameters:
    - alData: An object with .img attribute containing the image data and optionally .isMovie and .curFrame for movie data.
    - spotCenter: The center of the ROI (x, y, [z]).
    - cutWidth: Defines the size of the ROI around the center.
    - thickness: Used in calculating ROI boundaries.
    - ROIsize: 'small' or 'large', affects the size of the computed ROI.
    - median_type: 'local' or 'global', determines the median calculation method.
    
    Returns:
    - pl: An empty array (placeholder for compatibility with other functions).
    - stack_bg_corr: The ROI from the image with the median subtracted.
    - new_ctr: New center of the ROI (adjusted for possible image boundary effects).
    - ROIlimits: The boundaries of the computed ROI.
    """
    img = alData.img

    new_ctr, ROIlimits = compute_ROI_boundaries(img.shape, spotCenter, cutWidth, thickness, ROIsize)

    # Extract the ROI from the image
    stack_bg_corr = img[
        ROIlimits[0, 0]:ROIlimits[1, 0] + 1,
        ROIlimits[0, 1]:ROIlimits[1, 1] + 1,
    ]

    # Subtract the median
    if median_type == 'local':
        median_value = np.median(stack_bg_corr)
    elif median_type == 'global':
        median_value = np.median(img)
    
    stack_bg_corr = stack_bg_corr - median_value

    pl = np.array([])  # Placeholder, not used in this function
    return pl, stack_bg_corr, new_ctr, ROIlimits
