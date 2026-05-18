import numpy as np
from scipy.special import erf


def gauss2D(Coeffs, pts):
    """
    Calculate the 2D Gaussian intensity.

    Args:
    - Coeffs: Coefficients for the Gaussian calculation. An array where:
        - Coeffs[0] is Imax, 
        - Coeffs[1] is background intensity (bg), 
        - Coeffs[2] is standard deviation in x and y (sxy),
        - Coeffs[3] is center in x (xc),
        - Coeffs[4] is center in y (yc).
    - pts: Points where the Gaussian intensity should be calculated. A 2D array 
           where the first dimension is for different points, and the second dimension 
           represents [r, c], corresponding to rows and columns.

    Returns:
    - I: The calculated Gaussian intensity for each point.
    """
    gridSize = pts.shape[1]
    r = pts[:, :gridSize//2]
    c = pts[:, gridSize//2:]

    # Assuming intensity_gaussian2D is defined elsewhere and works similarly to the MATLAB version.
    I = Coeffs[1] + Coeffs[0] * intensity_gaussian2D(r, c, Coeffs[3], Coeffs[4], Coeffs[2], 1, 1)

    return I


def gauss_integrated2D(Coeffs, pts):
    """
    Calculate the integrated 2D Gaussian intensity.

    Args:
    - Coeffs: Coefficients for the Gaussian calculation. An array where:
        - Coeffs[0] is Imax,
        - Coeffs[1] is background intensity (bg),
        - Coeffs[2] is standard deviation in x and y (sxy),
        - Coeffs[3] is center in x (xc),
        - Coeffs[4] is center in y (yc).
    - pts: Points where the Gaussian intensity should be calculated. A 2D array
           where the first dimension is for different points, and the second dimension
           represents [r, c], corresponding to rows and columns.

    Returns:
    - I: The calculated integrated Gaussian intensity for each point.
    """
    gridSize = pts.shape[1]
    r = pts[:, :gridSize//2]
    c = pts[:, gridSize//2:]

    # Compute max intensity so prefactor (Coeffs[0]) is actually Imax.
    I0 = intensity_integrated_gaussian2D(1, 1, 0.5, 0.5, Coeffs[2], 1, 1)

    # Compute intensity over all data points.
    I = Coeffs[1] + Coeffs[0] * intensity_integrated_gaussian2D(
        r, c, Coeffs[3], Coeffs[4], Coeffs[2], 1, 1) / I0

    return I


def gauss3D(Coeffs, pts):
    """
    Calculate the 3D Gaussian intensity.

    Args:
    - Coeffs: Coefficients for the Gaussian calculation. An array where:
        - Coeffs[0] is Imax, 
        - Coeffs[1] is background intensity (bg), 
        - Coeffs[2] is standard deviation in x and y (sxy),
        - Coeffs[3] is standard deviation in z (sz),
        - Coeffs[4] is center in x (xc),
        - Coeffs[5] is center in y (yc),
        - Coeffs[6] is center in z (zc).
    - pts: Points where the Gaussian intensity should be calculated. A 3D array 
           where the first dimension is for different points, and the second dimension 
           represents [r, c, h], corresponding to rows, columns, and height.

    Returns:
    - I: The calculated Gaussian intensity for each point.
    """
    gridSize = pts.shape[1]
    r = pts[:, :gridSize//3]
    c = pts[:, gridSize//3:2*gridSize//3]
    h = pts[:, 2*gridSize//3:]

    # Assuming intensity_gaussian3D is defined elsewhere and works similarly to the MATLAB version.
    I = Coeffs[1] + Coeffs[0] * intensity_gaussian3D(r, c, h, Coeffs[4], Coeffs[5], Coeffs[6], Coeffs[2], Coeffs[3], 1, 1, 1)

    return I


def gauss_integrated3D(Coeffs, pts):
    """
    Calculate the integrated 3D Gaussian intensity.

    Args:
    - Coeffs: Coefficients for the Gaussian calculation. An array where:
        - Coeffs[0] is Imax, 
        - Coeffs[1] is background intensity (bg), 
        - Coeffs[2] is standard deviation in x and y (sxy),
        - Coeffs[3] is standard deviation in z (sz),
        - Coeffs[4] is center in x (xc),
        - Coeffs[5] is center in y (yc),
        - Coeffs[6] is center in z (zc).
    - pts: Points where the Gaussian intensity should be calculated. A 3D array 
           where the first dimension is for different points, and the second dimension 
           represents [r, c, h], corresponding to rows, columns, and planes.

    Returns:
    - I: The calculated integrated Gaussian intensity for each point.
    """
    gridSize = pts.shape[1]
    r = pts[:, :gridSize//3]
    c = pts[:, gridSize//3:2*gridSize//3]
    h = pts[:, 2*gridSize//3:]

    # Compute max intensity so prefactor (Coeffs[1]) is actually Imax.
    # Assuming intensity_integrated_gaussian3D is defined elsewhere and correctly adapted from MATLAB.
    I0 = intensity_integrated_gaussian3D(1, 1, 1, 0.5, 0.5, 1, Coeffs[2], Coeffs[3], 1, 1, 1)

    # Compute intensity over all data points.
    I = Coeffs[1] + Coeffs[0] * intensity_integrated_gaussian3D(
        r, c, h, Coeffs[4], Coeffs[5], Coeffs[6], Coeffs[2], Coeffs[3], 1, 1, 1) / I0

    return I


def gauss_integrated3D_stdZ(Coeffs, pts):
    """
    Calculate the integrated 3D Gaussian intensity with standardized z-dimension.

    Args:
    - Coeffs: Coefficients for the Gaussian calculation. An array where:
        - Coeffs[0] is Imax,
        - Coeffs[1] is background intensity (bg),
        - Coeffs[2] is standard deviation in x and y (sxy),
        - Coeffs[3] is standard deviation in z (sz),
        - Coeffs[4] is center in x (xc),
        - Coeffs[5] is center in y (yc),
        - Coeffs[6] is center in z (zc).
    - pts: Points where the Gaussian intensity should be calculated. A 3D array
           where the first dimension is for different points, and the second dimension
           represents [r, c, h], corresponding to rows, columns, and planes.

    Returns:
    - I: The calculated integrated Gaussian intensity for each point.
    """
    gridSize = pts.shape[1]
    r = pts[:, :gridSize//3]
    c = pts[:, gridSize//3:2*gridSize//3]
    h = pts[:, 2*gridSize//3:]

    # Compute max intensity so prefactor (Coeffs[0]) is actually Imax.
    I0 = intensity_integrated_gaussian3D_stdZ(
        1, 1, 1, 0.5, 0.5, 1, Coeffs[2], Coeffs[3], 1, 1, 1)

    # Compute intensity over all data points.
    I = Coeffs[1] + Coeffs[0] * intensity_integrated_gaussian3D_stdZ(
        r, c, h, Coeffs[4], Coeffs[5], Coeffs[6], Coeffs[2], Coeffs[3], 1, 1, 1) / I0

    return I


def intensity_gaussian3D(xp, yp, zp, x, y, z, sxy, sz, dx, dy, dz):
    """
    Computes the intensity of pixels assuming a standard 3D Gaussian PSF.

    Parameters:
    - xp, yp, zp: arrays of pixel indices.
    - x, y, z: position of the Gaussian center in real units (use dx=dy=dz=1 for pixel units).
    - sxy: sigma in the xy-plane in pixels.
    - sz: sigma in the z-direction in pixels.
    - dx, dy, dz: voxel dimensions in real units.

    Returns:
    - intensity: a 3D array of the Gaussian intensity values.
    """
    # Computing the center position of each voxel
    xp_center = np.ceil(xp) - 0.5
    yp_center = np.ceil(yp) - 0.5
    zp_center = np.round(zp)
    
    # Gaussian intensity calculations
    gx = np.exp(-((xp_center * dx - x) ** 2) / (2 * sxy ** 2))
    gy = np.exp(-((yp_center * dy - y) ** 2) / (2 * sxy ** 2))
    gz = np.exp(-((zp_center * dz - z) ** 2) / (2 * sz ** 2))
    
    intensity = gx * gy * gz
    return intensity


def intensity_integrated_gaussian3D(xp, yp, zp, x, y, z, sxy, sz, dx, dy, dz):
    """
    Computes the intensity of pixels assuming a 3D Gaussian PSF that is
    integrated over each voxel (normalized intensity, i.e., no prefactor).

    Parameters:
    - xp, yp, zp: arrays of pixel indices.
    - x, y, z: position of the Gaussian in real units (if pixels, enter dx=dy=dz=1).
    - sxy: sigma in the xy-plane in pixels.
    - sz: sigma in the z-direction in pixels.
    - dx, dy, dz: voxel dimensions in real units.
    
    Returns:
    - intensity: a 3D array of the Gaussian intensity values integrated over each voxel.
    """
    # Compute the center position of each voxel
    xp_center = np.ceil(xp) - 0.5
    yp_center = np.ceil(yp) - 0.5
    zp_center = np.round(zp)
    
    # Compute the differences for integration limits
    diffx1 = (xp_center - 0.5) * dx - x
    diffx2 = (xp_center + 0.5) * dx - x
    diffy1 = (yp_center - 0.5) * dy - y
    diffy2 = (yp_center + 0.5) * dy - y
    diffz1 = (zp_center - 0.5) * dz - z
    diffz2 = (zp_center + 0.5) * dz - z
    
    # Normalize differences by the standard deviations
    diffx1 /= np.sqrt(2) * sxy
    diffx2 /= np.sqrt(2) * sxy
    diffy1 /= np.sqrt(2) * sxy
    diffy2 /= np.sqrt(2) * sxy
    diffz1 /= np.sqrt(2) * sz
    diffz2 /= np.sqrt(2) * sz
    
    # Calculate the integrated intensity
    intensity = np.abs(erf(diffx1) - erf(diffx2)) * \
                np.abs(erf(diffy1) - erf(diffy2)) * \
                np.abs(erf(diffz1) - erf(diffz2))
    
    return intensity


def intensity_integrated_gaussian3D_stdZ(xp, yp, zp, x, y, z, sxy, sz, dx, dy, dz):
    """
    Computes intensity of pixels assuming a 3D Gaussian PSF that is
    integrated over each pixel in the xy-plane; along z, the envelope is that of a standard
    Gaussian (no integration).
    
    Parameters:
    - xp, yp, zp: arrays of pixel indices.
    - x, y, z: position of the Gaussian in real units (if pixels, enter dx=dy=dz=1).
    - sxy: sigma in the xy-plane in pixels.
    - sz: sigma in the z direction in pixels.
    - dx, dy, dz: voxel dimensions in real units.
    
    Returns:
    - intensity: a 3D array of the Gaussian intensity values.
    """
    # Computing the center position of each voxel
    xp_center = np.ceil(xp) - 0.5
    yp_center = np.ceil(yp) - 0.5
    zp_center = np.round(zp)
    
    # Calculate differences for erf function in xy plane
    diffx1 = (xp_center - 0.5) * dx - x
    diffx2 = (xp_center + 0.5) * dx - x
    diffy1 = (yp_center - 0.5) * dy - y
    diffy2 = (yp_center + 0.5) * dy - y
    
    # Normalize differences by the standard deviations
    diffx1 /= np.sqrt(2) * sxy
    diffx2 /= np.sqrt(2) * sxy
    diffy1 /= np.sqrt(2) * sxy
    diffy2 /= np.sqrt(2) * sxy
    
    # Gaussian profile in z without integration
    gz = np.exp(-((zp_center * dz - z) ** 2) / (2 * sz ** 2))
    
    # Calculate integrated intensity
    intensity = np.abs(erf(diffx1) - erf(diffx2)) * \
                np.abs(erf(diffy1) - erf(diffy2)) * \
                gz
    return intensity


def intensity_gaussian2D(xp, yp, x, y, sxy, dx, dy):
    """
    Computes the intensity of pixels assuming a standard 2D Gaussian PSF.
    
    Parameters:
    - xp, yp: Arrays of pixel indices.
    - x, y: Position of the Gaussian center in real units (use dx=1, dy=1 for pixel units).
    - sxy: Sigma for the Gaussian in pixels.
    - dx, dy: Pixel dimensions in real units.
    
    Returns:
    - intensity: A 2D array of the Gaussian intensity values.
    """
    # Compute the center position of each pixel
    xp_center = np.ceil(xp) - 0.5
    yp_center = np.ceil(yp) - 0.5
    
    # Calculate the Gaussian intensity
    gx = np.exp(-((xp_center * dx - x) ** 2) / (2 * sxy ** 2))
    gy = np.exp(-((yp_center * dy - y) ** 2) / (2 * sxy ** 2))
    intensity = gx * gy
    
    return intensity


def intensity_integrated_gaussian2D(xp, yp, x, y, sxy, dx, dy):
    """
    Computes the intensity of pixels assuming an integrated 2D Gaussian PSF.
    
    Parameters:
    - xp, yp: Arrays of pixel indices.
    - x, y: Position of the Gaussian center in real units (use dx=dy=1 for pixel units).
    - sxy: Sigma for the Gaussian in pixels.
    - dx, dy: Pixel dimensions in real units.
    
    Returns:
    - intensity: A 2D array of the integrated Gaussian intensity values.
    """
    # Adjust xp and yp to the center position of each pixel
    xp_center = np.ceil(xp) - 0.5
    yp_center = np.ceil(yp) - 0.5
    
    # Compute differences for the erf function
    diffx1 = (xp_center - 0.5) * dx - x
    diffx2 = (xp_center + 0.5) * dx - x
    diffy1 = (yp_center - 0.5) * dy - y
    diffy2 = (yp_center + 0.5) * dy - y
    
    # Normalize differences by the standard deviations
    diffx1 /= np.sqrt(2) * sxy
    diffx2 /= np.sqrt(2) * sxy
    diffy1 /= np.sqrt(2) * sxy
    diffy2 /= np.sqrt(2) * sxy
    
    # Calculate the integrated intensity using the error function
    intensity = np.abs(erf(diffx1) - erf(diffx2)) * np.abs(erf(diffy1) - erf(diffy2))
    
    return intensity