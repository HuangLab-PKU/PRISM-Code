from lib.utils.os_utils import try_mkdir
import re
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from tifffile import imread, imwrite
# from skimage.io import imread
# from skimage.io import imsave
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift


def get_shift(ref, img):
    """Get the shift between two images.

    Parameters
    ----------
    ref : ndarray
        Reference image.
    img : ndarray
        Image to register.

    Returns
    -------
    shift : tuple
        (y, x) shift.

    """
    shift, _, _ = phase_cross_correlation(ref, img, upsample_factor=100)
    return shift


def register(img, shift):
    """Register an image to a reference image.

    Parameters
    ----------
    img : ndarray
        Image to register.
    shift : tuple
        (y, x) shift.

    Returns
    -------
    out_img : ndarray
        Registered image.

    """
    registered = fourier_shift(np.fft.fftn(img), shift)
    registered = np.fft.ifftn(registered)
    out_img = np.uint16(registered.real)
    return out_img


def register_manual(ref_dir, src_dir, dest_dir, im_names=None):
    """Register all images in a directory.
    
    Parameters
    ----------
    ref_dir : str
        Directory with the reference images.
    src_dir : str
        Directory with the images to register.
    dest_dir : str
        Output directory.
    im_names : list
        Names of the images to register.

    Returns
    -------
    None.
    
    """
    ref_dir = Path(ref_dir)
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)
    if im_names is None:
        im_names = [x.name for x in list(ref_dir.glob('*.tif'))]
    for im_name in tqdm(im_names, desc='Registering'):
        ref_im = imread(ref_dir/im_name)
        src_im = imread(src_dir/im_name)
        shift = get_shift(ref_im, src_im)
        out_im = register(src_im, shift)
        imwrite(dest_dir/im_name, out_im)


# def register_meta(in_dir, out_dir, chns, names, ref_cyc=1, ref_chn='cy3'):
#     """Register all images in a directory.

#     Parameters
#     ----------
#     in_dir : str
#         Input directory.
#     out_dir : str
#         Output directory.
#     chns : list
#         Channels to register.
#     names : list
#         Names of the images to register.
#     ref_cyc : int
#         Cycle to use as reference.
#     ref_chn : str
#         Channel to use as reference.

#     Returns
#     -------
#     df : pandas.DataFrame
#         DataFrame of the integer offsets for stitching.
#     """
#     try_mkdir(out_dir)
#     pattern = r'^cyc_\d+_\w+'
#     alt_chns = [c for c in chns if c != ref_chn]
#     cyc_chn_list = Path(in_dir).glob('cyc_*_*')
#     cyc_chn_list = [c for c in cyc_chn_list if re.match(pattern, c.name)]
#     src_list = [c for c in cyc_chn_list if c.name.split('_')[1] == str(ref_cyc)]
#     for d in tqdm(src_list, desc='Copying reference'): 
#         if not (Path(out_dir)/d.name).is_dir(): shutil.copytree(d, Path(out_dir)/d.name)
#     df = pd.DataFrame()
#     ref_list = [c for c in cyc_chn_list if c.name.split('_')[2] == ref_chn]
#     ref_dir = Path(in_dir)/f'cyc_{ref_cyc}_{ref_chn}'
#     ref_list.remove(ref_dir)
#     for name in tqdm(names, desc='Registering'):
#         offsets = []
#         ref_im = imread(ref_dir/name)
#         for cyc_chn in ref_list:
#             cyc = cyc_chn.name.split('_')[1]
#             src_im = imread(cyc_chn/name)
#             dest_dir = Path(out_dir)/cyc_chn.name
#             dest_dir.mkdir(exist_ok=True)
#             try:
#                 shift = get_shift(ref_im, src_im)
#             except AttributeError:
#                 offsets.append('0 0')
#                 continue
#             shift_res = shift - np.round(shift)
#             offsets.append(np.array2string(np.round(shift).astype(int)).strip('[ ]'))
#             out_im = register(src_im, shift_res)
#             imsave(dest_dir/name, out_im, check_contrast=False)
#             for chn in alt_chns:
#                 cyc_chn_alt = Path(in_dir)/f'cyc_{cyc}_{chn}'
#                 if cyc_chn_alt in cyc_chn_list:
#                     dest_dir_alt = Path(out_dir)/cyc_chn_alt.name
#                     dest_dir_alt.mkdir(exist_ok=True)
#                     src_im = imread(cyc_chn_alt/name)
#                     out_im = register(src_im, shift_res)
#                     imsave(dest_dir_alt/name, out_im, check_contrast=False)
#         df[name] = offsets
#     df.index = [s.name.split('_')[1] for s in ref_list]
#     return df


def register_meta(in_dir, out_dir, chns, names, ref_cyc=1, ref_chn='cy3'):
    """Register all images in a directory.

    Parameters
    ----------
    in_dir : str
        Input directory.
    out_dir : str
        Output directory.
    chns : list
        Channels to register.
    names : list
        Names of the images to register.
    ref_cyc : int
        Cycle to use as reference.
    ref_chn : str
        Channel to use as reference.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame of the integer offsets for stitching.
    """

    try_mkdir(out_dir)
    pattern = r'^cyc_\d+_\w+'
    cyc_chn_list = [c for c in Path(in_dir).glob('cyc_*_*') if re.match(pattern, c.name)]
    src_list = [c for c in cyc_chn_list if c.name.split('_')[1] == str(ref_cyc)]
    
    # Copy reference directories
    for d in tqdm(src_list, desc='Copying reference'):
        dest_path = Path(out_dir) / d.name
        if not dest_path.is_dir():
            shutil.copytree(d, dest_path)
    
    offset_dict = {}
    ref_dir = Path(in_dir) / f'cyc_{ref_cyc}_{ref_chn}'
    ref_list = [c for c in cyc_chn_list if c.name.split('_')[2] == ref_chn]
    if ref_dir in ref_list: ref_list.remove(ref_dir)

    for name in tqdm(names, desc='Registering'):
        offsets = []
        ref_im = imread(ref_dir / name)
        for cyc_chn in ref_list:
            cyc = cyc_chn.name.split('_')[1]
            src_im = imread(cyc_chn / name)
            dest_dir = Path(out_dir) / cyc_chn.name
            dest_dir.mkdir(exist_ok=True)
            try:
                shift = get_shift(ref_im, src_im)
                shift_res = shift - np.round(shift)
                offset = np.array2string(np.round(shift).astype(int)).strip('[ ]')
            except AttributeError:
                offset = '0 0'
            
            offsets.append(offset)
            out_im = register(src_im, shift_res)
            imwrite(dest_dir / name, out_im)
            
            for chn in [c for c in chns if c != ref_chn]:
                cyc_chn_alt = Path(in_dir) / f'cyc_{cyc}_{chn}'
                if cyc_chn_alt in cyc_chn_list:
                    dest_dir_alt = Path(out_dir) / cyc_chn_alt.name
                    dest_dir_alt.mkdir(exist_ok=True)
                    src_im = imread(cyc_chn_alt / name)
                    out_im = register(src_im, shift_res)
                    imwrite(dest_dir_alt / name, out_im)

        offset_dict[name] = offsets

    # Create DataFrame once at the end
    df = pd.DataFrame(offset_dict, index=[cyc_chn.name.split('_')[1] for cyc_chn in ref_list])
    return df


def main():
    pass


if __name__ == "__main__":
    main()
