import os
import glob
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

import cv2
# from skimage.io import imread
from scipy.spatial import KDTree
from skimage.feature import peak_local_max
import tifffile
from pprint import pprint

# constants
CHANNELS = ['cy5', 'TxRed', 'cy3', 'FAM']
BASE_DIR = Path('path_to_processed_dataset')
RUN_ID = 'example_data'
PRISM_PANEL = 'PRISM30'
src_dir = BASE_DIR / RUN_ID
stc_dir = src_dir / 'stitched'
read_dir = src_dir / 'readout'
tmp_dir = read_dir / 'tmp'
read_dir.mkdir(exist_ok=True)
tmp_dir.mkdir(exist_ok=True)

# parameters
TOPHAT_KERNEL_SIZE = 7
TOPHAT_BREAK = 100 # 100
# LOCAL_MAX_ABS_THRE = 200 # 200
LOCAL_MAX_ABS_THRE_CH = {'cy5': 200, 'TxRed': 200, 'FAM': 200, 'cy3': 200}
# LOCAL_MAX_ABS_THRE_CH = {'cy3': None, 'cy5': None, 'FAM': None, 'TxRed': None} 
INTENSITY_THRE = None # INTENSITY_ABS_THRE should be a little bigger than LOCAL_MAX_ABS_THRE
CAL_SNR = False

# Threshold after extracting points
SUM_THRESHOLD = 800 # SUM_THRESHOLD should be 4 * INTENSITY_ABS_THRE
G_ABS_THRESHOLD = 1000
G_THRESHOLD = 3 #
G_MAXVALUE = 5 #


def tophat_spots(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL_SIZE, TOPHAT_KERNEL_SIZE))
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def divide_main(shape, max_volume=10**8, overlap=500, data_dict=None, verbose=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if len(shape) == 3:
                zrange, xrange, yrange = shape
            elif len(shape) == 2:
                yrange, xrange = shape
                zrange = 1

            xy_size = int(np.sqrt(max_volume / zrange))
            x_num = -(-(xrange - overlap) // (xy_size - overlap))
            y_num = -(-(yrange - overlap) // (xy_size - overlap))
            cut_x = xrange // x_num + overlap
            cut_y = yrange // y_num + overlap

            if verbose: print(f"n_tile: {x_num * y_num};",
                  f"\nx_slice_num: {x_num};", f"y_slice_num: {y_num};",
                  f"\nblock_x: {cut_x};", f"block_y: {cut_y};", f"overlap: {overlap};")
            
            with tqdm(total=x_num * y_num, desc='tile', disable=not verbose) as pbar:
                for x_pos in range(x_num):
                    pad_x = x_pos * (cut_x - overlap)  # 计算当前x块的pad_x
                    for y_pos in range(y_num):
                        pad_y = y_pos * (cut_y - overlap)  # 计算当前y块的pad_y
                        func_args = {
                            'pad_x': pad_x, 'cut_x': cut_x,
                            'pad_y': pad_y, 'cut_y': cut_y,
                            'x_pos': x_pos, 'y_pos': y_pos,
                            'x_num': x_num, 'y_num': y_num,
                            'overlap': overlap,
                            'data_dict': data_dict
                        }
                        func_args.update(kwargs)  # 将额外的kwargs合并进来
                        func(*args, **func_args)
                        pbar.update(1)
        return wrapper
    return decorator


def extract_coordinates(image, local_max_thre=200, intensity_thre=None): #quantile=0.96):
    meta = {}
    coordinates = peak_local_max(image, min_distance=2, threshold_abs=local_max_thre)
    meta['Coordinates brighter than given SNR'] = coordinates.shape[0]
    meta['Image mean intensity'] = float(np.mean(image))
    if intensity_thre is not None:
        if intensity_thre<=1:
            intensities = image[coordinates[:, 0], coordinates[:, 1]]
            meta[f'{intensity_thre} quantile'] = float(np.quantile(intensities, intensity_thre))
            intensity_thre = np.quantile(intensities, intensity_thre)
        coordinates = coordinates[image[coordinates[:, 0], coordinates[:, 1]]>intensity_thre]

    meta['Final spots count'] = coordinates.shape[0]
    return coordinates


def calculate_snr(image, points, neighborhood_size=10, verbose=True):
    """
    Calculate the Signal-to-Noise Ratio for given points in an image.
    SNR is defined as the value at the point divided by the minimum value in its neighborhood.
    
    :param image_path: Path to the TIF image file.
    :param points: List of tuples (x, y) representing the coordinates in the image.
    :param neighborhood_size: Size of the square neighborhood around the point.
    :return: Dictionary of SNR values for each point.
    """
    # Calculate the half size of the neighborhood
    offset = -int( - neighborhood_size // 2)
    padded_img = np.pad(image, offset, mode='edge')
    snr_values = []
    # Process each point
    for point in tqdm(points, disable=not verbose, desc='Calculating snr', position=1, leave=False):
        y, x = point
        # Define the neighborhood bounds
        x_min = x
        x_max = x + neighborhood_size
        y_min = y
        y_max = y + neighborhood_size
        
        # Extract the neighborhood
        neighborhood = padded_img[y_min:y_max, x_min:x_max]
        # Find the minimum value in the neighborhood
        # min_val = np.percentile(neighborhood, 5)
        min_val = np.min(neighborhood)
        # Calculate SNR
        point_val = image[y, x]
        snr = point_val / min_val if min_val != 0 else float('inf')  # Avoid division by zero
        # Store the result
        snr_values.append(snr)
    
    return snr_values


def extract_signal(image_big, pad_x, cut_x, pad_y, cut_y, 
                   tophat_mean, snr=8, abs_thre=200, # peak local max threshold
                   QUANTILE=0.1, # intensity threshold
                   check_snr=False, 
                   kernel=np.ones((5, 5), np.uint8)):

    image_raw = image_big[pad_y: pad_y + cut_y, pad_x: pad_x + cut_x]
    # tophat spots
    image = tophat_spots(image_raw)
    image[image < TOPHAT_BREAK] = 0

    # extract coordinates
    # coordinates = extract_coordinates(image, threshold_abs=snr * tophat_mean, quantile=QUANTILE)
    if abs_thre is None: coordinates = extract_coordinates(image, local_max_thre=tophat_mean * snr, intensity_thre=INTENSITY_THRE) #quantile=QUANTILE)
    else: coordinates = extract_coordinates(image, local_max_thre=min(abs_thre, tophat_mean * snr), intensity_thre=INTENSITY_THRE) #quantile=QUANTILE)

    if check_snr: snr = calculate_snr(image_raw, coordinates)
    else: snr = None
    del image_raw

    # find signal
    Maxima = np.zeros(image.shape, dtype=np.uint16)
    Maxima[coordinates[:, 0], coordinates[:, 1]] = 255
    image[Maxima <= 0] = 0  # Mask

    # dilation of image
    image = cv2.dilate(image, kernel, iterations=1)
    return coordinates, snr, image


def read_intensity(image_dict, coordinates, channel, snr=None):
    if snr is None: intensity = pd.DataFrame({'Y': coordinates[:, 0], 'X': coordinates[:, 1], 'Channel': [channel]*len(coordinates)})
    else: intensity = pd.DataFrame({'Y': coordinates[:, 0], 'X': coordinates[:, 1], 'Channel': [channel]*len(coordinates), 'snr': snr})

    for image_name, image in image_dict.items():
        intensity[image_name] = image[coordinates[:, 0], coordinates[:, 1]]
    return intensity


def remove_duplicates(coordinates):
    tree = KDTree(coordinates)
    pairs = tree.query_pairs(2)
    # print(f'{len(pairs)} duplicates pairs')
    neighbors = {}  # dictionary of neighbors
    for i, j in pairs:  # iterate over all pairs
        if i not in neighbors: neighbors[i] = set([j])
        else: neighbors[i].add(j)
        if j not in neighbors: neighbors[j] = set([i])
        else: neighbors[j].add(i)

    # print(f'{len(neighbors)} neighbor entries')
    keep = []
    discard = set()  # a list would work, but I use a set for fast member testing with `in`
    nodes = set([s[0] for s in pairs]+[s[1] for s in pairs])
    for node in nodes:
        if node not in discard:  # if node already in discard set: skip
            keep.append(node)  # add node to keep list
            # add node's neighbors to discard set
            discard.update(neighbors.get(node, set()))
    # print(f'{len(discard)} nodes discarded, {len(keep)} nodes kept')
    centroids_simplified = np.delete(coordinates, list(discard), axis=0)
    # print(f'{centroids_simplified.shape[0]} centroids after simplification')
    return centroids_simplified


def main(stc_dir=stc_dir, read_dir=read_dir, tmp_dir=tmp_dir, PRISM_PANEL='PRISM30', max_memory = 24): # unit: GB
    with tifffile.TiffFile(stc_dir/f'cyc_1_{CHANNELS[0]}.tif') as tif:
        image_shape = tif.series[0].shape
    if (image_shape[0] == 1 and len(image_shape)==3) or len(image_shape)<3:
        if len(image_shape)==3:
            image_shape = image_shape[1:]
        print('2D_image_shape_y, x: ({})'.format(', '.join(map(str, image_shape))))
    else:
        print('3D_image_shape_z, x, y: ({})'.format(', '.join(map(str, image_shape))))

    # 获取文件的存储空间大小
    file_size_bytes = os.path.getsize(stc_dir / f'cyc_1_{CHANNELS[0]}.tif')
    file_size_gegabytes = file_size_bytes / (1024 ** 3)  # 转换为G字节(GB)
    print(f'Image file size: {file_size_bytes} bytes ({file_size_gegabytes:.2f} GB)')
    # max_volume = (max_memory - file_size_gegabytes) * 1024 ** 3 / 2 / 4  # unit: pixel
    max_volume = (max_memory) * 1024 ** 3 # byte
    max_volume /= 2 # 2byte for one pixel
    max_volume /= 4 # 4 channel
    max_volume /= 2 # tophat to double
    max_volume /= 2 # more space

    # extract intensity
    tophat_mean_dict = dict()
    for channel in CHANNELS:
        with tifffile.TiffFile(stc_dir / f'cyc_1_{channel}.tif') as tif:
            image = tif.asarray()
            tophat_mean_dict[channel] = np.mean(tophat_spots(image))
            memmap_shape = image.shape
            memmap_dtype = image.dtype
            if f'cyc_1_{channel}.dat' not in os.listdir(tmp_dir):
                memmap_array = np.memmap(tmp_dir / f'cyc_1_{channel}.dat', dtype=memmap_dtype, mode='w+', shape=memmap_shape)
                memmap_array[:] = image[:]
                memmap_array.flush()
                del memmap_array

    print('tophat_mean:')
    pprint(tophat_mean_dict)
    tophat_mean_dict = {k: float(v) for k, v in tophat_mean_dict.items()}
    with open(tmp_dir / 'tophat_mean.yaml', 'w') as f: yaml.dump({'tophat_mean': tophat_mean_dict}, f)


    @divide_main(shape=image_shape, max_volume=max_volume, overlap=500)
    def extract_intensity(channels=CHANNELS, **kwargs):
        pad_x = kwargs['pad_x']
        cut_x = kwargs['cut_x']
        pad_y = kwargs['pad_y']
        cut_y = kwargs['cut_y']
        x_pos = kwargs['x_pos']
        y_pos = kwargs['y_pos']

        image_dict = {}
        coordinate_dict = {}
        snr_dict = {}

        def process_channel(channel, check_snr=CAL_SNR):
            image_path = tmp_dir / f'cyc_1_{channel}.dat'
            image = np.memmap(str(image_path), dtype=memmap_dtype, mode='r', shape=memmap_shape)
            coordinate, snr, image_data = extract_signal(image, pad_x, cut_x, pad_y, cut_y, 
                                                         snr=8, tophat_mean=tophat_mean_dict[channel], abs_thre=LOCAL_MAX_ABS_THRE_CH[channel], check_snr=check_snr)
            return channel, coordinate, snr, image_data
        
        with Pool() as pool:
            results = pool.map(process_channel, channels)

        for channel, coordinate, snr, image_data in results:
            coordinate_dict[channel] = coordinate
            image_dict[channel] = image_data
            if CAL_SNR: snr_dict[channel] = snr

        if CAL_SNR: intensity = pd.concat([read_intensity(image_dict, coordinate_dict[channel], channel=channel, snr=snr_dict[channel]) for channel in coordinate_dict.keys()])
        else: intensity = pd.concat([read_intensity(image_dict, coordinate_dict[channel], channel=channel) for channel in coordinate_dict.keys()])
        intensity['X'] = intensity['X'] + pad_x
        intensity['Y'] = intensity['Y'] + pad_y
        intensity.to_csv(tmp_dir / f'intensity_block_{x_pos}_{y_pos}.csv')

    extract_intensity(channels=CHANNELS)


    # stitch all block
    global intensity
    intensity = pd.DataFrame()

    @divide_main(shape=image_shape, max_volume=max_volume, overlap=500, verbose=False)
    def stitch_all_block(**kwargs):
        global intensity
        pad_x = kwargs['pad_x']
        cut_x = kwargs['cut_x']
        pad_y = kwargs['pad_y']
        cut_y = kwargs['cut_y']
        x_pos = kwargs['x_pos']
        y_pos = kwargs['y_pos']
        x_num = kwargs['x_num']
        y_num = kwargs['y_num']
        overlap = kwargs['overlap']

        tmp_intensity = pd.read_csv(tmp_dir / f'intensity_block_{x_pos}_{y_pos}.csv', index_col=0)
        xmin, xmax = pad_x + overlap//4, pad_x + cut_x - overlap//4
        if x_pos == 1: xmin = 0
        elif x_pos == x_num: xmax = pad_x + cut_x

        ymin, ymax = pad_y + overlap//4, pad_y + cut_y - overlap//4
        if y_pos == 1: ymin = 0
        elif y_pos == y_num: ymax = pad_y + cut_y

        tmp_intensity = tmp_intensity[(tmp_intensity['Y'] >= ymin) & (tmp_intensity['Y'] <= ymax) &
                                      (tmp_intensity['X'] >= xmin) & (tmp_intensity['X'] <= xmax)]
        
        intensity = pd.concat([intensity, tmp_intensity])

    stitch_all_block()


    # save original intensity file
    intensity = intensity.rename(columns={'cy5': 'R', 'TxRed': 'Ye', 'cy3': 'G', 'FAM': 'B'})
    intensity = intensity.reset_index(drop=True)
    intensity.to_csv(tmp_dir / 'intensity_raw.csv')
    print('raw_read:', len(intensity))

    # crosstalk elimination
    intensity['B'] = intensity['B'] - intensity['G'] * 0.25
    intensity['B'] = np.maximum(intensity['B'], 0)

    # Scale
    intensity['Scaled_R'] = intensity['R']
    intensity['Scaled_Ye'] = intensity['Ye']
    intensity['Scaled_G'] = intensity['G'] * 2.5
    intensity['Scaled_B'] = intensity['B'] * 0.75

    # threshold by intensity
    intensity['sum'] = intensity['Scaled_R'] + intensity['Scaled_Ye'] + intensity['Scaled_B']

    # normalize
    intensity['G/A'] = intensity['Scaled_G'] / intensity['sum']

    # threshold by intensity
    if PRISM_PANEL in ('PRISM31', 'PRISM46', 'PRISM64'):
        intensity = intensity[(intensity['sum'] >= SUM_THRESHOLD) | ((intensity['G/A'] >= G_THRESHOLD) & (intensity['Scaled_G'] > G_ABS_THRESHOLD))]
    elif PRISM_PANEL in ('PRISM30', 'PRISM45', 'PRISM63'):
        intensity = intensity[intensity['sum'] >= SUM_THRESHOLD]
        intensity = intensity.dropna()
    intensity.loc[intensity['G/A'] > G_MAXVALUE, 'G/A'] = G_MAXVALUE
    print('drop_low_intensity:', len(intensity))


    # deduplicate
    intensity['ID'] = intensity['Y'] * 10**7 + intensity['X']
    intensity = intensity.drop_duplicates(subset=['Y', 'X'])

    df = intensity[['Y', 'X', 'R', 'Ye', 'B', 'G']]
    df_reduced = pd.DataFrame()
    coordinates = df[['Y', 'X']].values
    coordinates = remove_duplicates(coordinates)
    df_reduced = pd.DataFrame(coordinates, columns=['Y', 'X'])

    df_reduced['ID'] = df_reduced['Y'] * 10**7 + df_reduced['X']
    intensity = intensity[intensity['ID'].isin(df_reduced['ID'])]
    intensity = intensity.drop(columns=['ID'])
    intensity.to_csv(read_dir / 'intensity_deduplicated.csv')
    print('deduplicate:', len(intensity))

    # remove tmp files
    for file_path in glob.glob(str(tmp_dir / '*.dat')): os.remove(file_path)
    print(f"Removed .dat files")
        
    # if os.path.exists(tmp_dir):
    #     shutil.rmtree(tmp_dir)
    #     print(f"The directory '{tmp_dir}' and all its contents have been removed.")
    # else: print(f"The directory '{tmp_dir}' does not exist.")

if __name__ == '__main__':
    # batch process
      # copy this file to the dest_dir
    current_file_path = os.path.abspath(__file__)
    target_file_path = os.path.join(read_dir, os.path.basename(current_file_path))
    try: shutil.copy(current_file_path, target_file_path)
    except shutil.SameFileError: print('The file already exists in the destination directory.')
    except PermissionError: print("Permission denied: Unable to copy the file.")
    except FileNotFoundError: print("File not found: Source file does not exist.")
    except Exception as e: print(f"An error occurred: {e}")
    print('RUN_ID:', RUN_ID)
    main(PRISM_PANEL=PRISM_PANEL, stc_dir=stc_dir, read_dir=read_dir, tmp_dir=tmp_dir, max_memory=24) # unit: GB