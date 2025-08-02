import os
import numpy as np
# import param
from skimage.io import imread, imsave
from scipy.ndimage import gaussian_laplace

# self defined packages
from lib.AIRLOCALIZE.image_process import scale_tiff, perform_DoG, substract_with_scale

class AirLocalizeData:
    def __init__(self):
        self.reset()

    def reset(self):
        self.srcDir = ''
        self.fList = []
        self.curFile = ''
        self.fileIdx = 0
        self.img = None
        self.smooth = None
        # self.isMovie = False
        # self.nFrames = 0
        # self.curFrame = 0

    def reset_current_file(self):
        """Resets everything but the file list and the isMovie flag."""
        self.curFile = ''
        self.fileIdx = 0
        self.img = None
        self.smooth = None
        # self.nFrames = 0
        # self.curFrame = 0

    def set_flist_from_params(self, params):
        # Placeholder for setting file list from parameters
        # This method will involve file handling and directory searching based on the parameters
        if params['fileProcessingMode'] == 'singleFile':
            self.fList = [params['dataFileName']]
        elif params['fileProcessingMode'] == 'batch':
            self.srcDir = params['dataFileName']
            # self.fList = [_ for _ in os.listdir(params['dataFileName']) if _.endswith('.tif')]
            self.fList = [f'{_}.tif' for _ in params['threshLevel'].keys()]

    def set_flist(self, flist):
        self.reset()
        if isinstance(flist, list):
            self.fList = flist
        else:
            print('Input file list has wrong format; resetting airlocalize data object.')

    def get_flist(self):
        return self.fList

    def set_file_idx(self, idx):
        if idx > len(self.fList) or idx <= 0:
            print(f'Cannot access file index {idx}; file list has {len(self.fList)} entries.')
            return
        self.reset_current_file()
        self.fileIdx = idx
        self.curFile = self.fList[idx - 1]  # Adjust for zero-based indexing

    def get_file_idx(self):
        return self.fileIdx

    def set_cur_file(self, file_name):
        if file_name not in self.fList:
            print(f'Desired file {file_name} is not part of existing file list; cannot set as current.')
        else:
            idx = self.fList.index(file_name) + 1  # Adjust for one-based indexing in the setter
            self.set_file_idx(idx)

    def get_cur_file(self):
        return self.curFile

    # def set_nframes(self):
    #     # Logic to determine the number of frames; may involve loading the image if not already loaded
    #     pass

    # def set_cur_frame(self, new_frame):
    #     # Logic to update the current frame; involves checking if new_frame is valid
    #     pass

    def is_file_index_img_loaded(self, idx):
        # Checks if the image for the given index is loaded
        return idx == self.fileIdx and self.img is not None

    def retrieve_img(self, params):
        # Placeholder for loading an image based on current file index
        verbose = params['verbose']
        if verbose: print(f"Retrieving image for {self.curFile}...")
        self.img = imread(os.path.join(self.srcDir, self.curFile))
        if len(self.img.shape) == 3: self.img = np.transpose(self.img, (2, 1, 0))
        if params['scale']: self.img = scale_tiff(self.img, 
                                                  scale_lower_percentile=params['scaleLower'], 
                                                  scale_upper_percentile=params['scaleUpper'], 
                                                  scaleMaxRatio=params['scaleMaxRatio'], verbose=verbose)
        return self.img

    def retrieve_feature_img(self, params, verbose=True):
        mode = params['featureExtract']
        if verbose: print(f"Smoothing {self.curFile}, mode: {mode}...")
        if mode == 'DoG':
            self.smooth = perform_DoG(self.img, dog_sigma=(params['filterLo'], params['filterHi']), enhance=params['enhance'])
            if verbose: print(f"Smoothing {self.curFile} done.")
            if params['saveSmoothed']: imsave(os.path.join(params['saveDirName'], self.curFile.replace('.tif', '_feature.tif')), np.transpose(self.smooth, (2, 1, 0)), check_contrast=False)

        elif mode == 'LoG':
            sigma = params['filterLo']
            self.smooth = scale_tiff(-gaussian_laplace(self.img.astype(np.float64), sigma=sigma), verbose=params['verbose'])
            if params['enhance']: self.smooth = scale_tiff(substract_with_scale(1.5 * self.smooth, np.mean(self.smooth)), verbose=params['verbose'])
            
            if verbose: print(f"Smoothing {self.curFile} done.")
            if params['saveSmoothed']: imsave(os.path.join(params['saveDirName'], self.curFile.replace('.tif', '_feature.tif')), np.transpose(self.smooth, (2, 1, 0)), check_contrast=False)
                
        elif mode == 'none':
            self.smooth = self.img
        
        