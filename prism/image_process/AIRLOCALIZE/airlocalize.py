import os
import sys
import shutil
import yaml

import numpy as np
import pandas as pd

# self defined packages
from lib.AIRLOCALIZE.airlocalize_data import AirLocalizeData
from lib.AIRLOCALIZE.predetect import find_isolated_maxima
from lib.AIRLOCALIZE.localize import run_gaussian_fit_on_all_spots_in_image
from lib.AIRLOCALIZE.image_process import save_points


def airlocalize(parameters_filename='parameters.yaml', default_parameters='parameters_default.yaml', update={}):
    # read parameters from config file
    # parameters_filename = sys.argv[1]

    with open(default_parameters, 'r') as file: params = yaml.safe_load(file)
    with open(parameters_filename, 'r') as file: params.update(yaml.safe_load(file))
    params.update(update)

    # verbose
    verbose = True if params['verbose'] else False
    if verbose: split = '='*50
    
    al_data = AirLocalizeData()
    al_data.set_flist_from_params(params)

    # load file list from parameters
    try: 
        al_data.get_flist()
        if verbose: print(f"Found {len(al_data.fList)} files to analyze.")
    except: raise ValueError('could not find files to analyze.')
    
    # set the output dir
    if 'saveDirName' not in params: params['saveDirName'] = os.path.dirname(al_data.flist[0])
    os.makedirs(params['saveDirName'], exist_ok=True)

    # save the parameters to the output dir
    shutil.copy(parameters_filename, os.path.join(params['saveDirName'], os.path.basename(parameters_filename)))

    
    if params['multiChannelCandidates']:
        # loop over files to get spot candidates
        spot_candidates = np.array([])
        for i, file in enumerate(al_data.fList, start=1):
            al_data.set_file_idx(i)
            if verbose: print(f'{split}\nAnalyzing file: {al_data.curFile}...')

            # Retrieve current image and smooth it
            al_data.retrieve_img(params)

            # Verify that dimensionality agrees with settings
            nd = al_data.img.ndim
            num_dim_expect = params['numdim']
            if nd != num_dim_expect:
                print(f'Current image has {nd} dimensions but expecting {num_dim_expect}D data; skipping file')
                continue
            
            # Smooth the image and subtract background
            al_data.retrieve_feature_img(params, verbose=verbose)
            # Pre-detection of local maxima
            spot_candidates_tmp = find_isolated_maxima(al_data, params, verbose)
            spot_candidates = np.concatenate((spot_candidates, spot_candidates_tmp), axis=0) if spot_candidates.size else spot_candidates_tmp
            
        # Save the spot candidates to a csv file
        if verbose: print(f'Found {len(spot_candidates)} spot candidates.')
        if params['saveCandidates']:
            df = pd.DataFrame(spot_candidates, columns=['x', 'y', 'z', 'intensity'])
            df.to_csv(os.path.join(params['saveDirName'], 'spot_candidates.csv'), index=False)


        # loop over files to localize spots
        if verbose: print(f'{split}\nLocalizing spots...')
        for i, file in enumerate(al_data.fList, start=1):
            filename = os.path.basename(file).replace('.tif', '')
            al_data.set_file_idx(i)
            
            # Retrieve current image and smooth it
            al_data.retrieve_img(params)
            
            # Detection/quantification
            loc, loc_vars = run_gaussian_fit_on_all_spots_in_image(spot_candidates, al_data, params, verbose=verbose)
            # Save spot coordinates and detection parameters to a csv file
            df = pd.DataFrame(loc, columns=loc_vars)
            df = df[df['integratedIntensity']>0]
            df = df.sort_values(by=['residuals'], ascending=True)
            df.to_csv(os.path.join(params['saveDirName'], f'{filename}_spots.csv'), index=False)

            # Save the spot image if needed
            if params['saveSpotsImage']: 
                save_points(points=df, shape=(al_data.img.shape[2], al_data.img.shape[1], al_data.img.shape[0]), 
                            output_tiff_path=os.path.join(params['saveDirName'], f'{filename}_spots.tif'), 
                            radius=3, verbose=params['verbose'])
                
    else:
        # loop over files
        for i, file in enumerate(al_data.fList, start=1):
            filename = os.path.basename(file).replace('.tif', '')
            al_data.set_file_idx(i)
            if verbose: print(f'Analyzing file: {al_data.curFile}...')

            # Retrieve current image and smooth it
            al_data.retrieve_img(params)

            # Verify that dimensionality agrees with settings
            nd = al_data.img.ndim
            num_dim_expect = params['numdim']
            if nd != num_dim_expect:
                print(f'Current image has {nd} dimensions but expecting {num_dim_expect}D data; skipping file')
                continue
            
            # main steps
            # Smooth the image and subtract background
            al_data.retrieve_feature_img(params, verbose=verbose) 

            # Pre-detection of local maxima
            spot_candidates = find_isolated_maxima(al_data, params, verbose) 
            # Save the spot candidates to a csv file
            if verbose: print(f'Found {len(spot_candidates)} spot candidates.')
            if params['saveCandidates']:
                df = pd.DataFrame(spot_candidates, columns=['x', 'y', 'z', 'intensity'])
                df.to_csv(os.path.join(params['saveDirName'], f'{filename}_spot_candidates.csv'), index=False)

            # Detection/quantification
            loc, loc_vars = run_gaussian_fit_on_all_spots_in_image(spot_candidates, al_data, params, verbose=verbose) 
            # Save spot coordinates and detection parameters to a csv file
            df = pd.DataFrame(loc, columns=loc_vars)
            df = df[df['integratedIntensity']>0]
            df = df.sort_values(by=['residuals'], ascending=True)
            df.to_csv(os.path.join(params['saveDirName'], f'{filename}_spots.csv'), index=False)

            # Save the spot image if needed
            if params['saveSpotsImage']: 
                save_points(points=df, shape=(al_data.img.shape[2], al_data.img.shape[0], al_data.img.shape[1]), 
                            output_tiff_path=os.path.join(params['saveDirName'], f'{filename}_spots.tif'), 
                            radius=2, verbose=params['verbose'])

if __name__ == '__main__':
    airlocalize(parameters_filename='parameters.yaml')