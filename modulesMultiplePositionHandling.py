import os
import glob
import pandas as pd
import shutil
import modules.modules as M

"""
These functions handle the processing specific to runs which have _multiple_ acquisitions, captured for _multiple_ positions

    Structure must be
    
root
|
|-- position_0
    |--1
        |-- *.ptu
    |--2
        |-- *.ptu
    |--[...]
|-- position_1
|-- [..]
|-- position_info.csv

position_info must have columns 'dir_name', 'dist_from_start'
"""

def moveFaultyPositionalAcquisitions(root_dir, histThreshold485, histThreshold640, failed_acqs_dir):
    """
    Sometimes the acquisition is clearly tainted which can be caught by finding runs where the intensity threshold
    exceeds a threshold provided

    This function moves these acqs into a separate 'failed_acquisitions' directory to allow for subsequent photon 
    processing (below), otherwise the processing seems to trip over...
    """
    for filedir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, filedir)):
            print(f'Checking subdir: {filedir}')

            # For each position find photon files
            files_in_dir = glob.glob(os.path.join(root_dir, filedir,'*/*.ptu'))

            # Get subdirectory associated with photon file (acquisition)
            acq_dirs = [os.path.split(f)[0] for f in files_in_dir]
            print(f'\tAcquisition directories: {len(acq_dirs)}')

            # for each acquisition check if intensity histogram is sensible 
            for acq_dir in acq_dirs:
                # Load Histogram
                hist_485 = pd.read_csv(os.path.join(acq_dir, '485ChannelOnHistogram.csv'), names = ['T', 'Hist'])
                hist_640 = pd.read_csv(os.path.join(acq_dir, '640ChannelOnHistogram.csv'), names = ['T', 'Hist'])

                # Run Check
                if max(hist_485.Hist) >= histThreshold485 or max(hist_640.Hist) >= histThreshold640:
                    acq_num = os.path.split(acq_dir)[1]
                    new_subdir_name = f'{filedir}_acq_{os.path.split(acq_dir)[1]}'
                    new_target_dir = os.path.join(failed_acqs_dir, new_subdir_name)
                    print(f"\nMOVING DIR {acq_dir} --> \n\t{new_target_dir}\n")
                    shutil.move(acq_dir, new_target_dir)
                else:
                    pass

def processSemiContinuousDeviceRun(root_dir, params, tBounds, tauSettings, debug=True):
    """
    Processing for each position capturing the tagging with positional data (using the position_info.csv)

    Saves each positional burst aggregation csv within the directory !this will be overwritten if already exists!
    """
    # Read positional information
    position_info = pd.read_csv(os.path.join(root_dir,'position_info.csv'), 
                            usecols = ['dir_name', 'dist_from_start'], 
                            index_col='dir_name')
    
    device_data = {}
    for filedir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, filedir)):
            # Get Position Info
            _channel_dist = position_info.loc[filedir].values[0]

            # Get files to join from this subdir
            files_in_dir = glob.glob(os.path.join(root_dir, filedir,'*/*.ptu'))
            print(f'Processing: {filedir}: {_channel_dist} um')
            print(f'\tNumber photon files in directory: {len(files_in_dir)}')

            # Crunch data 
            this_data_entry = M.parse_multiple_photon_files(file_list=files_in_dir, 
                                                            tBounds=tBounds, 
                                                            params=params, 
                                                            tauSettings=tauSettings, 
                                                            debug=debug)
            this_data_entry['position'] = filedir
            this_data_entry['channel_dist_um'] = _channel_dist
            
            # Save in directory
            _save_path = os.path.join(root_dir, filedir, "burstDataAggregated.csv")
            this_data_entry.to_csv(_save_path, index=False)

            # Add to total dict
            device_data[filedir] = this_data_entry
    return device_data