import modules.modules as M
import os
import glob
import pickle
import numpy as np

def getNoiseAndBurstCutDicts(photonData, cut):
    """
    As on tin, requires already parsed photon_data
    """
    # Cut Length set to 1 for a full burst/noise analysis
    _Nparams = {'burstCut': None, 'noiseCut': cut, 'burstLen': 1, 'noiseLen': 1}
    nLoc, nIdx, nLen = M.findNoise(photonData = photonData, params=_Nparams, debug=False)

    # Add to noise data dictionary at this specific burst cut value
    _noiseCutDict = {'Loc': nLoc, 'Idx': nIdx, 'Len': nLen}
      
    # Cut Length set to 1 for a full burst/noise analysis
    _Bparams = {'burstCut': cut, 'noiseCut': None, 'burstLen': 1, 'noiseLen': 1}                
    bLoc, bIdx, bLen = M.findBurst(photonData = photonData, params=_Bparams, debug=False)

    # Add to noise data dictionary at this specific burst cut value
    _burstCutDict = {'Loc': bLoc, 'Idx': bIdx, 'Len': bLen}
    
    return _noiseCutDict, _burstCutDict 

def getCutDataDictFromPTU_PositionalExperiments(root, tBounds, cuts, saveCutData=False, cutDataPath=None, debug=True, overwrite=False):
    """
    Calculates burst/noise information for a range of interphoton time thresholds (provided) for
    an entire experiment, assuming directory structure below
    
    Expects a directory structure
    Root
    |--position_0
       |-- 1
       |-- 2
       |-- [...]
    |--position_1
       |-- 1
       |-- 2
       |-- [...]
    |-- [...]
    """
    # Run some checks before hand to save time
    if saveCutData:
        assert cutDataPath is not None, "Provided save path for cutData!"
        if os.path.exists(cutDataPath) and not overwrite:
            raise f"cutData at {cutDataPath} exists and overwrite not set to true!"
                
    # Crunch Data
    cutData = {}    
    for positionDir in os.listdir(root):      
        if os.path.isdir(os.path.join(root,positionDir)): 
            
            # Important signature to find the ptu files!
            acquisitions = glob.glob(os.path.join(root,positionDir) + '/' + '/*/*.ptu')
            if debug:
                print(f'Processing: {positionDir}')
                print(f'\tNumber photon files in directory: {len(acquisitions)}')

            _cutData = {}
            for acq in acquisitions:
                acq_num = acq.split(os.sep)[-2]
                
                # Load & Add Photon Data
                _photon_data = M.getPhotonData(file=acq, tBounds=tBounds)

                # Calculate Burst Analysis
                _noiseCutData = {}
                _burstCutData = {}
                for cut in cuts:
                    _noiseCutData[cut], _burstCutData[cut] = getNoiseAndBurstCutDicts(photonData=_photon_data, cut=cut)
                _cutData[acq_num] = {'noise': _noiseCutData, 'burst':_burstCutData}

            cutData[positionDir] = _cutData
    
    # Save checks are carried out above
    if saveCutData:
        with open(cutDataPath, 'wb') as f:
            pickle.dump(cutData, f)
        
    return cutData

def calcHistFromCutDataDict(cutData, bins):
    """
    Calculates histogram for noise/burst data dictionary
    Returns dictionary structure with new 'Hist' field
    
    Expects a cutDataDict with structure
    {position_0:
       |-- 1
           |-- noise
               |-- cut1
                   |-- true location data (relates to original photon data)
                   |-- true index data (relates to original photon data)
                   |-- true length data
               |-- cut2
               |-- [...]
           |-- burst
               |-- [...]
       |-- 2
       |-- [...]
    position_1:
       |-- 1
       |-- 2
       |-- [...]
    [...]}
    """    
    # Loop through position
    for position, position_dict in cutData.items():
        # Loop through acquisitions
        for acq, acq_dict in position_dict.items():
            # Loop through each cut (for noise and burst)
            
            # Noise
            for cut, values in acq_dict['noise'].items():
                # Calculate new histogram
                nHist, _ = np.histogram(values['Len'], bins)
                # Update
                cutData[position][acq]['noise'][cut]['Hist'] = nHist
            
            # Burst
            for cut, values in acq_dict['burst'].items():
                # Calculate new histogram
                bHist, _ = np.histogram(values['Len'], bins)

                # Update
                cutData[position][acq]['burst'][cut]['Hist'] = bHist
    return cutData