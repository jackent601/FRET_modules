import struct
import time
import numpy as np
import pandas as pd
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import fftconvolve
import warnings

# Reads PTU file and returns lists of metadata, channel index, dtime and nsync in machine unit
def LoadPTU(filename, header_only = False, chatty = True):

	inputfile = open(filename, "rb")
	magic = inputfile.read(8).decode("utf-8").strip('\0')
	
	if magic != "PQTTTR":
		print("ERROR: Magic invalid, this is not a PTU file.")
		inputfile.close()
		raise RuntimeError()

	version = inputfile.read(8).decode("utf-8").strip('\0')

	tagDataList = []

	# Tag Types
	tyEmpty8	  = struct.unpack(">i", bytes.fromhex("FFFF0008"))[0]
	tyBool8	   = struct.unpack(">i", bytes.fromhex("00000008"))[0]
	tyInt8		= struct.unpack(">i", bytes.fromhex("10000008"))[0]
	tyBitSet64	= struct.unpack(">i", bytes.fromhex("11000008"))[0]
	tyColor8	  = struct.unpack(">i", bytes.fromhex("12000008"))[0]
	tyFloat8	  = struct.unpack(">i", bytes.fromhex("20000008"))[0]
	tyTDateTime   = struct.unpack(">i", bytes.fromhex("21000008"))[0]
	tyFloat8Array = struct.unpack(">i", bytes.fromhex("2001FFFF"))[0]
	tyAnsiString  = struct.unpack(">i", bytes.fromhex("4001FFFF"))[0]
	tyWideString  = struct.unpack(">i", bytes.fromhex("4002FFFF"))[0]
	tyBinaryBlob  = struct.unpack(">i", bytes.fromhex("FFFFFFFF"))[0]
	
	# Record types
	rtPicoHarpT3	 = struct.unpack(">i", bytes.fromhex('00010303'))[0]
	rtPicoHarpT2	 = struct.unpack(">i", bytes.fromhex('00010203'))[0]
	rtHydraHarpT3	= struct.unpack(">i", bytes.fromhex('00010304'))[0]
	rtHydraHarpT2	= struct.unpack(">i", bytes.fromhex('00010204'))[0]
	rtHydraHarp2T3   = struct.unpack(">i", bytes.fromhex('01010304'))[0]
	rtHydraHarp2T2   = struct.unpack(">i", bytes.fromhex('01010204'))[0]
	rtTimeHarp260NT3 = struct.unpack(">i", bytes.fromhex('00010305'))[0]
	rtTimeHarp260NT2 = struct.unpack(">i", bytes.fromhex('00010205'))[0]
	rtTimeHarp260PT3 = struct.unpack(">i", bytes.fromhex('00010306'))[0]
	rtTimeHarp260PT2 = struct.unpack(">i", bytes.fromhex('00010206'))[0]
	rtMultiHarpNT3   = struct.unpack(">i", bytes.fromhex('00010307'))[0]
	rtMultiHarpNT2   = struct.unpack(">i", bytes.fromhex('00010207'))[0]
	
	while True:
		tagIdent = inputfile.read(32).decode("utf-8").strip('\0')
		tagIdx = struct.unpack("<i", inputfile.read(4))[0]
		tagTyp = struct.unpack("<i", inputfile.read(4))[0]
		if tagIdx > -1:
			evalName = tagIdent + '(' + str(tagIdx) + ')'
		else:
			evalName = tagIdent
		#outputfile.write("\n%-40s" % evalName)
		if tagTyp == tyEmpty8:
			inputfile.read(8)
			#outputfile.write("<empty Tag>")
			tagDataList.append((evalName, "<empty Tag>"))
		elif tagTyp == tyBool8:
			tagInt = struct.unpack("<q", inputfile.read(8))[0]
			if tagInt == 0:
				#outputfile.write("False")
				tagDataList.append((evalName, "False"))
			else:
				#outputfile.write("True")
				tagDataList.append((evalName, "True"))
		elif tagTyp == tyInt8:
			tagInt = struct.unpack("<q", inputfile.read(8))[0]
			#outputfile.write("%d" % tagInt)
			tagDataList.append((evalName, tagInt))
		elif tagTyp == tyBitSet64:
			tagInt = struct.unpack("<q", inputfile.read(8))[0]
			#outputfile.write("{0:#0{1}x}".format(tagInt,18))
			tagDataList.append((evalName, tagInt))
		elif tagTyp == tyColor8:
			tagInt = struct.unpack("<q", inputfile.read(8))[0]
			#outputfile.write("{0:#0{1}x}".format(tagInt,18))
			tagDataList.append((evalName, tagInt))
		elif tagTyp == tyFloat8:
			tagFloat = struct.unpack("<d", inputfile.read(8))[0]
			#outputfile.write("%-3E" % tagFloat)
			tagDataList.append((evalName, tagFloat))
		elif tagTyp == tyFloat8Array:
			tagInt = struct.unpack("<q", inputfile.read(8))[0]
			#outputfile.write("<Float array with %d entries>" % tagInt/8)
			tagDataList.append((evalName, tagInt))
		elif tagTyp == tyTDateTime:
			tagFloat = struct.unpack("<d", inputfile.read(8))[0]
			tagTime = int((tagFloat - 25569) * 86400)
			tagTime = time.gmtime(tagTime)
			#outputfile.write(time.strftime("%a %b %d %H:%M:%S %Y", tagTime))
			tagDataList.append((evalName, tagTime))
		elif tagTyp == tyAnsiString:
			tagInt = struct.unpack("<q", inputfile.read(8))[0]
			tagString = inputfile.read(tagInt).decode("utf-8").strip("\0")
			#outputfile.write("%s" % tagString)
			tagDataList.append((evalName, tagString))
		elif tagTyp == tyWideString:
			tagInt = struct.unpack("<q", inputfile.read(8))[0]
			tagString = inputfile.read(tagInt).decode("utf-16le", errors="ignore").strip("\0")
			#outputfile.write(tagString)
			tagDataList.append((evalName, tagString))
		elif tagTyp == tyBinaryBlob:
			tagInt = struct.unpack("<q", inputfile.read(8))[0]
			#outputfile.write("<Binary blob with %d bytes>" % tagInt)
			tagDataList.append((evalName, tagInt))
		else:
			print("ERROR: Unknown tag type")
			raise RuntimeError()
		if tagIdent == "Header_End":
			break	
	tagNames = [tagDataList[i][0] for i in range(0, len(tagDataList))]
	tagValues = [tagDataList[i][1] for i in range(0, len(tagDataList))]

	# get important variables from headers
	numRecords = tagValues[tagNames.index("TTResult_NumberOfRecords")]
	globRes = tagValues[tagNames.index("MeasDesc_GlobalResolution")]
	Res = tagValues[tagNames.index("MeasDesc_Resolution")]
	total_time = tagValues[tagNames.index("MeasDesc_AcquisitionTime")]
	syncRate = tagValues[tagNames.index("TTResult_SyncRate")]
	recordType = tagValues[tagNames.index("TTResultFormat_TTTRRecType")]

	if recordType == rtPicoHarpT2:
		isT2 = True
		if chatty: print("PicoHarp T2 data")
	elif recordType == rtPicoHarpT3:
		isT2 = False
		if chatty: print("PicoHarp T3 data")
	elif recordType == rtHydraHarpT2:
		isT2 = True
		if chatty: print("HydraHarp V1 T2 data")
	elif recordType == rtHydraHarpT3:
		isT2 = False
		if chatty: print("HydraHarp V1 T3 data")
	elif recordType == rtHydraHarp2T2:
		isT2 = True
		if chatty: print("HydraHarp V2 T2 data")
	elif recordType == rtHydraHarp2T3:
		isT2 = False
		if chatty: print("HydraHarp V2 T3 data")
	elif recordType == rtTimeHarp260NT3:
		isT2 = False
		if chatty: print("TimeHarp260N T3 data")
	elif recordType == rtTimeHarp260NT2:
		isT2 = True
		if chatty: print("TimeHarp260N T2 data")
	elif recordType == rtTimeHarp260PT3:
		isT2 = False
		if chatty: print("TimeHarp260P T3 data")
	elif recordType == rtTimeHarp260PT2:
		isT2 = True
		if chatty: print("TimeHarp260P T2 data")
	elif recordType == rtMultiHarpNT3:
		isT2 = False
		if chatty: print("MultiHarp150N T3 data")
	elif recordType == rtMultiHarpNT2:
		isT2 = True
		if chatty: print("MultiHarp150N T2 data")
	else:
		print("ERROR: Unknown record type")
		raise RuntimeError()

	if chatty:
		print('\nTotal number of records: {0:.1g}\nGlobal resolution: {1:.1f} ns\nLocal resolution: {2:.1f} ps\nTotal resolved time: {5:.1f} ns\nTotal measurement time: {3:.1f} s\nSync rate: {4:.1f} MHz\n'
		  .format(numRecords, globRes  * 1e9, Res  * 1e12, total_time  * 1e-3, syncRate  * 1e-6, 2**15 * Res * 1e9))

	if header_only:
		return [tagNames, tagValues, globRes, Res, 0, 0]
	else:
		if chatty: print('Reading ptu...')
		data = struct.unpack("<{}I".format(numRecords), inputfile.read(4 * numRecords))
		if chatty: print('Converting data into numpy array...')
		data = np.array(data)

		T3WRAPAROUND = 1024

		# 1 000000 000000000000000 0000000000
		bitmask_special = 0x80000000
			
		# 0 111111 000000000000000 0000000000
		bitmask_channel = 0x7e000000
		
		# 1 111111 000000000000000 0000000000
		bitmask_header = bitmask_special | bitmask_channel
			
		# 0 000000 111111111111111 0000000000
		bitmask_dtime = 0x01fffc00
			
		# 0 000000 000000000000000 1111111111
		bitmask_nsync = 0x000003ff
			
		# 1 111111 000000000000000 0000000000
		bit_overflow = 0xfe000000

		bit_selected = []
		for channel in [0, 1]:
			bit_selected.append((channel) << 25)

		if chatty: print('Extraction step 1/6: getting header...')

		# ? ?????? 000000000000000 0000000000
		header = data & bitmask_header
		
		if chatty: print('Extraction step 2/6: getting overflows...')
		
		# list of True where a sync count overflow is happening
		mask_overflows = header == bit_overflow
		
		if chatty: print('Extraction step 3/6: getting photons...')
		
		# list of True where a photon event is recorded
		mask_data = (header == bit_selected[0]) | (header == bit_selected[1])
		
		if chatty: print('Extraction step 4/6: getting dt and sync numbers...')
		
		# read time and nsync
		_dtime = (data & bitmask_dtime) >> 10
		_nsync = (data & bitmask_nsync)
		_channel = (data & bitmask_channel) >> 25

		if chatty: print('Extraction step 5/6: getting cumulative sync numbers...')
		
		# list of cumulative overflows at all RECORD point
		_overflows_counts = np.cumsum(_nsync * mask_overflows, dtype='uint32')
		
		if chatty: print('Extraction step 6/6: getting channel data...')
		
		# list of cumulative overflows, dtime and individual nsync at all PHOTON point
		
		_noverflows = _overflows_counts[mask_data]
		dtime = _dtime[mask_data]
		nsync = _nsync[mask_data] + _noverflows  * T3WRAPAROUND
		channel = _channel[mask_data]
		
		if chatty: print("DONE!!\n")

		return [tagNames, tagValues, globRes, Res, dtime, nsync, channel]

# Basic Lee Filter
def LeeFilter(I, windowSize = 5):
	movingMean = uniform_filter(I, windowSize)
	movingSqrMean = uniform_filter(I**2, windowSize)
	movingVar = movingSqrMean - movingMean**2
	var = variance(I)
	return movingMean + (I - movingMean) * movingVar / (movingVar + var)

# Kernel Density Estimator with 5-tau cutoff
def KDE(t0, t, tau):
	dt = np.abs(t - t0)
	ee = np.exp( - dt / tau)
	return sum(ee[dt <= 5 * tau])

# Non-biased KDE with 5-tau cutoff
def nbKDE(t0, t, tau):
	dt = np.abs(t - t0)
	ee = np.exp( - dt / tau)
	return (1 + 2/len(t)) * (sum(ee[dt <= 5 * tau]) - 1)

# Calculate florescent decay time in ns within the given window
def getDecayTime(file, selectedChannel, tBound):

	[tagNames, tagValues, globRes, Res, dtime, nsync, channel] = LoadPTU(file, chatty = False)
	tBins = np.arange(int(globRes / Res))
	thist, tBins = np.histogram(dtime[channel == selectedChannel], tBins)
	tBins = tBins[1:] - 0.5

	backGround = np.mean(thist[ - 500 :  - 100]) # subtracting the background away from IRF, this is why we need to calculate the histogram
	thist = thist - backGround

	mask = (tBins > tBound[0] / Res / 1e9) * (tBins < tBound[1] / Res / 1e9)
	t = sum(thist[mask] * tBins[mask]) / sum(thist[mask])
	return t * Res * 1e9

# Get basic photon dataframe with DD, AD and AA labelled. t in ns unit and T in s unit
def getPhotonData(file, tBounds = None):
	# Reading ptu file
	[tagNames, tagValues, globRes, Res, dtime, nsync, channel] = LoadPTU(file, chatty = False)

	# Data
	df = pd.DataFrame({
		't' : dtime + 0.5,
		'T' : (nsync + 0.5) * globRes + (dtime + 0.5) * Res,
		'channel' : channel,
		'Res': Res,
		'globRes': globRes
		})

	df.Res = Res # pass the dtime resolution as an attribute
	df.globRes = globRes # pass the dtime resolution as an attribute

	if tBounds is None:
		return df
	else:
		# Write to df re. D/A information
		df['DD'] = False
		df['AD'] = False
		df['AA'] = False
		tD = tBounds[0]
		tA = tBounds[1]
		df['DD'] = (df['t'].values * Res * 1e9 > tD[0]) & (df['t'].values * Res * 1e9 < tD[1]) & (df['channel'] == 0)
		df['AD'] = (df['t'].values * Res * 1e9 > tD[0]) & (df['t'].values * Res * 1e9 < tD[1]) & (df['channel'] == 1)
		df['AA'] = (df['t'].values * Res * 1e9 > tA[0]) & (df['t'].values * Res * 1e9 < tA[1]) & (df['channel'] == 1)
		
		return df

# Burst search, writes burst and noise index into DataFrame
# ** see visualiseIRF for an explanation of how burst lengths calculated
def findBurst(photonData, params, debug=True):

	# Calculate raw interphoton times with Lee filter
	interPhotonTime = photonData['T'].values[1:] - photonData['T'].values[:-1]
	interPhotonTime = LeeFilter(interPhotonTime)
	
	burstLoc = np.argwhere(interPhotonTime < params['burstCut'])[:, 0]
	burstLoc = np.append(burstLoc, burstLoc[-1])
	
	noiseLoc = np.argwhere(interPhotonTime > params['noiseCut'])[:, 0]
	noiseLoc = np.append(noiseLoc, noiseLoc[-1])
	
	burstIdx = np.insert(np.where(burstLoc[1:] - burstLoc[:-1] > 1)[0] + 1, 0, 0)
	noiseIdx = np.insert(np.where(noiseLoc[1:] - noiseLoc[:-1] > 1)[0] + 1, 0, 0)
	
	burstLen = np.zeros(len(burstIdx)).astype(np.int)
	for i in range(len(burstIdx)):
		_length = 1
		_idx = burstIdx[i]
		while burstLoc[_idx+1] == burstLoc[_idx] + 1:
			_length += 1
			_idx += 1
		burstLen[i] = _length
		
	noiseLen = np.zeros(len(noiseIdx)).astype(np.int)
	for i in range(len(noiseIdx)):
		_length = 1
		_idx = noiseIdx[i]
		while noiseLoc[_idx+1] == noiseLoc[_idx] + 1:
			_length += 1
			_idx += 1
		noiseLen[i] = _length
		
	burstMask = burstLen > params['burstLen']
	trueBurstIdx = burstIdx[burstMask]
	trueBurstLen = burstLen[burstMask]
	
	noiseMask = noiseLen > params['noiseLen']
	trueNoiseIdx = noiseIdx[noiseMask]
	trueNoiseLen = noiseLen[noiseMask]
	
	# Write to df re. burst and noise state
	photonData['burst'] = int(-1)
	photonData['noise'] = int(-1)
	
	for i in range(len(trueBurstIdx)):
		_i = int(burstLoc[trueBurstIdx[i]])
		_j = int(burstLoc[trueBurstIdx[i] + trueBurstLen[i] - 1] + 1)
		photonData.loc[_i: _j, 'burst'] = np.ones(len(photonData.loc[_i : _j, 'burst']), dtype = int) * i
	
	for i in range(len(trueNoiseIdx)):
		_i = int(noiseLoc[trueNoiseIdx[i]])
		_j = int(noiseLoc[trueNoiseIdx[i] + trueNoiseLen[i] - 1] + 1)
		photonData.loc[_i: _j, 'noise'] = np.ones(len(photonData.loc[_i : _j, 'noise']), dtype = int) * i
		
	if debug:
		print('Burst count: {0:}\nNoise count: {1:}\n'.format(len(trueBurstIdx), len(trueNoiseIdx)))

	return photonData

# Burst analysis, outputting df with S and E values and applies bleaching filter
# Retains Noise info to check when noise is greater than signal
def JE_burstAnalysisSE_WithRaw(photonData, burstData = pd.DataFrame({}), debug = False):

	burstNumber = int(max(photonData['burst'].values) + 1)
	noiseNumber = int(max(photonData['noise'].values) + 1)

	# Calculate background signal in units of photons per second. Noise rates should be around 1 kHz
	noiseDD = 0
	noiseAD = 0
	noiseAA = 0
	noiseT = 0
	
	for i in range(noiseNumber):
		_df = photonData[photonData['noise'] == i]
		noiseT += (_df['T'].values[-1] - _df['T'].values[0])
		noiseDD += len(_df[_df['DD']]) - 1
		noiseAD += len(_df[_df['AD']]) - 1
		noiseAA += len(_df[_df['AA']]) - 1
	
	if noiseT != 0:
		noiseDD /= noiseT
		noiseAD /= noiseT
		noiseAA /= noiseT

	if debug:
		print('Noise rates:\nDD : {0:.2g} kHz\nAD : {1:.2g} kHz\nAA : {2:.2g} kHz\n'.format(noiseDD/1000, noiseAD/1000, noiseAA/1000))

	# Calculate photon counts per burst and burst duration

	# Raw Channels
	rawDD = np.zeros(burstNumber)
	rawAD = np.zeros(burstNumber)
	rawAA = np.zeros(burstNumber)

	# Estimated Noise during Burst
	bgDD = np.zeros(burstNumber)
	bgAD = np.zeros(burstNumber)
	bgAA = np.zeros(burstNumber)

	# Corrected Channels
	numDD = np.zeros(burstNumber)
	numAD = np.zeros(burstNumber)
	numAA = np.zeros(burstNumber)
	burstDuration = np.zeros(burstNumber)

	TDA = np.zeros(burstNumber)

	E = np.zeros(burstNumber)
	S = np.zeros(burstNumber)
	
	for i in range(burstNumber):
		_df = photonData[photonData.burst == i]
		burstDuration[i] = _df['T'].values[-1] - _df['T'].values[0]

		# DD (minus 1 has been removed from original func)
		rawDD[i] = len(_df[_df['DD']]) 
		bgDD[i] = noiseDD * burstDuration[i]
		numDD[i] = rawDD[i] - bgDD[i] 

		# AD (minus 1 has been removed from original func)
		rawAD[i] = len(_df[_df['AD']])
		bgAD[i] = noiseAD * burstDuration[i]
		numAD[i] = rawAD[i] - bgAD[i]

		# AA (minus 1 has been removed from original func)
		rawAA[i] = len(_df[_df['AA']]) 
		bgAA[i] = noiseAA * burstDuration[i]
		numAA[i] = rawAA[i] - bgAA[i]

		DPhotons = _df.loc[((_df['channel'] == 0) | (_df['AD'])), 'T']
		APhotons = _df.loc[_df['AA'], 'T']
		if (len(DPhotons) == 0) | (len(APhotons) == 0):
			TDA[i] = 0
		else:
			TDA[i] = DPhotons.mean() - APhotons.mean()

	E = (numAD) / (numDD + numAD)
	S = (numAD + numDD) / (numDD + numAD + numAA)
	

	burstData['rawDD'] = rawDD
	burstData['noiseDD'] = bgDD
	burstData['DD'] = numDD

	burstData['rawAD'] = rawAD
	burstData['noiseAD'] = bgAD
	burstData['AD'] = numAD
	
	burstData['rawAA'] = rawAA
	burstData['noiseAA'] = bgAA
	burstData['AA'] = numAA
	
	burstData['E'] = E
	burstData['S'] = S
	burstData['TDA'] = TDA
	burstData['duration'] = burstDuration

	return photonData, burstData

def JE_multiple_photon_files(file_list, tBounds, params, tauSettings = None, debug=False):
	"""
	Loop through all photon data
	Note: The total data set should first be combined then the burst analysis done, this would require updating macroscopic times
	"""

	all_burst_data = []

	for this_sample in file_list:

		# Get Photon Data
		photon_data = getPhotonData(file=this_sample, tBounds=tBounds)

		# Find and Label Bursts
		burst_labeled_photon_data = findBurst(photonData = photon_data, params=params, debug=debug)

		# Analyse Bursts
		burst_labeled_photon_data, burst_data = JE_burstAnalysisSE_WithRaw(photonData=burst_labeled_photon_data, burstData = pd.DataFrame({}))

		# ALEX FRET 2 CDE Calculations for corrections
		burst_labeled_photon_data, burst_data = burstAnalysis2CDE(photonData = burst_labeled_photon_data, burstData = burst_data)
		
		# tau calculations
		if tauSettings is not None:
			burst_labeled_photon_data, burst_data = burstAnalysisTau(photonData = burst_labeled_photon_data, tauSettings = tauSettings, burstData = burst_data)
		
		if debug:
			print(f'Adding: {this_sample}')
			print(f'photo data length: {len(photon_data)}')
			print(f'burst labelled photon data length: {len(burst_labeled_photon_data)}')
			print(f'ALEX 2 CDE: {len(burst_data)}')	
		
		all_burst_data.append(burst_data)
	return pd.concat(all_burst_data, ignore_index=True)

def JE_unpack_burst_values(burstData):
	return burstData['rawDD'], burstData['rawAD'], burstData['rawAA'], burstData['DD'], burstData['AD'], burstData['AA'], burstData['E'], burstData['S'], burstData['ALEX2CDE'], burstData['FRET2CDE'], burstData['duration']


# Burst analysis, outputting df with S and E values and applies bleaching filter
def burstAnalysisSE(photonData, burstData = pd.DataFrame({})):

	burstNumber = int(max(photonData['burst'].values) + 1)
	noiseNumber = int(max(photonData['noise'].values) + 1)

	# Calculate background signal in units of photons per second. Noise rates should be around 1 kHz
	noiseDD = 0
	noiseAD = 0
	noiseAA = 0
	noiseT = 0
	
	for i in range(noiseNumber):
		_df = photonData[photonData['noise'] == i]
		noiseT += (_df['T'].values[-1] - _df['T'].values[0])
		noiseDD += len(_df[_df['DD']]) - 1
		noiseAD += len(_df[_df['AD']]) - 1
		noiseAA += len(_df[_df['AA']]) - 1
	
	if noiseT != 0:
		noiseDD /= noiseT
		noiseAD /= noiseT
		noiseAA /= noiseT

	print('Noise rates:\nDD : {0:.2g} kHz\nAD : {1:.2g} kHz\nAA : {2:.2g} kHz\n'.format(noiseDD/1000, noiseAD/1000, noiseAA/1000))

	# Calculate photon counts per burst and burst duration

	numDD = np.zeros(burstNumber)
	numAD = np.zeros(burstNumber)
	numAA = np.zeros(burstNumber)
	burstDuration = np.zeros(burstNumber)

	TDA = np.zeros(burstNumber)

	E = np.zeros(burstNumber)
	S = np.zeros(burstNumber)
	
	for i in range(burstNumber):
		_df = photonData[photonData.burst == i]
		burstDuration[i] = _df['T'].values[-1] - _df['T'].values[0]
		numDD[i] = len(_df[_df['DD']]) - 1 - noiseDD * burstDuration[i]
		numAD[i] = len(_df[_df['AD']]) - 1 - noiseAD * burstDuration[i]
		numAA[i] = len(_df[_df['AA']]) - 1 - noiseAA * burstDuration[i]

		DPhotons = _df.loc[((_df['channel'] == 0) | (_df['AD'])), 'T']
		APhotons = _df.loc[_df['AA'], 'T']
		if (len(DPhotons) == 0) | (len(APhotons) == 0):
			TDA[i] = 0
		else:
			TDA[i] = DPhotons.mean() - APhotons.mean()

	E = (numAD) / (numDD + numAD)
	S = (numAD + numDD) / (numDD + numAD + numAA)
	
	burstData['DD'] = numDD
	burstData['AD'] = numAD
	burstData['AA'] = numAA
	burstData['E'] = E
	burstData['S'] = S
	burstData['TDA'] = TDA
	burstData['duration'] = burstDuration

	return photonData, burstData

# Burst search, calculates ALEX2CDE and FRET2CDE values
def burstAnalysis2CDE(photonData, burstData = pd.DataFrame({})):

	burstNumber = int(max(photonData['burst'].values) + 1)

	ALEX2CDE = np.zeros(burstNumber)
	FRET2CDE = np.zeros(burstNumber)
	
	for i in range(burstNumber):

		_df = photonData[photonData.burst == i]

		# Calculate ALEX2CDE and FRET2CDE values
		try:

			_trueAA = _df.loc[_df.AA, 'T'].values
			_trueAD = _df.loc[_df.AD, 'T'].values
			_trueDD = _df.loc[_df.DD, 'T'].values
			_trueDDAD = _df.loc[_df.AD | _df.DD, 'T'].values
			
			# FRET2CDE calculation
			
			_tauFRET = 12.5 * 1e-6
			
			_ED = np.nan
			_1mEA = np.nan
			
			_frac1 = []
			for j in range(len(_trueDD)):
				_KdeDA = KDE(_trueDD[j], _trueAD, _tauFRET)
				_nbKdeDD = nbKDE(_trueDD[j], _trueDD, _tauFRET)
				if (_KdeDA + _nbKdeDD) != 0:
					_frac1.append(_KdeDA/(_KdeDA + _nbKdeDD))
			if len(_frac1) > 0 :
				_ED = sum(_frac1)/len(_frac1)
			
			_frac2 = []
			for j in range(len(_trueAD)):
				_KdeAD = KDE(_trueAD[j], _trueDD, _tauFRET)
				_nbKdeAA = nbKDE(_trueAD[j], _trueAD, _tauFRET)
				if (_KdeAD + _nbKdeAA) != 0:
					_frac2.append(_KdeAD/(_KdeAD + _nbKdeAA))
			if len(_frac2) > 0:
				_1mEA = (1/len(_frac2))*sum(_frac2)
			
			_val = 110 - 100 * (_ED + _1mEA)
			if (np.isnan(_val) | (_val < 0)):
				FRET2CDE[i] = 0
			else:
				FRET2CDE[i] = _val
			
			# ALEX2CDE calculation
			
			_tauALEX = 75 * 1e-6
			
			_KdeDA = np.zeros(len(_trueDDAD))
			_KdeDD = np.zeros(len(_trueDDAD))
			for j in range(len(_trueDDAD)):
				_KdeDA[j] = KDE(_trueDDAD[j], _trueAA, _tauALEX)
				_KdeDD[j] = KDE(_trueDDAD[j], _trueDDAD, _tauALEX)
			if len(_trueAA) == 0:
				_BrD = 0
			else:
				_BrD = sum(_KdeDA / _KdeDD)/len(_trueAA)
			
			_KdeAD = np.zeros(len(_trueAA))
			_KdeAA = np.zeros(len(_trueAA))
			for j in range(len(_trueAA)):
				_KdeAA[j] = KDE(_trueAA[j], _trueAA, _tauALEX)
				_KdeAD[j] = KDE(_trueAA[j], _trueDDAD, _tauALEX)
				
			if len(_trueDDAD) == 0:
				_BrA = 0
			else:
				_BrA = sum(_KdeAD / _KdeAA)/len(_trueDDAD)
			
			ALEX2CDE[i] = 100 - 50 * (_BrA + _BrD)
			
		except ZeroDivisionError:
			pass

	burstData['ALEX2CDE'] = ALEX2CDE
	burstData['FRET2CDE'] = FRET2CDE

	return photonData, burstData

# Burst search for lifetime
def burstAnalysisTau(photonData, tauSettings, burstData = pd.DataFrame({})):

	burstNumber = int(max(photonData['burst'].values) + 1)
	tau = np.ones((2, burstNumber)) * (-1)
	Res = photonData.Res[0]
	
	# 0 mode: compute average t for IRF and signal, and subtract
	if tauSettings['mode'] == 0:

		IRFPaths = tauSettings['IRFPaths']
		tBounds = tauSettings['tBounds']
		XX = ['DD', 'AA']

		decayTimes = []
		for i in range(2):
			decayTimes.append(getDecayTime(IRFPaths[i], i, tBounds[i]))

		for i in np.arange(burstNumber):
			_df = photonData[photonData['burst'] == i]
			for j in range(2):
				tau[j][i] = _df.loc[_df[XX[j]], 't'].mean() * Res * 1e9 - decayTimes[j]

	# 1 mode: focus on the decay branch and estimate lifetime in the region of interest
	elif tauSettings['mode'] == 1:

		tInterest = [[3.4, 10], [23.4, 30]]

		for i in np.arange(burstNumber):
			_df = photonData[photonData['burst'] == i]

			for j in range(2):
				t1 = tInterest[j][0]  / Res / 1e9
				t2 = tInterest[j][1]  / Res / 1e9
				DT = t2 - t1
				_t = _df.loc[(photonData['channel'] == j) & (photonData['t'] > t1) & (photonData['t'] < t2), 't']
				if _t.empty:
					pass
				else:
					_eta = (_t.values - t1)/DT
					_F = np.mean(_eta * (1 - _eta))
					_Fp = np.mean(1 - 2 * _eta)
					_tau = DT * _F / _Fp
					tau[j][i] = _tau * Res * 1e9

	# 2 mode: estimate photon noise, outdated
	elif tauSettings['mode'] == 2:

		tInterest = tauSettings['tInterest']

		noiseNumber = int(max(photonData['noise'].values) + 1)

		# Calculate background signal in units of photons per second. Noise rates should be around 1 kHz
		noise = np.zeros(2)
		noiseT = 0
	
		for i in range(noiseNumber):
			_df = photonData[photonData['noise'] == i]
			noiseT += (_df['T'].values[-1] - _df['T'].values[0])
			for j in range(2):
				t1 = tInterest[j][0]  / Res / 1e9
				t2 = tInterest[j][1]  / Res / 1e9
				noise[j] += len(_df.loc[(photonData['channel'] == j) & (photonData['t'] > t1) & (photonData['t'] < t2), 't'].values) - 1
		
		if noiseT != 0:
			noise /= noiseT

		for i in np.arange(burstNumber):
			_df = photonData[photonData['burst'] == i]

			for j in range(2):

				t1 = tInterest[j][0]  / Res / 1e9
				t2 = tInterest[j][1]  / Res / 1e9
				_t = _df.loc[(photonData['channel'] == j) & (photonData['t'] > t1) & (photonData['t'] < t2), 't']
				if _t.empty:
					pass
				else:

					nNoise = noise[j] * (_df['T'].values[-1] - _df['T'].values[0])

					_t = _t.values - t1
					tmax = t2 - t1

					sumT1 = np.sum(_t ** 1)
					sumT2 = np.sum(_t ** 2)
					sumT3 = np.sum(_t ** 3)

					delT1 = sumT1 - tmax ** 1 * nNoise / 2
					delT2 = sumT2 - tmax ** 2 * nNoise / 3
					delT3 = sumT3 - tmax ** 3 * nNoise / 4

					_tau = (tmax * delT2 - delT3) / (2 * tmax * delT1 - 3 * delT2)
					tau[j][i] = _tau * Res * 1e9

	burstData['tauD'] = tau[0]
	burstData['tauA'] = tau[1]

	return photonData, burstData

# Burst search for lifetime
def burstAnalysisTauDouble(photonData, tauSettings, burstData = pd.DataFrame({})):

	burstNumber = int(max(photonData['burst'].values) + 1)
	tauP = np.ones(burstNumber) * (-1)
	tauM = np.ones(burstNumber) * (-1)
	g = np.ones(burstNumber) * (-1)
	Res = photonData.Res
	
	
	if tauSettings['mode'] == 0:

		tInterest = tauSettings['tInterest']

		for i in np.arange(burstNumber):
			_df = photonData[photonData['burst'] == i]

			t1 = tInterest[0][0]  / Res / 1e9
			t2 = tInterest[0][1]  / Res / 1e9
			DT = t2 - t1
			_t = _df.loc[(photonData['channel'] == 0) & (photonData['t'] > t1) & (photonData['t'] < t2), 't']
			if _t.empty:
				pass
			else:
				t = (_t.values - t1)

				H0 = np.mean(DT**3 * t**3 - 3 * DT**2 * t**4 + 3 * DT * t**5 - t**6)
				H1 = np.mean(3 * DT**3 * t**2 - 12 * DT**2 * t**3 + 15 * DT * t**4 - 6*t**5)
				H2 = np.mean(6 * DT**3 * t**1 - 36 * DT**2 * t**2 + 60 * DT * t**3 - 30*t**4)
				H3 = np.mean(6 * DT**3 - 72 * DT**2 * t + 180 * DT * t**2 - 120*t**3)

				D0 = H0 * H3 - H1 * H2
				D1 = H1 * H3 - H2 * H2
				D2 = H0 * H2 - H1 * H1
				T0 = D0/D1
				Q = D0**2/D1**2-4*D2/D1
				
				LEOT1 = (T0 + np.sqrt(Q)) / 2
				LEOT2 = (T0 - np.sqrt(Q)) / 2
				
				Hp = (LEOT1 * H0 - LEOT2 * LEOT1 * H1) / (LEOT1 - LEOT2)
				Hm = (LEOT2 * H0 - LEOT1 * LEOT2 * H1) / (LEOT2 - LEOT1)
				
				intH1 = 6 * np.exp(-(DT/LEOT1)) * LEOT1 ** 4 * (DT**3 + 12* DT**2 *LEOT1 + 60* DT* LEOT1**2 + 120* LEOT1**3 + np.exp(DT/LEOT1) * (DT**3 - 12 *DT**2 *LEOT1 + 60 *DT *LEOT1**2 - 120* LEOT1**3))
				intH2 = 6 * np.exp(-(DT/LEOT2)) * LEOT2 ** 4 * (DT**3 + 12* DT**2 *LEOT2 + 60* DT* LEOT2**2 + 120* LEOT2**3 + np.exp(DT/LEOT2) * (DT**3 - 12 *DT**2 *LEOT2 + 60 *DT *LEOT2**2 - 120* LEOT2**3))
				
				Ap = Hp / intH1
				Am = Hm / intH2
				
				ApInt = Ap * LEOT1 * (1 - np.exp(-DT/LEOT1))
				AmInt = Am * LEOT2 * (1 - np.exp(-DT/LEOT2))
				
				gamma = ApInt / (ApInt + AmInt)
				tau1 = LEOT1
				tau2 = LEOT2
				tauP[i] = tau1 * Res * 1e9
				tauM[i] = tau2 * Res * 1e9
				g[i] = gamma


	burstData['tauDP'] = tauP
	burstData['tauDM'] = tauM

	burstData['gammaD'] = g

	return photonData, burstData

# Old burst search for lifetime
def burstAnalysisTauOld(photonData, burstData = pd.DataFrame({}), IRFPaths = None, tBounds = None, useMLE = 0):

	burstNumber = int(max(photonData['burst'].values) + 1)

	tauD = np.zeros(burstNumber)
	tauA = np.zeros(burstNumber)
	
	Res = photonData.Res
	
	if useMLE == -1:

		IRFRates = []
		IRFDecayTimes = []

		for i in range(2):
			[_tagNames, _tagValues, _globRes, _Res, _dtime, _nsync, _channel] = LoadPTU(IRFPaths[i], chatty = False)
			tBins = np.arange(int(_globRes / _Res))
			IRFHist, _tBins = np.histogram(_dtime + 0.5, tBins)
			backGround = np.mean(IRFHist[ - 500 :  - 100])
			IRFHist = IRFHist - backGround

			DataHist, _tBins = np.histogram(photonData.loc[photonData['channel'] == i, 't'], tBins)
			backGround = np.mean(DataHist[ - 500 :  - 100])
			DataHist = DataHist - backGround

			tBins = tBins[:-1] + 0.5
			
			maxDataHist = np.max(DataHist)
			maxIRFHist = np.max(IRFHist)
			
			IRFHist = IRFHist * maxDataHist / maxIRFHist
			
			tMask = (tBins * Res * 1e9 > tBounds[i][0]) & (tBins * Res * 1e9 < tBounds[i][1])
			IRFHist = IRFHist[tMask]
			DataHist = DataHist[tMask]
			tBins = tBins[tMask]

			maxDataHist = np.mean(DataHist[argmax(DataHist) - 100 : argmax(DataHist) + 100])
			maxIRFHist = np.mean(IRFHist[argmax(IRFHist) - 100 : argmax(IRFHist) + 100])
			IRFHist = IRFHist * maxDataHist / maxIRFHist

			IRFPhotons = sum(IRFHist)
			IRFRate = IRFPhotons/(photonData['T'].values[-1] - photonData['T'].values[0])
			IRFRates.append(IRFRate)
			IRFDecayTimes.append(sum(tBins * IRFHist)/sum(IRFHist))
		
		for i in np.arange(len(trueBurstIdx))[filterAll]:
			
			numDDReflection = IRFRates[0] * burstDuration[i]
			numAAReflection = IRFRates[1] * burstDuration[i]
	
			_df = photonData[photonData.burst == i]
			DDTotal = len(_df[_df['DD']])
			AATotal = len(_df[_df['AA']])

			if DDTotal > numDDReflection:
				gammaDD = 1 - numDDReflection / DDTotal
				print(gammaDD)
				tauD[i] = (_df.loc[_df['DD'], 't'].mean() - IRFDecayTimes[0])/ gammaDD * Res * 1e9
			if AATotal > numAAReflection:
				gammaAA = 1 - numAAReflection / AATotal
				tauA[i] = (_df.loc[_df['AA'], 't'].mean() - IRFDecayTimes[1])/ gammaAA * Res * 1e9
	
	elif useMLE == 1:

		tD = tBounds[0]
		tA = tBounds[1]

		[_tagNames, _tagValues, _globRes, _Res, _dtime, _nsync, _channel] = LoadPTU(IRFPaths[0], chatty = False)
		tBins = np.arange(int(_globRes / _Res)) # machine unit
		_hist, _bins = np.histogram(_dtime[_channel == 0] + 0.5, bins = tBins)
		lastPositiveIndex = len(_hist) - np.where(_hist[::-1] > 0)[0][0]
		backGround = np.mean(_hist[lastPositiveIndex - 500 : lastPositiveIndex - 100])
		_Dhist = _hist - backGround

		[_tagNames, _tagValues, _globRes, _Res, _dtime, _nsync, _channel] = LoadPTU(IRFPaths[1], chatty = False)
		_hist, _bins = np.histogram(_dtime[_channel == 1] + 0.5, bins = tBins)
		lastPositiveIndex = len(_hist) - np.where(_hist[::-1] > 0)[0][0]
		backGround = np.mean(_hist[lastPositiveIndex - 500 : lastPositiveIndex - 100])
		_Ahist = _hist - backGround

		tBins = tBins[:-1] + 0.5
		tBinMasks = []
		tBinMasks.append((tBins * _Res * 1e9 > tD[0]) & (tBins * _Res * 1e9 < tD[1]))
		tBinMasks.append((tBins * _Res * 1e9 > tA[0]) & (tBins * _Res * 1e9 < tA[1]))

		IRFHist = []
		IRFHist.append(_Dhist[tBinMasks[0]])
		IRFHist.append(_Ahist[tBinMasks[1]])

		def II(p, dHist, iHist):
			[tau, r] = p
			r = 1
			
			iHist = iHist / sum(iHist)

			expTau = np.exp(-(np.arange(len(iHist)) + 0.5)/tau)/tau
			expTau = fftconvolve(expTau, iHist)[0:len(iHist)]
			expTau = r * expTau + (1 - r) * iHist
			
			N = sum(dHist)
			expTau = np.abs(N * expTau / sum(expTau))
			nonZMask = dHist > 0
			result = ((2 / (len(iHist) - 2)) * np.sum(dHist[nonZMask] * np.log(dHist[nonZMask] / np.abs(expTau[nonZMask]))))
			return(1000 * result)

		for i in np.arange(len(trueBurstIdx))[filterAll]:

			# DD decay time
			_df = photonData[(photonData['burst'] == i) & photonData['DD']]
			_hist, _bins = np.histogram(_df.t.values, bins = np.arange(int(_globRes / _Res)))
			dHist = _hist[tBinMasks[0]]
			dHist[tBins[tBinMasks[0]] * _Res * 1e9 < 5] = 0
			if sum(dHist) > 0:
				res = minimize(II, x0 = (1000, 1), args = (dHist, IRFHist[0]), bounds = ((10, 4000), (0.5, 1) ))
				if res.x[0] == 1000:
					pass
				else:
					tauD[i] = res.x[0] * Res * 1e9

			# AA decay time
			_df = photonData[(photonData['burst'] == i) & photonData['AA']]
			_hist, _bins = np.histogram(_df.t.values, bins = np.arange(int(_globRes / _Res)))
			dHist = _hist[tBinMasks[1]]
			if sum(dHist) > 0:
				res = minimize(II, x0 = (1000, 0.5), args = (dHist, IRFHist[1]), bounds = ((10, 4000), (0.5, 1) ))
				if res.x[0] == 1000:
					pass
				else:
					tauA[i] = res.x[0] * Res * 1e9

	# MLE without using scipy.optimize.minimize
	elif useMLE == 2:

		tD = tBounds[0]
		tA = tBounds[1]

		[_tagNames, _tagValues, _globRes, _Res, _dtime, _nsync, _channel] = LoadPTU(IRFPaths[0], chatty = False)
		tBins = np.arange(int(_globRes / _Res)) # machine unit

		_hist, _bins = np.histogram(_dtime[_channel == 0] + 0.5, bins = tBins)
		lastPositiveIndex = len(_hist) - np.where(_hist[::-1] > 0)[0][0]
		backGround = np.mean(_hist[lastPositiveIndex - 500 : lastPositiveIndex - 100])
		_Dhist = _hist - backGround

		[_tagNames, _tagValues, _globRes, _Res, _dtime, _nsync, _channel] = LoadPTU(IRFPaths[1], chatty = False)
		_hist, _bins = np.histogram(_dtime[_channel == 1] + 0.5, bins = tBins)
		lastPositiveIndex = len(_hist) - np.where(_hist[::-1] > 0)[0][0]
		backGround = np.mean(_hist[lastPositiveIndex - 500 : lastPositiveIndex - 100])
		_Ahist = _hist - backGround

		tBins = tBins[:-1] + 0.5
		tBinMasks = []
		tBinMasks.append((tBins * _Res * 1e9 > tD[0]) & (tBins * _Res * 1e9 < tD[1]))
		tBinMasks.append((tBins * _Res * 1e9 > tA[0]) & (tBins * _Res * 1e9 < tA[1]))

		IRFHist = []
		_Dhist = _Dhist[tBinMasks[0]]
		_Ahist = _Ahist[tBinMasks[1]]

		_Dhist = _Dhist / sum(_Dhist)
		_Ahist = _Ahist / sum(_Ahist)

		IRFHist.append(np.abs(_Dhist))
		IRFHist.append(np.abs(_Ahist))

		# Get background noise from noise 'bursts'
		tMin = 3.0
		limitedNoiseDD = 0
		limitedNoiseAA = 0
		limitedNoiseT = 0
	
		for i in range(len(trueNoiseIdx)):
			_df = photonData[photonData['noise'] == i]
			limitedNoiseT += (_df['T'].values[-1] - _df['T'].values[0])
			_dfDD = _df[(_df['t'].values * Res * 1e9 > tMin) & (_df['t'].values * Res * 1e9 < tD[1]) & (_df['channel'] == 0)]
			_dfAA = _df[(_df['t'].values * Res * 1e9 > tA[0]) & (_df['t'].values * Res * 1e9 < tA[1]) & (_df['channel'] == 1)]
			limitedNoiseDD += len(_dfDD) - 1
			limitedNoiseAA += len(_dfAA) - 1
	
		if limitedNoiseT != 0:
			limitedNoiseDD /= limitedNoiseT
			limitedNoiseAA /= limitedNoiseT


		guessTau = np.linspace(1, 5000, 50)
		dBins = []
		expTau = []
		for i in range(2):
			dBins.append(tBins[tBinMasks[i]])
			_TTau, _TT = np.meshgrid(guessTau, np.arange(len(_Dhist)) + 0.5)
			_expTau = np.exp(-_TT/_TTau)/_TTau
			_expIRF = np.vstack([IRFHist[i]] * len(guessTau)).T # [time, tau]. IRF, the tau dimension is dummy
			expTau.append(fftconvolve(_expTau, _expIRF, axes = 0)) # Convolved guess
		
		for i in np.arange(len(trueBurstIdx))[filterAll]:

			# DD decay time
			_df = photonData[(photonData['burst'] == i) & photonData['DD']]
			_hist, _bins = np.histogram(_df.t.values, bins = np.arange(int(_globRes / _Res)))
			dHist = _hist[tBinMasks[0]]
			dHist[dBins[0] * Res * 1e9 < tMin] = 0

			N = sum(dHist) # total number of photons in the window
			NDDNoise = limitedNoiseDD * burstDuration[i]

			if N > NDDNoise:

				II = []
				for j in range(len(guessTau)):
					_expTau = expTau[0][0:len(dBins[0]), j]
					_expTau = (N - NDDNoise) * _expTau / sum(_expTau) + NDDNoise / len(_expTau)
					_nonZMask = dHist > 0
					_Mask = _nonZMask
					_II = (2 / (len(dBins[0]) - 2)) * np.sum(dHist[_Mask] * np.log(dHist[_Mask] / _expTau[_Mask]))
					II.append(_II)
				ii = np.argmin(II)

				guessTau2 = np.linspace(guessTau[max(0, ii-2)], guessTau[min(len(guessTau) - 1, ii+2)], 50)
				dBins2 = []
				dBins2 = tBins[tBinMasks[0]]
				_TTau, _TT = np.meshgrid(guessTau2, np.arange(len(_Dhist)) + 0.5)
				expTau2 = np.exp(-_TT/_TTau)/_TTau
				_expIRF = np.vstack([IRFHist[0]] * len(guessTau2)).T # [time, tau]. IRF, the tau dimension is dummy
				expTau2 = fftconvolve(expTau2, _expIRF, axes = 0) # Convolved guess

				III = []
				for j in range(len(guessTau2)):
					_expTau = expTau2[0:len(dBins[0]), j]
					_expTau = (N - NDDNoise) * _expTau / sum(_expTau) + NDDNoise / len(_expTau)
					_nonZMask = dHist > 0
					_Mask = _nonZMask
					_III = (2 / (len(dBins[0]) - 2)) * np.sum(dHist[_Mask] * np.log(dHist[_Mask] / _expTau[_Mask]))
					III.append(_III)
				iii = np.argmin(III)

				tauD[i] = guessTau2[iii] * Res * 1e9

			# AA decay time
			_df = photonData[(photonData.burst == i) & photonData['AA']]
			_hist, _bins = np.histogram(_df.t.values, bins = np.arange(int(_globRes / _Res)))
			dHist = _hist[tBinMasks[1]]

			N = sum(dHist) # total number of photons in the window
			NAANoise = limitedNoiseAA * burstDuration[i]

			if N > NAANoise:
				II = []
				for j in range(len(guessTau)):
					_expTau = expTau[1][0:len(dBins[1]), j]
					_expTau = (N - NAANoise) * _expTau / sum(_expTau) + NAANoise / len(_expTau)
					_nonZMask = dHist > 0
					_II = (2 / (len(dBins[1]) - 2)) * np.sum(dHist[_nonZMask] * np.log(dHist[_nonZMask] / _expTau[_nonZMask]))
					II.append(_II)
				ii = np.argmin(II)
				
				guessTau2 = np.linspace(guessTau[max(0, ii-2)], guessTau[min(len(guessTau) - 1, ii+2)], 50)
				dBins2 = []
				dBins2 = tBins[tBinMasks[1]]
				_TTau, _TT = np.meshgrid(guessTau2, np.arange(len(_Ahist)) + 0.5)
				expTau2 = np.exp(-_TT/_TTau)/_TTau
				_expIRF = np.vstack([IRFHist[1]] * len(guessTau2)).T # [time, tau]. IRF, the tau dimension is dummy
				expTau2 = fftconvolve(expTau2, _expIRF, axes = 0) # Convolved guess

				III = []
				for j in range(len(guessTau2)):
					_expTau = expTau2[0:len(dBins[1]), j]
					_expTau = (N - NAANoise) * _expTau / sum(_expTau) + NAANoise / len(_expTau)
					_nonZMask = dHist > 0
					_Mask = _nonZMask
					_III = (2 / (len(dBins[1]) - 2)) * np.sum(dHist[_Mask] * np.log(dHist[_Mask] / _expTau[_Mask]))
					III.append(_III)
				iii = np.argmin(III)

				tauA[i] = guessTau2[iii] * Res * 1e9
		
	else:
		# Decay times in ns unit from IRF
		decayTimes = []
		for i in range(2):
			decayTimes.append(getDecayTime(IRFPaths[i], i, tBounds[i]))
		for i in np.arange(len(trueBurstIdx))[filterAll]:
			# Calculate decay timee
			_df = photonData[photonData.burst == i]
			tauD[i] = _df.loc[_df['DD'], 't'].mean() * Res * 1e9 - decayTimes[0]
			tauA[i] = _df.loc[_df['AA'], 't'].mean() * Res * 1e9 - decayTimes[1]
	
	burstData['tauD'] = tauD
	burstData['tauA'] = tauA

	return photonData, burstData

# Visualise preliminary data processing
def visualiseIRF(file, IRFfiles, tBounds):

	colors = ['blue', 'red']
	tD = tBounds[0]
	tA = tBounds[1]

	# Initialise figure
	fig, axes = plt.subplots(2, 1, figsize = (6,4), sharex = True)
	axes = axes.reshape(-1)

	# Reading ptu file
	[tagNames, tagValues, globRes, Res, dtime, nsync, channel] = LoadPTU(file, chatty = False)

	tBins = np.arange(int(globRes / Res))
	tHist = []
	for i in range(2):
		_tHist, _tBins = np.histogram(dtime[channel == i] + 0.5, tBins)
		tHist.append(_tHist)
	tBins = tBins[:-1] + 0.5

	# Make the masks
	tBinMasks = []
	tBinMasks.append(tBins * Res * 1e9 < tD[0])
	tBinMasks.append((tBins * Res * 1e9 > tD[0]) & (tBins * Res * 1e9 < tD[1]))
	tBinMasks.append((tBins * Res * 1e9 > tD[1]) & (tBins * Res * 1e9 < tA[0]))
	tBinMasks.append((tBins * Res * 1e9 > tA[0]) & (tBins * Res * 1e9 < tA[1]))
	tBinMasks.append((tBins * Res * 1e9 > tA[1]))

	for i in range(2):
		for j in range(5):
			_mask = tBinMasks[j]
			_a = 1 if ((j == 1) | (j == 3)) else 0.25
			axes[i].plot(tBins[_mask] * Res * 1e9, tHist[i][_mask], color = colors[i], alpha = _a, zorder = 1, linewidth = 0.5)
			axes[i].set_yscale('log')
		axes[i].set_ylabel('counts')
		

	# Plot IRF histograms
	for i in range(2):
		[_tagNames, _tagValues, _globRes, _Res, _dtime, _nsync, _channel] = LoadPTU(IRFfiles[i], header_only = False, chatty = False)
		# Calculate decay histogram
		_tBins = np.arange(int(_globRes/_Res))
		_hist, _bins = np.histogram(_dtime[_channel == i] + 0.5, bins = _tBins)
		_tBins = _tBins[:-1] + 0.5
		_maxHist = max(tHist[i])
		_maxIRFHist = max(_hist)
		axes[i].plot(_tBins * _Res * 1e9, _hist * _maxHist / _maxIRFHist, color = 'grey', alpha = 0.25, zorder = 0, linewidth = 0.5)
	axes[1].set_xlabel('t(ns)')
	axes[1].set_xlim(0, globRes * 1e9)

# Preliminary burst search
def testBurstSearch(file, tBounds, params, plotting = [0, 10000]):
	# Reading ptu file
	[tagNames, tagValues, globRes, Res, dtime, nsync, channel] = LoadPTU(file, chatty = False)

	# Data
	df = pd.DataFrame({
		't' : (dtime + 0.5) * Res * 1e9,
		'T' : nsync * globRes + dtime * Res,
		'channel' : channel
		})
	# Write to df re. D/A information
	df['DD'] = False
	df['AD'] = False
	df['AA'] = False
	tD = tBounds[0]
	tA = tBounds[1]
	df['DD'] = (df['t'].values > tD[0]) & (df['t'].values < tD[1]) & (df['channel'] == 0)
	df['AD'] = (df['t'].values > tD[0]) & (df['t'].values < tD[1]) & (df['channel'] == 1)
	df['AA'] = (df['t'].values > tA[0]) & (df['t'].values < tA[1]) & (df['channel'] == 1)

	# Calculate raw interphoton times with Lee filter
	interPhotonTime = df['T'].values[1:] - df['T'].values[:-1]
	interPhotonTime = LeeFilter(interPhotonTime)
	
	burstLoc = np.argwhere(interPhotonTime < params['burstCut'])[:, 0]
	burstLoc = np.append(burstLoc, burstLoc[-1])
	
	noiseLoc = np.argwhere(interPhotonTime > params['noiseCut'])[:, 0]
	noiseLoc = np.append(noiseLoc, noiseLoc[-1])
	
	burstIdx = np.insert(np.where(burstLoc[1:] - burstLoc[:-1] > 1)[0] + 1, 0, 0)
	noiseIdx = np.insert(np.where(noiseLoc[1:] - noiseLoc[:-1] > 1)[0] + 1, 0, 0)
	
	burstLen = np.zeros(len(burstIdx)).astype(np.int)
	for i in range(len(burstIdx)):
		_length = 1
		_idx = burstIdx[i]
		while burstLoc[_idx+1] == burstLoc[_idx] + 1:
			_length += 1
			_idx += 1
		burstLen[i] = _length
		
	noiseLen = np.zeros(len(noiseIdx)).astype(np.int)
	for i in range(len(noiseIdx)):
		_length = 1
		_idx = noiseIdx[i]
		while noiseLoc[_idx+1] == noiseLoc[_idx] + 1:
			_length += 1
			_idx += 1
		noiseLen[i] = _length
		
	burstMask = burstLen > params['burstLen']
	trueBurstIdx = burstIdx[burstMask]
	trueBurstLen = burstLen[burstMask]
	
	noiseMask = noiseLen > params['noiseLen']
	trueNoiseIdx = noiseIdx[noiseMask]
	trueNoiseLen = noiseLen[noiseMask]
	
	# Write to df re. burst and noise state
	df['burst'] = int(-1)
	df['noise'] = int(-1)
	
	for i in range(len(trueBurstIdx)):
		_i = int(burstLoc[trueBurstIdx[i]])
		_j = int(burstLoc[trueBurstIdx[i] + trueBurstLen[i] - 1] + 1)
		df.loc[_i: _j, 'burst'] = np.ones(len(df.loc[_i : _j, 'burst']), dtype = int) * i
	
	for i in range(len(trueNoiseIdx)):
		_i = int(noiseLoc[trueNoiseIdx[i]])
		_j = int(noiseLoc[trueNoiseIdx[i] + trueNoiseLen[i] - 1] + 1)
		df.loc[_i: _j, 'noise'] = np.ones(len(df.loc[_i : _j, 'noise']), dtype = int) * i
		
	print('Burst count: {0:}\nNoise count: {1:}\n'.format(len(trueBurstIdx), len(trueNoiseIdx)))

	# Calculate background signal in units of photons per second. Noise rates should be around 1 kHz
	noiseDD = 0
	noiseAD = 0
	noiseAA = 0
	noiseT = 0
	
	for i in range(len(trueNoiseIdx)):
		_df = df[df['noise'] == i]
		noiseT += (_df['T'].values[-1] - _df['T'].values[0])
		noiseDD += len(_df[_df['DD']]) - 1
		noiseAD += len(_df[_df['AD']]) - 1
		noiseAA += len(_df[_df['AA']]) - 1
	
	if noiseT != 0:
		noiseDD /= noiseT
		noiseAD /= noiseT
		noiseAA /= noiseT

	print('Noise rates:\nDD : {0:.2g} kHz\nAD : {1:.2g} kHz\nAA : {2:.2g} kHz\n'.format(noiseDD/1000, noiseAD/1000, noiseAA/1000))

	plt.figure(figsize=(9,4))
	_start = 0
	_end = len(interPhotonTime)
	_interPhotonTimeSlice = interPhotonTime[_start:_end] 
	_n = np.arange(_start, _end)
	plt.plot(_n, _interPhotonTimeSlice * 1e6, alpha = 0.25, color = 'grey', zorder = 0, linewidth = 0.5)
	
	for i in range(len(trueBurstIdx)):
		_i = burstLoc[trueBurstIdx[i]]
		_j = burstLoc[trueBurstIdx[i] + trueBurstLen[i] - 1] + 1
		if (_i > _start) & (_j < (_end)):
			plt.plot(np.arange(_i, _j), interPhotonTime[_i : _j] * 1e6 , color = 'red', zorder = 5, linewidth = 0.5)
	
	for i in range(len(trueNoiseIdx)):
		_i = noiseLoc[trueNoiseIdx[i]]
		_j = noiseLoc[trueNoiseIdx[i] + trueNoiseLen[i] - 1] + 1
		if (_i > _start) & (_j < (_end)):
			plt.plot(np.arange(_i, _j), interPhotonTime[_i : _j] * 1e6, color = 'blue', zorder = 2, linewidth = 0.5)
	
	plt.xlim(plotting[0], plotting[1])
	plt.ylim(0, 1000)
	plt.xlabel('photon number')
	plt.ylabel('inter photon time (us)')

# Test burst cutoff parameter
def verifyBurstCut(sample, maxCut, minCut, cutNum, maxLen):

	# Reading ptu file
	[tagNames, tagValues, globRes, Res, dtime, nsync, channel] = LoadPTU(sample, chatty = False)
	T = np.array(nsync * globRes + dtime * Res)

	# Calculate raw interphoton times with Lee filter
	interPhotonTime = T[1:] - T[:-1]
	interPhotonTime = LeeFilter(interPhotonTime)

	plt.figure()

	bins = np.linspace(0, maxLen, 50)
	jet = cm.get_cmap('jet', cutNum + 2)
	cuts = np.linspace(minCut, maxCut, cutNum)
	for i in range(cutNum):
		# Finds all photons that are part of a burst
		# Specifically find indices where inter-photon time is less than threshold
		burstLoc = np.argwhere(interPhotonTime < cuts[i])[:, 0]

		# TODO - Add an extra index?
		burstLoc = np.append(burstLoc, burstLoc[-1])

		# Finds the _index_ in burstLoc of where each burst begins in burstLoc
		# because the array in np.where is made from burstLoc[1:] each indice found must add 1 to account for array subsection
		# finally prepend with zero to account for very first burst index that is lost
		burstIdx = np.insert(np.where(burstLoc[1:] - burstLoc[:-1] > 1)[0] + 1, 0, 0)

		# Initialise Burst length array to be populated
		burstLen = np.zeros(len(burstIdx)).astype(np.int)

		for j in range(len(burstIdx)):
			_length = 1
			# Remember burstIdx is an index for the first burst photon in burstLoc
			_idx = burstIdx[j]
			while burstLoc[_idx+1] == burstLoc[_idx] + 1:
				# if the indexes are adjacent in *burstLoc* then photon in _same_ burst
				_length += 1
				_idx += 1
			burstLen[j] = _length
		_hist, _bins = np.histogram(burstLen, bins = bins)
		if (i == 0) | (i == cutNum - 1):
			plt.plot(_bins[1:], _hist, color = jet(i + 1), label = '{0:.2g} us'.format(cuts[i] * 1e6))
		plt.plot(_bins[1:], _hist, color = jet(i + 1))
	plt.legend()
	plt.yscale('log')
	plt.ylabel('counts')
	plt.xlabel('burst length')


def ex_test_func(filedir, all_hairpin_data):
	print(filedir)
	# Get files to join from this subdir
	files_in_dir = glob.glob(parent_directory + filedir + '/' + '/*/*.ptu')
	print(f'\tNumber photon files in directory: {len(files_in_dir)}')

	# Crunch data 
	this_data_entry = JE_multiple_photon_files(file_list=files_in_dir, tBounds=tBounds, params=params, tauSettings = tauSettings, debug=True)

	# Add Alpha and Delta Corrections
	this_data_entry = addAlphaAndDeltaCorrection(this_data_entry)

	# Add experiment param data
	this_data_entry['NaCl_sample_mM'] = NaCl_conc_sample
	this_data_entry['NaCl_inject_mM'] = NaCl_conc_inject
	this_data_entry['Flow_ul_phr'] = flow_rate
	this_data_entry['HP_sample_num'] = hp_sample_num_int
	this_data_entry['position'] = filedir

	# Add to total dict
	all_hairpin_data[filedir] = this_data_entry

	return all_hairpin_data
