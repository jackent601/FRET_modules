import numpy as np
import glob
import os
import modules as M
import modulesCorrectionFactorsAndPlots as MCF
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    # ===================================================================================
    # PARSER
    # ===================================================================================
    parser = argparse.ArgumentParser()
    # File Details
    parser.add_argument("root_dir", help="root directory of acquisitions to parse", type=str)
    parser.add_argument("--ptu_file_signature", help="signature to find .ptu files from root dir, plugged into glob(root_dir + signature)", type=str, default = "/*/*.ptu")
    # Parsing Params
    parser.add_argument("--burstCut_us", help="Burst Cut Threshold in us", type=float, default = 60)
    parser.add_argument("--noiseCut_us", help="Noise Cut Threshold in us", type=float, default = 60)
    parser.add_argument("--burstLen", help="Burst length Threshold in #photons", type=int, default = 40)
    parser.add_argument("--noiseLen", help="Noise length Threshold in #photons", type=int, default = 40)
    # Debug
    parser.add_argument('-v', '--verbose', action='store_true')
    # Get Parser
    args = parser.parse_args()

    # ===================================================================================
    # PROCESS
    # ===================================================================================
    # Get ptu files to parse
    ptus = glob.glob(args.root_dir + args.ptu_file_signature)
    print(f'found {len(ptus)} ptu files to parse')
    if args.verbose:
        print(f'ptus: {ptus}')

    # Time parameters (Hard Coded, Don't Change Often)
    tD = np.array([1, 19]) # Donor channel
    tA = np.array([21, 39]) # Acceptor channel
    tBounds = [tD, tA] # Grouped together

    # Parsing Parameters - All can be overwritten
    params = {'burstCut': args.burstCut_us*1e-6,
              'noiseCut': args.noiseCut_us*1e-6,
              'burstLen': args.burstLen,
              'noiseLen': args.noiseLen}
    if args.verbose:
        print(f'Params: {params}')

    # Tau parameters (comment out to skip0)
    #tauSettings = None
    tauSettings = {'mode' : 1,'tInterest' : [[3.4, 19], [23.4, 39]]}

    # Parse
    data = M.parse_multiple_photon_files(ptus,
                                     tBounds,
                                     params,
                                     tauSettings=None,
                                     debug=True)

    # Save Data
    data.to_csv(os.path.join(args.root_dir, "burstAggregated.csv"), index=False)

    # ===================================================================================
    # PROCESS
    # ===================================================================================
    plt_title = f'Uncorrected/Unfiltered E-S\n 0.15 < S < 0.85'
    E = data['E']
    S = data['S']

    # Emask = E>0
    mask = (S > 0.15) & (S < 0.85) 
    E = E[mask]
    S = S[mask]

    ax = MCF.plot_S_E_kernel(E, S, plt_title, show_plot=False)
    plt.savefig(os.path.join(args.root_dir, 'burstAggreated_S_E_uncorrected.png'))
    plt.show()
