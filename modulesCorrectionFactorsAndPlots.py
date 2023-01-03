from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
CORRECTION FACTORS

Correction factors explained in notes of "Precision and accuracy of single-molecule FRET measurements a multi-laboratory benchmark study"  
(https://www.nature.com/articles/s41592-018-0085-0)
"""

def calculate_alpha_and_delta_correction(burst_data,D_only = [0,1], D_only_threshold = 0.2, A_only = [[-0.2,1.2],[-0.15, 0.15]]):
    """
    Expects a dataframe with E, S, AA, DD, AD burst info calculated
    Returns the alpha and delta corrected values E and S values
    Location of A-only and D-only population mask can be overwritten
        D_only - E,S coordinate of cenre of A only population
        A_only - pair of E coordinates where E_0 < A_only < E_1, and S_0 < A_only < S_1
    """
    # Unpack
    E = burst_data['E']
    S = burst_data['S']
    AD = burst_data['AD']
    AA = burst_data['AA']
    DD = burst_data['DD']
    
    # Alpha correction
    DonorMask = np.sqrt((E - D_only[0]) ** 2 + (S - D_only[1]) ** 2) < D_only_threshold
    DonorE = np.mean(E[DonorMask])
    Alpha = DonorE / (1 - DonorE)
    
    # Delta correction
    AcceptorMask = (E > A_only[0][0]) & (E < A_only[0][1]) & (S > A_only[1][0]) & (S < A_only[1][1])
    AcceptorS = np.mean(S[AcceptorMask])
    Delta = AcceptorS / (1 - AcceptorS)
    
    return Alpha, Delta

def add_alpha_and_delta_correction(burst_data, Alpha, Delta): 
    # Unpack
    AD = burst_data['AD']
    AA = burst_data['AA']
    DD = burst_data['DD']
    
    # Calculate new E, S corrections
    AD2 = AD - Alpha * DD - Delta * AA
    E2 = AD2 / (AD2 + DD)
    S2 = (AD2 + DD)/(AD2 + DD + AA)
    
    # Add corrected columns
    burst_data['E2'] = E2
    burst_data['S2'] = S2
    
    return burst_data

def calculate_and_add_alpha_and_delta_correction(burst_data):
    alpha, delta = calculate_alpha_and_delta_correction(burst_data)
    burst_data = add_alpha_and_delta_correction(burst_data, alpha, delta)
    return burst_data, alpha, delta

def calculate_beta_and_gamma_correction(burst_data, population_E_limits = [[0,1]], E = None, S = None, sample_percent = 0.05, plot_fig = False):
    """
    Calculates beta and gamma correction factors using population inputs
    sample_percent is percentage of point selected around KDE maximum

    Data should be filtered for FRET relevant burst events (i.e ALEX/FRET CDE filter)
    
    population_E_limits: array of arrays, each giving the lower and upper E of a different FRET population
    population_E_limits MUST have at least two FRET populations in order to calculate 
    """
    assert len(population_E_limits) >= 2, "This function requires two separate FRET populations defined in population_E_limits"
    
    if E is None:
        E = burst_data['E']
    if S is None:
        S = burst_data['S']
        
    if plot_fig:
        plt.figure()
    
    exp_E = []
    exp_S = []
    
    for pop in population_E_limits:
        E_lower = pop[0]
        E_upper = pop[1]
        E_mask = (E >= E_lower) & (E <= E_upper)
        
        _E = E[E_mask]
        _S = S[E_mask]
        _ES = np.vstack([_E, _S])
        
        k = gaussian_kde([_E, _S])
        _z = k(_ES)
        
        if plot_fig:
            plt.scatter(_E, _S, c = _z, cmap = 'jet', s = 0.1)
        
        # Creates a coordinate covering E-S space
        _E_coords, _S_coords = np.meshgrid(np.linspace(min(_E), max(_E)), np.linspace(min(_S), max(_S)))
        _E_coords = _E_coords.flatten()
        _S_coords = _S_coords.flatten()

        # Calculates Kernel Density over all E-S grid
        _z_coord_vals = k(np.vstack([_E_coords, _S_coords]))

        # Samples only the top 5% of z points?
        # Also takes transpose to get form [[z], [x], [y]] from [(z, x, y), ...]
        _ZES = np.array(sorted(zip(_z_coord_vals, _E_coords, _S_coords), reverse= True)[0:int(len(_z_coord_vals) * sample_percent)]).T

        # For this sample, calculates the expectation value, <E>, and <S>
        # <x> = int f(x)*x / f(x), where f(x) is a density function
        _exp_E = np.sum(_ZES[0] * _ZES[1])/np.sum(_ZES[0])
        _exp_S = np.sum(_ZES[0] * _ZES[2])/np.sum(_ZES[0])
        
        exp_E.append(_exp_E)
        exp_S.append(_exp_S)
    
    if plot_fig:
        plt.scatter(exp_E, exp_S, s = 10, facecolor = 'white', edgecolor = 'black')
        plt.xlim(-0.1,1.1)
        plt.ylim(-0.1,1.1)
        plt.show()

    # Fit the <E>, <S> values to line
    fit_vals = np.polyfit(exp_E, 1/np.array(exp_S), 1)
    a = fit_vals[1]
    b = fit_vals[0]
    Beta = a + b - 1
    Gamma = (a - 1) / (a + b - 1)
    
    return Beta, Gamma

def add_all_corrections(burst_data, Beta, Gamma, Alpha, Delta, E_column_name = 'E3', S_column_name = 'S3'):
    """
    Adds in total correction. alpha, beta, gamma, delta can either be an array matching length of burst data or 
    singly valued

    This can be used to add in columns exploring correction factor limits (though default is standard E3, S3 column)
    """
    # Unpack
    AD = burst_data['AD']
    AA = burst_data['AA']
    DD = burst_data['DD']
    
    # Calculate adjusted E, S
    E_corrected = (AD - Alpha * DD - Delta * AA) / (AD - Alpha * DD - Delta * AA + Gamma * DD)
    S_corrected = (AD - Alpha * DD - Delta * AA + Gamma * DD)/(AD - Alpha * DD - Delta * AA + Gamma * DD + AA/Beta)
    
    # Add corrected columns
    #burst_data['E3'] = E3
    #burst_data['S3'] = S3
    burst_data[E_column_name] = E_corrected
    burst_data[S_column_name] = S_corrected
    
    return burst_data

def calculate_and_add_all_corrections(burst_data, population_E_limits = [[0,1]], sample_percent = 0.05, plot_fig = False, Smask = [0.85, 0.15], ALEX2CDEmask = 20, FRET2CDEmask = 20, include_corrections = False):
    """
    Combines all of the above
    Some default filtering (FRET/ALEX CDE) is currently hard coded but can be changed easily
    """
    # Alpha and Delta First
    burst_data, alpha, delta = calculate_and_add_alpha_and_delta_correction(burst_data)

    # This has added the E2, S2, however for beta gamma we also want to filter for FRET relevant events
    cde_filtered_data = typical_S_ALEX_FRET_CDE_filter(burst_data, S=burst_data['S2'], Smask=Smask, ALEX2CDEmask=ALEX2CDEmask, FRET2CDEmask=FRET2CDEmask)
    
    # columns which we want to use to calculate Beta and Gamma
    E2 = cde_filtered_data['E2']
    S2 = cde_filtered_data['S2']

    # Then Beta and Gamma
    beta, gamma = calculate_beta_and_gamma_correction(burst_data, population_E_limits = population_E_limits, E = E2, S = S2, sample_percent = sample_percent, plot_fig = plot_fig)
    burst_data = add_all_corrections(burst_data, beta, gamma, alpha, delta)

    # Include correction details
    if include_corrections:
        burst_data['alpha'] = alpha
        burst_data['beta'] = beta
        burst_data['gamma'] = gamma
        burst_data['delta'] = delta

    return burst_data, alpha, delta, beta, gamma


def calculate_and_add_corrections_by_group(processed_data, group_vals, group_name, population_E_limits=[[0,1]],plot_fig=True, include_corrections = True):
    """
    Very similar to above but automates multiple diferent (grouped) runs
    Some default filtering (FRET/ALEX CDE) is currently hard coded but can be changed easily
    """
    corrections = {}
    df_copies = {}

    for g in group_vals:
        # Make a copy to stop pandas worrying about chained indexes 
        # (as indexes accessed through function also)
        print(f'Correcting group {g}')
        _df_copy = processed_data[processed_data[group_name]==g].copy()
        
        # Run Corrections
        _df_copy, _alpha, _delta, _beta, _gamma = \
        calculate_and_add_all_corrections(_df_copy, population_E_limits=population_E_limits,plot_fig=plot_fig, include_corrections = include_corrections)

        # Store Results
        corrections[g] = {'alpha': _alpha, 'beta': _beta, 'delta': _delta, 'gamma':_gamma}
        df_copies[g] = _df_copy

    corrections_df = pd.DataFrame(corrections)
    corrected_data = pd.concat([df_copies[g] for g in df_copies.keys()], ignore_index=True)
    
    return corrected_data, corrections_df
    
"""
PLOTTING & UTILITIES
"""

def typical_S_ALEX_FRET_CDE_filter(data, S=None, ALEX2CDE=None, FRET2CDE=None, Smask = [0.85, 0.15], ALEX2CDEmask = 20, FRET2CDEmask = 20):
    """
    Typical S, ALEX, and 2CDE filters
    Default values can be overwritten
    data is a pandas df
    """
    if S is None:
        S = data['S']
    if ALEX2CDE is None:
        ALEX2CDE = data['ALEX2CDE']
    if FRET2CDE is None:
        FRET2CDE = data['FRET2CDE']
        
    S_Filter = (S < Smask[0]) & (S > Smask[1])
    Alex_Filter = ALEX2CDE < ALEX2CDEmask
    CDE_Filter = FRET2CDE < FRET2CDEmask
    Total_Filter = S_Filter & Alex_Filter & CDE_Filter
    
    return data[Total_Filter]


def plot_S_E_kernel(E_vals, S_vals, title, save_fig = False, save_path = None, 
                    bw_method_override = None, shade_val = 0.5, ax_lims = [[-0.1, 1.1], [-0.1, 1.1]]):
    """
    Plots S-E kernel density,
    Requires E and S to already be suitably masked
    """
    # Get Kernel Density
    if bw_method_override is None:
        k = gaussian_kde([E_vals, S_vals], bw_method = 0.8 * len(E_vals)**(-0.2))
    else:
        raise("unhandled bw_method")
    
    # Get Kernel shading
    ES = np.vstack([E_vals,S_vals])
    z = k(ES)
    
    # Figure
    plt.figure()
    plt.scatter(E_vals, S_vals, c = z, cmap = 'jet', s = shade_val)
    
    # Labels
    plt.ylabel("S")
    plt.xlabel("E")
    plt.title(title)
    
    # limits
    plt.xlim(ax_lims[0][0], ax_lims[0][1])
    plt.ylim(ax_lims[1][0], ax_lims[1][1])
    
    if save_fig:
        plt.savefig(save_path)
    plt.show()

def plot_kernels_by_group(df, group_vals, group_name, cde_filter = True, E_column_name = 'E', S_column_name = 'S', title = 'Group {}'):
    """
    title can be customised using {} for g
    """
    for g in group_vals:
        _df = df[df[group_name]==g]
        plt_title = title.format(g)
        if cde_filter:         
            _df = typical_S_ALEX_FRET_CDE_filter(_df)
        E = _df[E_column_name]
        S = _df[S_column_name]
        plot_S_E_kernel(E, S, plt_title)      

def calculate_relative_population_proportions(E, population_limits):
    """
    Needs filtered for S, ALEX2CDE, FRET2CDE ranges before using
    population_limits is an array of arrays, each array element being an E range
    proportions are relative to each population not the entire E range (i.e. there may be E values not counted if not in range)
    """
    populations = []
    
    for population in population_limits:
        pop_mask = (E >= population[0]) & (E <= population[1])
        populations.append(sum(pop_mask))
        
    total_population_count = sum(populations)
    
    population_proportions = [p/total_population_count for p in populations]
    
    return population_proportions, populations


def get_dict_string(d):
    s = ''
    for key, val in d.items():
        s += f'{key}: {val:0.2f}; '
    return s

def combine_burstDataFrame_dictionary(burst_data_dictionary):
    # Join all dataframes into single list
    burstdata_list = [burst_data_dictionary[key] for key in burst_data_dictionary.keys()]
    combined_burst_data_df = pd.concat(burstdata_list, ignore_index=True)
    return combined_burst_data_df

def save_burstDataFrame(burst_dataframe, save_path, overwrite=False):
    if os.path.exists(save_path) and not overwrite:
        raise("File exists and overwrite set to false!")
    else:
        burst_dataframe.to_csv(save_path, index=False)