import modules.modulesCorrectionFactorsAndPlots as MCF
import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


"""
These are pretty specific functions (hence the number of defaulted argument) however moved to here
to clean up the jupyter notebooks

Functions are specific to fitting the E distributions to funnctions provided
"""


def get_and_fit_hist(ax, 
                     x, 
                     bins, 
                     curve_func, 
                     initial_guesses=None, 
                     bounds = None, 
                     density = True, 
                     label = "", 
                     alpha = 0.3,
                     plt_title = None):
    """
    Importantly curve_func must have signature curve_func(x_array, *args) otherwise plotting fit won't work
    """
    # Histogram      
    ax.hist(x, bins = bins, density = True, label = label, alpha = alpha)

    # Fit Curve to Histogram
    db = (bins[1] - bins[0])/2
    xHist, _ = np.histogram(x, bins = bins, density = True)
    
    if initial_guesses is None and bounds is None:
        popt, popv = curve_fit(curve_func, bins[:-1]+db, xHist)
    elif initial_guesses is not None and bounds is None:
        popt, popv = curve_fit(curve_func, bins[:-1]+db, xHist, p0 = initial_guesses)
    elif initial_guesses is None and bounds is not None:
        popt, popv = curve_fit(curve_func, bins[:-1]+db, xHist, bounds = bounds)
    else:
        popt, popv = curve_fit(curve_func, bins[:-1]+db, xHist, p0 = initial_guesses, bounds = bounds)

    # Plot curve fit
    binspace_higher_res = np.linspace(bins[0], bins[-1], 4*len(bins))
    ax.plot(binspace_higher_res, curve_func(binspace_higher_res, *popt), label = label)  
    
    # Title
    if plt_title is not None:
        ax.set_title(plt_title)

    return ax, popt, popv


def get_fit_and_hist_by_group(df, 
                              group_vals, 
                              group_name, 
                              figs_per_row, 
                              bins,
                              curve_func, 
                              S_column = 'S3', 
                              E_column = 'E3',
                              Smask = [0.85, 0.15],
                              ALEX2CDEmask = 20, 
                              FRET2CDEmask = 20,
                              initial_guesses=None,
                              bounds = None,
                              density = True,
                              label = "",
                              alpha = 0.3,
                              plt_title = '{}'):
    """
    plot title can be formatted with {g}
    """
    group_fits = {}
    
    num_rows = int(math.ceil(len(group_vals)/figs_per_row))
    fig, axs = plt.subplots(num_rows, figs_per_row)
    
    for indx, g in enumerate(group_vals):
        # Get Ax
        _ax = axs[indx//figs_per_row, indx%figs_per_row]

        # Get Specific Group data
        _df = df[df[group_name]==g]
        S_vals = _df[S_column]
        
        # S & CDE filter  
        _df_cde = MCF.typical_S_ALEX_FRET_CDE_filter(_df, S=S_vals, 
                                                     Smask = Smask, 
                                                     ALEX2CDEmask = ALEX2CDEmask, 
                                                     FRET2CDEmask = FRET2CDEmask)

        # E Values
        x = _df_cde[E_column]
        
        # title
        plt_title = plt_title.format(g)

        # Histogram      
        _ax, popt, popv = get_and_fit_hist(ax=_ax, 
                               x=x, 
                               bins=bins,
                               curve_func=curve_func,
                               initial_guesses=initial_guesses,
                               bounds = bounds,
                               density = density,
                               label = label,
                               alpha = alpha,
                               plt_title = plt_title)
        
        #group_fits.append(popt)
        #group_fits.append(popv)
        group_fits[g] = {'popt': popt, 'popv': popv}
        
    for ax in axs.flat:
        ax.set(xlabel='E', ylabel='Nrmlsed Distr.')
        ax.label_outer()  
    
    return fig, group_fits