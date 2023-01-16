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


def fit_hist(x, bins, curve_func, initial_guesses=None, bounds = None, density = True):
    """
    General, combines calculating histogram (x) and fitting resulting histogram to a function provided (curve_func)
    """

    # Fit Curve to Histogram
    db = (bins[1] - bins[0])/2
    xHist, _ = np.histogram(x, bins = bins, density = density)
    
    if initial_guesses is None and bounds is None:
        popt, popv = curve_fit(curve_func, bins[:-1]+db, xHist)
    elif initial_guesses is not None and bounds is None:
        popt, popv = curve_fit(curve_func, bins[:-1]+db, xHist, p0 = initial_guesses)
    elif initial_guesses is None and bounds is not None:
        popt, popv = curve_fit(curve_func, bins[:-1]+db, xHist, bounds = bounds)
    else:
        popt, popv = curve_fit(curve_func, bins[:-1]+db, xHist, p0 = initial_guesses, bounds = bounds)

    return xHist, popt, popv


def fit_Ehist_by_group(df,
                      group_vals,
                      group_name,
                      bins,
                      curve_func,
                      S_column='S3',
                      E_column='E3',
                      Smask=[0.85, 0.15],
                      ALEX2CDEmask=20,
                      FRET2CDEmask=20,
                      initial_guesses=None,
                      bounds=None,
                      density=True):
    """
    Specific for FRET, fits E values for multiple groups
    """
    group_fits = {}
    
    for g in group_vals:
        # Get Specific Group data
        _df = df[df[group_name]==g]
        S_vals = _df[S_column]
        
        # S & CDE filter
        _df_cde = MCF.typical_S_ALEX_FRET_CDE_filter(_df, S=S_vals,
                                                     Smask=Smask,
                                                     ALEX2CDEmask=ALEX2CDEmask,
                                                     FRET2CDEmask=FRET2CDEmask)

        # E Values
        x = _df_cde[E_column]

        # Histogram fit      
        EHist, _popt, _popv = fit_hist(x=x, bins=bins, curve_func=curve_func, 
        initial_guesses=initial_guesses, bounds=bounds, density=density)

        # Add fit values
        group_fits[g] = {'EHist': EHist, 'popt': _popt, 'popv': _popv}
    
    return group_fits

def plot_hist_fit(ax, bin_values, bin_centres, fit_x, fit_y, alpha):
    
    # Histogram      
    ax.bar(bin_centres, bin_values, alpha = alpha)

    # Hist Fit
    ax.plot(fit_x, fit_y)

"""
PLOTTING

Assumes curve parameters have already been calculated
"""
def plot_hist_fit(bin_centres, 
                       bin_values, 
                       fit_x, 
                       fit_y, 
                       ax=None, 
                       show_plot = True, 
                       plt_title=None,
                       plot_parameters={'barKwargs':{}, 'plotKwargs':{},'setKwargs': {}}):
    """
    plot_parameters aesthetics
    """
    
    if ax is None:
        ax = plt.axes()
    
    # Histogram      
    ax.bar(bin_centres, bin_values, width = abs(bin_centres[1]-bin_centres[0]), **plot_parameters['barKwargs'])

    # HistFit
    ax.plot(fit_x, fit_y, **plot_parameters['plotKwargs'])
    
    # Labels
    ax.set(**plot_parameters['setKwargs'])
    if plt_title is not None:
        ax.set_title(plt_title)
    
    if show_plot:
        plt.show()
    
    return ax


def plot_hist_fit_by_group(group_fit_dict,
                           bin_centres,
                           curve_func,
                           fit_x,
                           figs_per_row,
                           HistKey = 'EHist',
                           FitParamsKey = 'popt',
                           plt_title='{}',
                           plot_parameters={'barKwargs': {}, 'plotKwargs': {}, 'setKwargs': {}}):
    """
    Expects a dictionary with group as keys and EHist/group fit info as values

    dict must have 'popt' as fitted parameters to curve_func aaand...

    curve_func must have signature curve_func(x, *args)
    """
    num_rows = int(math.ceil(len(group_fit_dict)/figs_per_row))
    fig, axs = plt.subplots(num_rows, figs_per_row)
    
    indx = 0
    for g, g_dict in group_fit_dict.items():
        # Get Ax
        _ax = axs[indx//figs_per_row, indx%figs_per_row]
        indx += 1
        
        # Get curve fit values from g dict
        _popt = g_dict[FitParamsKey]
        _fit_y = curve_func(fit_x, *_popt)
        
        # title
        _plt_title = plt_title.format(g)
        
        # get subplot
        _ax = plot_hist_fit(bin_centres=bin_centres, 
                       bin_values=g_dict[HistKey], 
                       fit_x=fit_x, 
                       fit_y=_fit_y, 
                       ax=_ax, 
                       show_plot = False, 
                       plt_title=_plt_title,
                       plot_parameters=plot_parameters)
        
    for ax in axs.flat:
        ax.label_outer() 

    return fig


def plot_group_fit_together(group_fits, curve_fit, x_vals, title, plot_bar=False, plot_fit=True, legend=True,
                            plot_parameters={'barKwargs': {}, 'plotKwargs': {}}):
    """
    group_fits entries must have popt
    """
    for k, d in group_fits.items():
        _popt = d['popt']
        if plot_bar:
            plt.bar(x_vals, d['EHist'], width = abs(x_vals[1]-x_vals[0]), **plot_parameters['barKwargs'])
        if plot_fit:
            plt.plot(x_vals, curve_fit(x_vals, *_popt), label = k, **plot_parameters['plotKwargs'])
    if legend:
        plt.legend()
    plt.title(title)
    return plt

# def get_and_fit_hist(ax, 
#                      x, 
#                      bins, 
#                      curve_func, 
#                      initial_guesses=None, 
#                      bounds = None, 
#                      density = True, 
#                      label = "", 
#                      alpha = 0.3,
#                      plt_title = None):
#     """
#     Importantly curve_func must have signature curve_func(x_array, *args) otherwise plotting fit won't work
#     """
#     # Histogram      
#     ax.hist(x, bins = bins, density = density, label = label, alpha = alpha)

#     # Fit Curve to Histogram
#     db = (bins[1] - bins[0])/2
#     xHist, _ = np.histogram(x, bins = bins, density = density)
    
#     if initial_guesses is None and bounds is None:
#         popt, popv = curve_fit(curve_func, bins[:-1]+db, xHist)
#     elif initial_guesses is not None and bounds is None:
#         popt, popv = curve_fit(curve_func, bins[:-1]+db, xHist, p0 = initial_guesses)
#     elif initial_guesses is None and bounds is not None:
#         popt, popv = curve_fit(curve_func, bins[:-1]+db, xHist, bounds = bounds)
#     else:
#         popt, popv = curve_fit(curve_func, bins[:-1]+db, xHist, p0 = initial_guesses, bounds = bounds)

#     # Plot curve fit
#     binspace_higher_res = np.linspace(bins[0], bins[-1], 4*len(bins))
#     ax.plot(binspace_higher_res, curve_func(binspace_higher_res, *popt), label = label)  
    
#     # Title
#     if plt_title is not None:
#         ax.set_title(plt_title)

#     return ax, popt, popv


# def get_fit_and_hist_by_group(df, 
#                               group_vals, 
#                               group_name, 
#                               figs_per_row, 
#                               bins,
#                               curve_func, 
#                               S_column = 'S3', 
#                               E_column = 'E3',
#                               Smask = [0.85, 0.15],
#                               ALEX2CDEmask = 20, 
#                               FRET2CDEmask = 20,
#                               initial_guesses=None,
#                               bounds = None,
#                               density = True,
#                               label = "",
#                               alpha = 0.3,
#                               plt_title = '{}'):
#     """
#     plot title can be formatted with {g}
#     """
#     group_fits = {}
    
#     num_rows = int(math.ceil(len(group_vals)/figs_per_row))
#     fig, axs = plt.subplots(num_rows, figs_per_row)
    
#     for indx, g in enumerate(group_vals):
#         # Get Ax
#         _ax = axs[indx//figs_per_row, indx%figs_per_row]

#         # Get Specific Group data
#         _df = df[df[group_name]==g]
#         S_vals = _df[S_column]
        
#         # S & CDE filter  
#         _df_cde = MCF.typical_S_ALEX_FRET_CDE_filter(_df, S=S_vals, 
#                                                      Smask = Smask, 
#                                                      ALEX2CDEmask = ALEX2CDEmask, 
#                                                      FRET2CDEmask = FRET2CDEmask)

#         # E Values
#         x = _df_cde[E_column]
        
#         # title
#         plt_title = plt_title.format(g)

#         # Histogram      
#         _ax, popt, popv = get_and_fit_hist(ax=_ax, 
#                                x=x, 
#                                bins=bins,
#                                curve_func=curve_func,
#                                initial_guesses=initial_guesses,
#                                bounds = bounds,
#                                density = density,
#                                label = label,
#                                alpha = alpha,
#                                plt_title = plt_title)
        
#         #group_fits.append(popt)
#         #group_fits.append(popv)
#         group_fits[g] = {'popt': popt, 'popv': popv}
        
#     for ax in axs.flat:
#         ax.set(xlabel='E', ylabel='Nrmlsed Distr.')
#         ax.label_outer()  
    
#     return fig, group_fits
