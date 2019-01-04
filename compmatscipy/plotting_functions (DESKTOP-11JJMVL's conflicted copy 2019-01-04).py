# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:26:39 2019

@author: Chris
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import linregress
from compmatscipy.HelpWithVASP import VASPDOSAnalysis, ProcessDOS, VASPBasicAnalysis
import os

def tableau_colors():
    """
    Args:
        
    Returns:
        dictionary of {color (str) : RGB (tuple) for the dark tableau20 colors}
    """
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
      
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)
    names = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'yellow', 'turquoise']
    colors = [tableau20[i] for i in range(0, 20, 2)]
    return dict(zip(names,colors))

def rc_params():
    """
    Args:
        
    Returns:
        dictionary of settings for mpl.rcParams
    """
    params = {'axes.linewidth' : 1.5,
              'axes.unicode_minus' : False,
              'figure.dpi' : 300,
              'font.size' : 20,
              'legend.frameon' : False,
              'legend.handletextpad' : 0.4,
              'legend.handlelength' : 1,
              'legend.fontsize' : 12,
              'mathtext.default' : 'regular',
              'savefig.bbox' : 'tight',
              'xtick.labelsize' : 20,
              'ytick.labelsize' : 20,
              'xtick.major.size' : 6,
              'ytick.major.size' : 6,
              'xtick.major.width' : 1.5,
              'ytick.major.width' : 1.5,
              'xtick.top' : True,
              'ytick.right' : True,
              'axes.edgecolor' : 'black'}
    for p in params:
        mpl.rcParams[p] = params[p]
    return params
    
def dos(calc_dir, 
        elements=['total'], orbitals=['all'], spins=['summed'], colors=['black'],
        xlim=(0, 0.1), ylim=(-10, 4), 
        xticks=False, yticks=False, 
        xlabel=r'$DOS/e^-$', ylabel=r'$E-E_F\/(eV)$',
        legend=True,
        smearing=0.2,
        shift='Fermi', normalization='electron', energy_limits=(-100, 100),
        show=False):
    """
    Args:
        calc_dir (str) - path to calculation with DOSCAR
        xlim (tuple) - (xmin (float), xmax (float))
        ylim (tuple) - (ymin (float), ymax (float))
        xticks (tuple or False) - (xtick0, xtick1, ...) if not False
        yticks (tuple or False) - (ytick0, ytick1, ...) if not False
        xlabel (str) - x-axis label
        ylabel (str) - y-axis label
           
    Returns:
        
    """
    if show == True:
        fig = plt.figure(figsize=(2,4))
        ax = plt.subplot(111)
    for i in range(len(elements)):
        element, orbital, spin, color = elements[i], orbitals[i], spins[i], colors[i]
        d = VASPDOSAnalysis(calc_dir).energies_to_populations(element=element,
                                                              orbital=orbital,
                                                              spin=spin)
        if spin == 'down':
            flip_sign = True
        else:
            flip_sign = False
        if shift == 'Fermi':
            shift = -VASPBasicAnalysis(calc_dir).Efermi
        if normalization == 'electron':
            normalization = VASPBasicAnalysis(calc_dir).params_from_outcar(num_params=['NELECT'], str_params=[])['NELECT']
        elif normalization == 'atom':
            normalization = VASPBasicAnalysis(calc_dir).nsites
        d = ProcessDOS(d, shift=shift, 
                       energy_limits=energy_limits, 
                       flip_sign=flip_sign,
                       normalization=normalization).energies_to_populations
        energies = sorted(list(d.keys()))
        populations = [d[E] for E in energies]
        if smearing:
            populations = gaussian_filter1d(populations, smearing)
        ax = plt.plot(populations, energies, color=color, label=element)
    ax = plt.xlim(xlim)
    ax = plt.ylim(ylim)
    if xticks:
        ax = plt.xticks(xticks)
    else:
        ax = plt.gca().xaxis.set_ticklabels([])
    if xticks:
        ax = plt.yticks(yticks)
    else:
        ax = plt.gca().yaxis.set_ticklabels([])  
    ax = plt.xlabel(xlabel)
    ax = plt.ylabel(ylabel)
    if legend == True:
        ax = plt.legend()
    if show == True:
        plt.show()
        
    return ax

def main():
    calc_dir = os.path.join('tests', 'test_data', 'SCAN_geometry')
    d = VASPDOSAnalysis(calc_dir).detailed_dos_dict(remake=False)
    tableau = tableau_colors()
    elements, orbitals, spins, colors = [], [], [], []
    elements = ['Cs' ,'Ag', 'Au', 'Cl']
    colors = [tableau['blue'], tableau['orange'], tableau['red'], tableau['green'], 'black']
    el_to_color = dict(zip(elements, colors))
    for element in elements:
        orbitals += ['all']
        colors += [el_to_color[element]]
        spins += ['summed']
    xlim = [0, 0.5]
    xticks, yticks = xlim, [-10, -8, -6, -4, -2, 0, 2, 4]
    ax = dos(calc_dir,
             elements=elements, orbitals=orbitals, spins=spins, colors=colors,
             xlim=xlim,
             xticks=xticks, yticks=yticks,
             show=True)
    return d

if __name__ == '__main__':
    d = main()
    rc_params()
    
    
    