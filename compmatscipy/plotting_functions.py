import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import linregress
from compmatscipy.HelpWithVASP import VASPDOSAnalysis, ProcessDOS, VASPBasicAnalysis, LOBSTERAnalysis, DOEAnalysis
from compmatscipy.CompAnalyzer import CompAnalyzer
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

def set_rc_params():
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
              'axes.edgecolor' : 'black',
              'figure.figsize': [6, 4]}
    for p in params:
        mpl.rcParams[p] = params[p]
    return params

def set_rc_params_small():
    """
    Args:
        
    Returns:
        dictionary of settings for mpl.rcParams
    """
    params = {'axes.linewidth' : 0.75,
              'axes.unicode_minus' : False,
              'figure.dpi' : 600,
              'font.size' : 10,
              'legend.frameon' : False,
              'legend.handletextpad' : 0.2,
              'legend.handlelength' : 0.5,
              'legend.fontsize' : 6,
              'mathtext.default' : 'regular',
              'savefig.bbox' : 'tight',
              'xtick.labelsize' : 10,
              'ytick.labelsize' : 10,
              'xtick.major.size' : 3,
              'ytick.major.size' : 3,
              'xtick.major.width' : 0.75,
              'ytick.major.width' : 0.75,
              'xtick.top' : True,
              'ytick.right' : True,
              'axes.edgecolor' : 'black',
              'figure.figsize': [3, 2]}
    for p in params:
        mpl.rcParams[p] = params[p]
    return params
    
def dos(calc_dir, 
        what_to_plot={'total' : {'spins' : ['summed'],
                                 'orbitals' : ['all']}},
        colors_and_labels = {'total-summed-all' : {'color' : 'black',
                                                   'label' : 'total'}},
        xlim=(0, 0.1), ylim=(-10, 4), 
        xticks=(False, [0, 0.1]), yticks=(False, [-10, 4]), 
        xlabel=r'$DOS/e^-$', ylabel=r'$E-E_F\/(eV)$',
        legend=True,
        smearing=0.2,
        shift='Fermi', normalization='electron',
        cb_shift=False,
        vb_shift=False,
        show=False,
        doscar='DOSCAR.lobster',
        ):
    """
    Args:
        calc_dir (str) - path to calculation with DOSCAR
        what_to_plot (dict) - {element or 'total' (str) : {'spins' : list of spins to include ('summed', 'up', and or 'down'),
                                                           'orbitals' : list of orbitals to include (str)}}
        colors_and_labels (dict) - {element-spin-orbital (str) : {'color' : color (str),
                                                                  'label' : label (str)}}
        xlim (tuple) - (xmin (float), xmax (float))
        ylim (tuple) - (ymin (float), ymax (float))
        xticks (tuple) - (bool to show label or not, (xtick0, xtick1, ...))
        yticks (tuple) - (bool to show label or not, (ytick0, ytick1, ...))
        xlabel (str) - x-axis label
        ylabel (str) - y-axis label
        legend (bool) - include legend or not
        smearing (float or False) - std. dev. for Gaussian smearing of DOS or False for no smearing
        shift (float or 'Fermi') - if 'Fermi', make Fermi level 0; else shift energies by shift
        cb_shift (tuple or False) - shift all energies >= cb_shift[0] (float) by cb_shift[1] (float)
        vb_shift (tuple or False) - shift all energies <= vb_shift[0] (float) by vb_shift[1] (float)             
        normalization ('electron', 'atom', or False) - divide populations by number of electrons, number of atoms, or not at all
        show (bool) - if True, show figure; else just return ax
                   
    Returns:
        matplotlib axes object
    """
    set_rc_params()    
    if show == True:
        fig = plt.figure(figsize=(2.5,4))
        ax = plt.subplot(111)
    if 'lobster' in doscar:
        Efermi = 0.
    else:
        Efermi = VASPBasicAnalysis(calc_dir).Efermi        
    if shift == 'Fermi':
        shift = -Efermi
    if normalization == 'electron':
        normalization = VASPBasicAnalysis(calc_dir).params_from_outcar(num_params=['NELECT'], str_params=[])['NELECT']
    elif normalization == 'atom':
        normalization = VASPBasicAnalysis(calc_dir).nsites
    occupied_up_to = Efermi + shift
    print(occupied_up_to)
    dos_lw = 1    
    for element in what_to_plot:
        for spin in what_to_plot[element]['spins']:
            for orbital in what_to_plot[element]['orbitals']:
                tag = '-'.join([element, spin, orbital])
                color = colors_and_labels[tag]['color']
                label = colors_and_labels[tag]['label']
                d = VASPDOSAnalysis(calc_dir, doscar=doscar).energies_to_populations(element=element,
                                                                      orbital=orbital,
                                                                      spin=spin)
                if spin == 'down':
                    flip_sign = True
                else:
                    flip_sign = False
                d = ProcessDOS(d, shift=shift, 
                               flip_sign=flip_sign,
                               normalization=normalization,
                               cb_shift=cb_shift,
                               vb_shift=vb_shift).energies_to_populations
                               
                energies = sorted(list(d.keys()))
                populations = [d[E] for E in energies]
                occ_energies = [E for E in energies if E <= occupied_up_to]
                occ_populations = [d[E] for E in occ_energies]
                unocc_energies = [E for E in energies if E > occupied_up_to]
                unocc_populations = [d[E] for E in unocc_energies]    
                if smearing:
                    occ_populations = gaussian_filter1d(occ_populations, smearing)
                    unocc_populations = gaussian_filter1d(unocc_populations, smearing)
                ax = plt.plot(occ_populations, occ_energies, color=color, label=label, alpha=0.9, lw=dos_lw)
                ax = plt.plot(unocc_populations, unocc_energies, color=color, label='__nolegend__', alpha=0.9, lw=dos_lw)                    
                ax = plt.fill_betweenx(occ_energies, occ_populations, color=color, alpha=0.2, lw=0)                               
                                  
    ax = plt.xticks(xticks[1])
    ax = plt.yticks(yticks[1])
    if not xticks[0]:
        ax = plt.gca().xaxis.set_ticklabels([])      
    if not yticks[0]:
        ax = plt.gca().yaxis.set_ticklabels([])
    ax = plt.xlabel(xlabel)
    ax = plt.ylabel(ylabel)
    ax = plt.xlim(xlim)
    ax = plt.ylim(ylim)    
    if legend:
        ax = plt.legend(loc='upper right')
    if show:
        plt.show()
    return ax

def cohp(calc_dir,
         pairs_to_plot=['total'],
         colors_and_labels = {'total' : {'color' : 'black',
                                         'label' : 'total'}},
         tdos=False,                                        
         xlim=(-0.5, 0.5), ylim=(-10, 4), 
         xticks=(False, [-0.5, 0.5]), yticks=(False, [-10, 4]),
         xlabel=r'$-COHP/e^-$', ylabel=r'$E-E_F\/(eV)$',
         legend=True,
         smearing=1,
         shift=0, normalization='electron',
         show=False,
         zero_line='horizontal'):
    """
    Args:
        calc_dir (str) - path to calculation with DOSCAR
        pairs_to_plot (list) - list of 'el1_el2' to plot and/or 'total'
        colors_and_labels (dict) - {pair (str) : {'color' : color (str),
                                                  'label' : label (str)}}
        tdos (str or bool) - if not False, 'DOSCAR' or 'DOSCAR.losbter' to retrieve tDOS from
        xlim (tuple) - (xmin (float), xmax (float))
        ylim (tuple) - (ymin (float), ymax (float))
        xticks (tuple) - (bool to show label or not, (xtick0, xtick1, ...))
        yticks (tuple) - (bool to show label or not, (ytick0, ytick1, ...))
        xlabel (str) - x-axis label
        ylabel (str) - y-axis label
        legend (bool) - include legend or not
        smearing (float or False) - std. dev. for Gaussian smearing of DOS or False for no smearing
        shift (float or 'Fermi') - if 'Fermi', make Fermi level 0; else shift energies by shift
        normalization ('electron', 'atom', or False) - divide populations by number of electrons, number of atoms, or not at all
        show (bool) - if True, show figure; else just return ax
        zero_line (str) - if 'horizontal', 'vertical', 'both', or False
                   
    Returns:
        matplotlib axes object
    """
    set_rc_params()    
    if show == True:
        fig = plt.figure(figsize=(2.5,4))
        ax = plt.subplot(111)         
    if normalization == 'electron':
        normalization = VASPBasicAnalysis(calc_dir).params_from_outcar(num_params=['NELECT'], str_params=[])['NELECT']
    elif normalization == 'atom':
        normalization = VASPBasicAnalysis(calc_dir).nsites
    occupied_up_to = shift
    dos_lw = 1
    if isinstance(tdos, str):
        d = VASPDOSAnalysis(calc_dir, doscar=tdos).energies_to_populations()
        if 'lobster' not in tdos:
            shift -= VASPBasicAnalysis(calc_dir).Efermi
        d = ProcessDOS(d, shift=shift, normalization=normalization).energies_to_populations
        energies = sorted(list(d.keys()))
        populations = [d[E] for E in energies]
        occ_energies = [E for E in energies if E <= occupied_up_to]
        occ_populations = [d[E] for E in occ_energies]
        unocc_energies = [E for E in energies if E > occupied_up_to]
        unocc_populations = [d[E] for E in unocc_energies]    
        color = 'black'
        label = 'tDOS'
        if smearing:
            occ_populations = gaussian_filter1d(occ_populations, smearing)
            unocc_populations = gaussian_filter1d(unocc_populations, smearing)
        ax = plt.plot(occ_populations, occ_energies, color=color, label=label, alpha=0.9, lw=dos_lw)
        ax = plt.plot(unocc_populations, unocc_energies, color=color, label='__nolegend__', alpha=0.9, lw=dos_lw)                    
        ax = plt.fill_betweenx(occ_energies, occ_populations, color=color, alpha=0.2, lw=0)         
    for pair in pairs_to_plot:
        color = colors_and_labels[pair]['color']
        label = colors_and_labels[pair]['label']
        d = LOBSTERAnalysis(calc_dir).energies_to_populations(element_pair=pair)
        flip_sign = True
        d = ProcessDOS(d, shift=shift, 
                       flip_sign=flip_sign,
                       normalization=normalization).energies_to_populations
        energies = sorted(list(d.keys()))
        populations = [d[E] for E in energies]
        occ_energies = [E for E in energies if E <= occupied_up_to]
        occ_populations = [d[E] for E in occ_energies]
        unocc_energies = [E for E in energies if E > occupied_up_to]
        unocc_populations = [d[E] for E in unocc_energies]
        if smearing:
            occ_populations = gaussian_filter1d(occ_populations, smearing)
            unocc_populations = gaussian_filter1d(unocc_populations, smearing)
        ax = plt.plot(occ_populations, occ_energies, color=color, label=label, alpha=0.9, lw=dos_lw)
        ax = plt.plot(unocc_populations, unocc_energies, color=color, label='__nolegend__', alpha=0.9, lw=dos_lw) 
        ax = plt.fill_betweenx(occ_energies, occ_populations, color=color, alpha=0.2, lw=0)
    ax = plt.xticks(xticks[1])
    ax = plt.yticks(yticks[1])
    if not xticks[0]:
        ax = plt.gca().xaxis.set_ticklabels([])      
    if not yticks[0]:
        ax = plt.gca().yaxis.set_ticklabels([])
    ax = plt.xlabel(xlabel)
    ax = plt.ylabel(ylabel)
    ax = plt.xlim(xlim)
    ax = plt.ylim(ylim)
    if zero_line in ['horizontal', 'both']:
        ax = plt.plot(xlim, [0, 0], lw=1, ls='--', color='black')
    if zero_line in ['vertical', 'both']:
        ax = plt.plot([0, 0], ylim, lw=1, ls='--', color='black')      
    if legend:
        ax = plt.legend(loc='upper right')
    if show:
        plt.show()
    return ax

def doe(calc_dir, 
        what_to_plot={'total' : {'spins' : ['summed']}},
        colors_and_labels = {'total-summed' : {'color' : 'black',
                                                   'label' : 'total'}},
        xlim=(0, 0.1), ylim=(-10, 4), 
        xticks=(False, [0, 0.1]), yticks=(False, [-10, 4]), 
        xlabel=r'$DOE/e^-$', ylabel=r'$E-E_F\/(eV)$',
        legend=True,
        smearing=0.2,
        shift='Fermi', normalization='electron',
        cb_shift=False,
        vb_shift=False,
        show=False
        ):
    """
    Args:
        calc_dir (str) - path to calculation with DOSCAR
        what_to_plot (dict) - {element or 'total' (str) : {'spins' : list of spins to include ('summed', 'up', and or 'down'),
                                                           'orbitals' : list of orbitals to include (str)}}
        colors_and_labels (dict) - {element-spin-orbital (str) : {'color' : color (str),
                                                                  'label' : label (str)}}
        xlim (tuple) - (xmin (float), xmax (float))
        ylim (tuple) - (ymin (float), ymax (float))
        xticks (tuple) - (bool to show label or not, (xtick0, xtick1, ...))
        yticks (tuple) - (bool to show label or not, (ytick0, ytick1, ...))
        xlabel (str) - x-axis label
        ylabel (str) - y-axis label
        legend (bool) - include legend or not
        smearing (float or False) - std. dev. for Gaussian smearing of DOS or False for no smearing
        shift (float or 'Fermi') - if 'Fermi', make Fermi level 0; else shift energies by shift
        cb_shift (tuple or False) - shift all energies >= cb_shift[0] (float) by cb_shift[1] (float)
        vb_shift (tuple or False) - shift all energies <= vb_shift[0] (float) by vb_shift[1] (float)             
        normalization ('electron', 'atom', or False) - divide populations by number of electrons, number of atoms, or not at all
        show (bool) - if True, show figure; else just return ax
                   
    Returns:
        matplotlib axes object
    """
    set_rc_params()    
    if show == True:
        fig = plt.figure(figsize=(2.5,4))
        ax = plt.subplot(111)
    Efermi = 0.
    if shift == 'Fermi':
        shift = -Efermi
    if normalization == 'electron':
        normalization = VASPBasicAnalysis(calc_dir).params_from_outcar(num_params=['NELECT'], str_params=[])['NELECT']
    elif normalization == 'atom':
        normalization = VASPBasicAnalysis(calc_dir).nsites
    occupied_up_to = Efermi + shift
    dos_lw = 1    
    for element in what_to_plot:
        for spin in what_to_plot[element]['spins']:
            tag = '-'.join([element, spin])
            color = colors_and_labels[tag]['color']
            label = colors_and_labels[tag]['label']
            d = DOEAnalysis(calc_dir).energies_to_populations(spin=spin)
            d = ProcessDOS(d, shift=shift,
                           flip_sign=True,
                           normalization=normalization,
                           cb_shift=cb_shift,
                           vb_shift=vb_shift).energies_to_populations
                           
            energies = sorted(list(d.keys()))
            occ_energies = [E for E in energies if E <= occupied_up_to]
            occ_populations = [d[E] for E in occ_energies]
            unocc_energies = [E for E in energies if E > occupied_up_to]
            unocc_populations = [d[E] for E in unocc_energies]    
            if smearing:
                occ_populations = gaussian_filter1d(occ_populations, smearing)
                unocc_populations = gaussian_filter1d(unocc_populations, smearing)
            ax = plt.plot(occ_populations, occ_energies, color=color, label=label, alpha=0.9, lw=dos_lw)
            ax = plt.plot(unocc_populations, unocc_energies, color=color, label='__nolegend__', alpha=0.9, lw=dos_lw)                    
            ax = plt.fill_betweenx(occ_energies, occ_populations, color=color, alpha=0.2, lw=0)                               
                                  
    ax = plt.xticks(xticks[1])
    ax = plt.yticks(yticks[1])
    if not xticks[0]:
        ax = plt.gca().xaxis.set_ticklabels([])      
    if not yticks[0]:
        ax = plt.gca().yaxis.set_ticklabels([])
    ax = plt.xlabel(xlabel)
    ax = plt.ylabel(ylabel)
    ax = plt.xlim(xlim)
    ax = plt.ylim(ylim)    
    if legend:
        ax = plt.legend(loc='upper right')
    if show:
        plt.show()
    return ax

def hull2d(data, 
           el1, el2, 
           xlim, ylim, 
           xticks, yticks,
           xlabel, ylabel,
           leg,
           labels):
    tableau = tableau_colors()
    cmpds = [k for k in data if CompAnalyzer(k).els == sorted([el1, el2])]
    stable_cmpds = [k for k in cmpds if data[k]['stability']]
    unstable_cmpds = [k for k in cmpds if not data[k]['stability']]
    x_stable = [CompAnalyzer(k).fractional_amt_of_el(el1) for k in stable_cmpds]
    x_unstable = [CompAnalyzer(k).fractional_amt_of_el(el1) for k in unstable_cmpds]
    stable_indices = np.argsort(x_stable)
    unstable_indices = np.argsort(x_unstable)
    x_stable = [0] + [x_stable[i] for i in stable_indices] + [1]
    x_unstable = [x_unstable[i] for i in unstable_indices]
    y_stable = [0] + [data[stable_cmpds[i]]['Ef'] for i in stable_indices] + [0]
    y_unstable = [data[unstable_cmpds[i]]['Ef'] for i in unstable_indices]
    
    ax = plt.plot(x_stable, y_stable,
                  color=tableau['blue'],
                  ls='-',
                  markeredgecolor=tableau['blue'],
                  markerfacecolor='white',
                  marker='o',
                  label='stable')
    ax = plt.scatter(x_unstable, y_unstable,
                     edgecolor=tableau['red'],
                     color='white',
                     marker='^',
                     label='unstable')
    
    if ylabel:
        ax = plt.ylabel(ylabel)
    if xlabel:
        ax = plt.xlabel(r'$x\/in\/%s_{1-x}%s_x$' % (el2, el1))
    ax = plt.xticks(xticks[1])
    ax = plt.xlim(xlim)
    ax = plt.yticks(yticks[1])
    ax = plt.ylim(ylim)
    if not yticks[0]:
        ax = plt.gca().yaxis.set_ticklabels([])
    if not xticks[0]:
        ax = plt.gca().xaxis.set_ticklabels([])
    if leg:
        ax = plt.legend(loc=leg)
    return ax

def hull3d():
    return

def main():
    return

if __name__ == '__main__':
    main()
