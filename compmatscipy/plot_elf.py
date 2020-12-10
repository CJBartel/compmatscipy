#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:48:12 2020

@author: chrisbartel
"""
from pymatgen.io.vasp.outputs import Elfcar
from compmatscipy.plotting_functions import set_rc_params, tableau_colors
from vaspy.electro import ElfCar, ChgCar
import matplotlib.pyplot as plt
import numpy as np


def plot_elf(felfcar, axis_cut,
             savename='tmp.png',
             vmin=0.4, vmax=1,
             thresh_for_slice_ID='auto',
             cmap='Blues',
             nlevels=100,
             widths=(1,1,1),
             in_screen_color='pink',
             out_of_screen_color='orange',
             ax_it=False,
             make_colorbar=True,
             abs_tol_for_site_relevancy=2,
             coord_system='lattice',
             cut_dist='auto',
             site_font=12,
             cbar_pos=[0.95, 0.3, 0.03, 0.5],
             label_size=14, tick_size=14, tick_len=4, tick_width=1.5):
    
    """
    Args:
        felfcar (os.PathLike) - path to ELFCAR
        axis_cut (str) - 'x', 'y', 'z' (cut normal to)
        savename (os.PathLike)
        vmin (float), vmax (float) - for colorbar
        thresh_for_slice_ID (str or float) - used by cut_dist
        cmap (str)
        nlevels (int) - for colorbar
        widths (3-element tuple) - (1,1,1) = unit cell
        in_screen_color (str) - color for sites that are behind screen
        out_of_screen_color (str) - color for sites that are in front of screen
        ax_it (bool) - return ax (True) or make fig (False)
        make_colorbar (bool)
        abs_tol_for_site_relevancy (float) - dist in Ang from "distance" to find sites for
        coord_system (str) - "lattice", "grid", "fractional"
        cut_dist (str) - "distance" from origin (can be automatically determined as high prob dens distance)
        site_font (int) - size for site labels
        
    """

    in_screen_color = tableau_colors()[in_screen_color]
    out_of_screen_color = tableau_colors()[out_of_screen_color]
    obj = ElfCar(felfcar)
    grid = obj.grid
    mg_obj = Elfcar.from_file(felfcar)
    s = mg_obj.structure
    data = mg_obj.data['total']
    if thresh_for_slice_ID == 'auto':
        thresh_for_slice_ID = 0.5*(np.max(data) - vmin)
    if vmax == 'max':
        vmax = np.max(data)
    lat = s.lattice
    a, b, c = lat.a, lat.b, lat.c
    if axis_cut == 'x':
        idx = 0
        x_lab = r'$\it{c}$'
        y_lab = r'$\it{b}$'
        cut_idx = 0
        x_idx = 2
        y_idx = 1
        cut_vector = a
        x_vector = c
        y_vector = b
    elif axis_cut == 'y':
        idx = 1
        x_lab = r'$\it{c}$'
        y_lab = r'$\it{a}$'
        cut_idx = 1
        x_idx = 2
        y_idx = 0
        cut_vector = b
        x_vector = c
        y_vector = a
    elif axis_cut == 'z':
        idx = 2
        x_lab = r'$\it{b}$'
        y_lab = r'$\it{a}$'
        cut_idx = 2
        x_idx = 1
        y_idx = 0
        cut_vector = c
        x_vector = b
        y_vector = a
    if cut_dist == 'auto':
        pts = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                for k in range(len(data[i][j])):
                    if data[i][j][k] > thresh_for_slice_ID:
                        pts.append((i,j,k))
        _s = [pt[idx] for pt in pts]
        common_ = max(set(_s), key=_s.count)
        distance = common_/grid[idx]
    else:
        distance = cut_dist
        
    print('distance = %.2f' % distance)

    X, Y, Z = obj.ax_contour(widths=widths,
                             axis_cut=axis_cut,
                             distance=distance)
    
    if vmax == 'max':
        vmax = np.max(Z)
        
        
    ticks = [vmin]
    v = vmin
    delta = 0.2
    while v <= vmax:
        v = v + delta
        ticks.append(np.round(v, 1))
    if not ax_it:
        fig = plt.figure()
        ax = plt.subplot(111)
        
    if coord_system == 'grid':
        X = X 
        Y = Y
    elif coord_system == 'fractional':
        X = [v/grid[x_idx] for v in X]
        Y = [v/grid[y_idx] for v in Y]
    elif coord_system == 'lattice':
        X = [v*x_vector/grid[x_idx] for v in X]
        Y = [v*y_vector/grid[y_idx] for v in Y]
    ax = plt.contourf(X, Y, Z, 
                      nlevels,
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vmax)    
    tol = abs_tol_for_site_relevancy/cut_vector
    relevant_sites = [site for site in s 
                      if abs(site.frac_coords[cut_idx]-distance) < tol]
    for site in relevant_sites:
        color = in_screen_color if site.frac_coords[cut_idx] <= distance else out_of_screen_color
        for i in range(widths[y_idx]):
            for j in range(widths[x_idx]):
                if coord_system == 'grid':
                    coords = (j+site.frac_coords[x_idx])*grid[x_idx], (i+site.frac_coords[y_idx])*grid[y_idx]
                elif coord_system == 'fractional':
                    coords = (j+site.frac_coords[x_idx]), (i+site.frac_coords[y_idx])
                elif coord_system == 'lattice':
                    coords = (j+site.frac_coords[x_idx])*x_vector, (i+site.frac_coords[y_idx])*y_vector
                ax = plt.text(coords[0], coords[1], 
                              site.species.formula[:-1], 
                              color=color, fontsize=site_font,
                              verticalalignment='center',
                              horizontalalignment='center')
    if coord_system == 'grid':
        unit = ''
    elif coord_system == 'fractional':
        unit = ' (fractional)'
    elif coord_system == 'lattice':
        unit = r'$\/(\AA)$' 
    ax = plt.xlabel(x_lab+unit)
    ax = plt.ylabel(y_lab+unit)
    
    ax = plt.plot([x_vector, x_vector], [y_vector, 0],
                  color='black', ls='--', lw=1)

    ax = plt.plot([x_vector, 0], [y_vector, y_vector],
                  color='black', ls='--', lw=1) 
    
    if make_colorbar:
        add_colorbar(fig, 
                     r'$ELF$', 
                     ticks,
                     cmap,
                     vmin, vmax,
                     cbar_pos,
                     label_size, tick_size, tick_len, tick_width)
        
    if not ax_it:
        plt.show()
        fig.savefig(savename)
        
def add_colorbar(fig, label, ticks, 
                 cmap, vmin, vmax, position, 
                 label_size, tick_size, tick_len, tick_width):
    import matplotlib as mpl
    norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
    cax = fig.add_axes(position)    
    cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label(label, fontsize=label_size)
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels(ticks)
    cb.ax.tick_params(labelsize=tick_size, length=tick_len, width=tick_width)
    return fig 

def plot_elf_diffs(felfcar1, felfcar2, axis_cut,
                   savename='tmp.png',
                   vmin=0.4, vmax=1,
                   thresh_for_slice_ID='auto',
                   cmap='bwr',
                   nlevels=100,
                   widths=(1,1,1),
                   in_screen_color='pink',
                   out_of_screen_color='orange',
                   ax_it=False,
                   make_colorbar=True,
                   abs_tol_for_site_relevancy=3,
                   coord_system='lattice',
                   cut_dist='auto',
                   site_font=12,
                   cbar_pos=[0.95, 0.3, 0.03, 0.5],
                   label_size=14, tick_size=14, tick_len=4, tick_width=1.5
                   ):
    
    """
    Args:
        felfcar (os.PathLike) - path to ELFCAR
        axis_cut (str) - 'x', 'y', 'z' (cut normal to)
        savename (os.PathLike)
        vmin (float), vmax (float) - for colorbar
        thresh_for_slice_ID (str or float) - used by cut_dist
        cmap (str)
        nlevels (int) - for colorbar
        widths (3-element tuple) - (1,1,1) = unit cell
        in_screen_color (str) - color for sites that are behind screen
        out_of_screen_color (str) - color for sites that are in front of screen
        ax_it (bool) - return ax (True) or make fig (False)
        make_colorbar (bool)
        abs_tol_for_site_relevancy (float) - dist in Ang from "distance" to find sites for
        coord_system (str) - "lattice", "grid", "fractional"
        cut_dist (str) - "distance" from origin (can be automatically determined as high prob dens distance)
        site_font (int) - size for site labels
        
    """

    in_screen_color = tableau_colors()[in_screen_color]
    out_of_screen_color = tableau_colors()[out_of_screen_color]
    
    
    obj1 = ElfCar(felfcar1)
    grid1 = obj1.grid
    mg_obj1 = Elfcar.from_file(felfcar1)
    s1 = mg_obj1.structure
    data1 = mg_obj1.data['total']
    
    obj2 = ElfCar(felfcar2)
    grid2 = obj2.grid
    mg_obj2 = Elfcar.from_file(felfcar2)
    s2 = mg_obj2.structure
    data2 = mg_obj2.data['total']  
    
    if grid1 != grid2:
        print('hmmmm grids arent the same. This could be tricky')
        raise NotImplementedError
        
    if s1.lattice != s2.lattice:
        print('hmmm lattices arent the same. This could be tricky')
        raise NotImplementedError
        
    grid = grid1
    data = np.zeros(data1.shape)
    for i in range(grid[0]):
        for j in range(grid[1]):
            for k in range(grid[2]):
                data[i][j][k] = data2[i][j][k] - data1[i][j][k]
    
            
    if thresh_for_slice_ID == 'auto':
        thresh_for_slice_ID = 0.5*(np.max(abs(data)) - 0)
    if vmax == 'max':
        vmax = np.max(data)
    if vmin == 'min':
        vmin = np.min(data)
    
    lat = s2.lattice
    a, b, c = lat.a, lat.b, lat.c
    if axis_cut == 'x':
        idx = 0
        x_lab = r'$\it{c}$'
        y_lab = r'$\it{b}$'
        cut_idx = 0
        x_idx = 2
        y_idx = 1
        cut_vector = a
        x_vector = c
        y_vector = b
    elif axis_cut == 'y':
        idx = 1
        x_lab = r'$\it{c}$'
        y_lab = r'$\it{a}$'
        cut_idx = 1
        x_idx = 2
        y_idx = 0
        cut_vector = b
        x_vector = c
        y_vector = a
    elif axis_cut == 'z':
        idx = 2
        x_lab = r'$\it{b}$'
        y_lab = r'$\it{a}$'
        cut_idx = 2
        x_idx = 1
        y_idx = 0
        cut_vector = c
        x_vector = b
        y_vector = a
    if cut_dist == 'auto':
        pts = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                for k in range(len(data[i][j])):
                    if abs(data[i][j][k]) > thresh_for_slice_ID:
                        pts.append((i,j,k))
        _s = [pt[idx] for pt in pts]
        common_ = max(set(_s), key=_s.count)
        distance = common_/grid[idx]
    else:
        distance = cut_dist
        
    print('distance = %.2f' % distance)

    X1, Y1, Z1 = obj1.ax_contour(widths=widths,
                                 axis_cut=axis_cut,
                                 distance=distance)
    X2, Y2, Z2 = obj2.ax_contour(widths=widths,
                                 axis_cut=axis_cut,
                                 distance=distance)
    
    if not X1.all() == X2.all():
        print('hmmm X grids arent the same. This could be tricky')
        raise NotImplementedError    
    if not Y1.all() == Y2.all():
        print('hmmm Y grids arent the same. This could be tricky')
        raise NotImplementedError   
    if not Z1.all() == Z2.all():
        print('hmmm Z grids arent the same. This could be tricky')
        raise NotImplementedError   
    
    X, Y = X2, Y2
    
    Z = np.zeros(Z2.shape)
    
    for i in range(len(Z2)):
        for j in range(len(Z1[i])):
            Z[i][j] = Z2[i][j] - Z1[i][j]
    
    if vmin == 'min':
        vmin = np.min(Z)
    if vmax == 'max':
        vmax = np.max(Z)
    if (vmin == 'sym') and (vmax == 'sym'):
        lim = np.max([abs(np.min(Z)), abs(np.max(Z))])
        vmin = -lim
        vmax = lim

    ticks = [np.round(vmin, 1)]
    v = vmin
    delta = 0.3
    while v <= vmax:
        v = v + delta
        ticks.append(np.round(v, 1))
    if not ax_it:
        fig = plt.figure()
        ax = plt.subplot(111)
        
    if coord_system == 'grid':
        X = X 
        Y = Y
    elif coord_system == 'fractional':
        X = [v/grid[x_idx] for v in X]
        Y = [v/grid[y_idx] for v in Y]
    elif coord_system == 'lattice':
        X = [v*x_vector/grid[x_idx] for v in X]
        Y = [v*y_vector/grid[y_idx] for v in Y]
    ax = plt.contourf(X, Y, Z, 
                      nlevels,
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vmax)    
    tol = abs_tol_for_site_relevancy/cut_vector
    
    s = s2
    
    relevant_sites = [site for site in s 
                      if abs(site.frac_coords[cut_idx]-distance) < tol]
    for site in relevant_sites:
        color = in_screen_color if site.frac_coords[cut_idx] <= distance else out_of_screen_color
        for i in range(widths[y_idx]):
            for j in range(widths[x_idx]):
                if coord_system == 'grid':
                    coords = (j+site.frac_coords[x_idx])*grid[x_idx], (i+site.frac_coords[y_idx])*grid[y_idx]
                elif coord_system == 'fractional':
                    coords = (j+site.frac_coords[x_idx]), (i+site.frac_coords[y_idx])
                elif coord_system == 'lattice':
                    coords = (j+site.frac_coords[x_idx])*x_vector, (i+site.frac_coords[y_idx])*y_vector
                ax = plt.text(coords[0], coords[1], 
                              site.species.formula[:-1], 
                              color=color, fontsize=site_font,
                              verticalalignment='center',
                              horizontalalignment='center')
    if coord_system == 'grid':
        unit = ''
    elif coord_system == 'fractional':
        unit = ' (fractional)'
    elif coord_system == 'lattice':
        unit = r'$\/(\AA)$' 
    ax = plt.xlabel(x_lab+unit)
    ax = plt.ylabel(y_lab+unit)
    
    ax = plt.plot([x_vector, x_vector], [y_vector, 0],
                  color='black', ls='--', lw=1)

    ax = plt.plot([x_vector, 0], [y_vector, y_vector],
                  color='black', ls='--', lw=1)    
    if make_colorbar:
        add_colorbar(fig, 
                     r'$ELF$', 
                     ticks,
                     cmap,
                     vmin, vmax,
                     cbar_pos,
                     label_size, tick_size, tick_len, tick_width)
        
    if not ax_it:
        plt.show()
        fig.savefig(savename)
    
        
def test():
    
    felfcar = '/Users/chrisbartel/Downloads/MgCr2S4_6_rs_pbe_elf.vasp'
    axis_cut = 'x'
    plot_elf(felfcar,
             axis_cut,
             vmin=0.4,
             thresh_for_slice_ID=0.1,
             cut_dist=0.58,
             abs_tol_for_site_relevancy=1.5,
             savename='one.png',
             widths=(2,2,2))
    
def test_diffs():
    felfcar2 = '/Users/chrisbartel/Downloads/MgCr2S4_6_rs_pbe_elf.vasp'
    felfcar1 = '/Users/chrisbartel/Downloads/MgCr2S4_0_rs_pbe_elf.vasp'
    axis_cut = 'x'
    
    out = plot_elf_diffs(felfcar1, felfcar2, axis_cut,
                         vmin='sym',
                         vmax='sym',
                         thresh_for_slice_ID=0.1,
                         widths=(2,2,2),
                         abs_tol_for_site_relevancy=1.5,
                         savename='diff.png')

    return out

def triple():
    
    fig = plt.figure(figsize=(5,12))
    felfcar1 = '/Users/chrisbartel/Downloads/MgCr2S4_0_rs_pbe_elf.vasp'
    felfcar2 = '/Users/chrisbartel/Downloads/MgCr2S4_6_rs_pbe_elf.vasp'
    axis_cut = 'x'

    ax1 = plt.subplot(311)
    
    ax1 = plot_elf_diffs(felfcar1, felfcar2, axis_cut,
                         vmin='sym',
                         vmax='sym',
                         thresh_for_slice_ID=0.1,
                         widths=(2,2,2),
                         abs_tol_for_site_relevancy=1.5,
                         savename='diff.png',
                         ax_it=True,
                         make_colorbar=False)  
    
    add_colorbar(fig, 
                 r'$ELF$', 
                 ticks=(0.4, 0.6, 1),
                 cmap='Blues',
                 vmin=0.4, vmax=1,
                 position=[0.95, 0.3, 0.03, 0.5],
                 label_size=14, tick_size=14, tick_len=4, tick_width=1.5)
    
    ax2 = plt.subplot(312)
    ax2 = plot_elf(felfcar2,
                       axis_cut,
                       vmin=0.4,
                       thresh_for_slice_ID=0.1,
                       cut_dist=0.58,
                       abs_tol_for_site_relevancy=1.5,
                       savename='one.png',
                       widths=(2,2,2),
                         ax_it=True,
                         make_colorbar=False)
    ax3 = plt.subplot(313)
    ax3 = plot_elf(felfcar1,
                       axis_cut,
                       vmin=0.4,
                       thresh_for_slice_ID=0.1,
                       cut_dist=0.58,
                       abs_tol_for_site_relevancy=1.5,
                       savename='one.png',
                       widths=(2,2,2),
                         ax_it=True,
                         make_colorbar=False)
    
    add_colorbar(fig, 
                 r'$ELF$', 
                 ticks=(0.4, 0.6, 1),
                 cmap='Blues',
                 vmin=0.4, vmax=1,
                 position=[0.95, 0.3, 0.03, 0.5],
                 label_size=14, tick_size=14, tick_len=4, tick_width=1.5)

    fig.savefig('triple.png')

def main():
    import os
    data_dir = '/Users/chrisbartel/Dropbox/postdoc/projects/Mg/cathodes/Cr4/data/storage/200228'
    cmpd = 'NaCrS2'
    _dir = '1_2_0_pbe_resp'
    felf2 = os.path.join(data_dir, cmpd, _dir, 'ELFCAR')
    
    plot_elf(felf2,
                'z',
                vmin=0.6,
                abs_tol_for_site_relevancy=3,
                savename='NaCrS2_anti.png',
                widths=(2,2,2))   
    
    _dir = '6_pbe_resp'
    felf1 = os.path.join(data_dir, cmpd, _dir, 'ELFCAR')

    plot_elf(felf1,
             'z',
             vmin=0.6,
                abs_tol_for_site_relevancy=3,
                savename='NaCrS2_6.png',
                widths=(2,2,2))
    
    """
    plot_elf_diffs(felf1, felf2, 'z',
                         vmin='sym',
                         vmax='sym',
                         thresh_for_slice_ID=0.1,
                         widths=(2,2,2),
                         abs_tol_for_site_relevancy=1.5,
                         savename='diff.png',
                         cut_dist=distance)
    """
    #test()
    #out = test_diffs()
    #triple()
    return 

if __name__ == '__main__':
    #set_rc_params()
    main()