#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:39:33 2019

@author: chrisbartel
"""
import matplotlib.pyplot as plt
#import ternary
from compmatscipy.plotting_functions import tableau_colors, set_rc_params
from compmatscipy.handy_functions import read_json
import numpy as np
import os
import matplotlib as mpl
from compmatscipy.CompAnalyzer import CompAnalyzer
import itertools
from compmatscipy.HullAnalysis import GetHullInputData, AnalyzeHull
from compmatscipy.ThermoEq import ThermoEq
import math

class TrianglePD(object):
    
    def __init__(self, input_data, els):
        """
        Args:
            input_data (dict) - stability data (dict) for all compounds in the specified chemical space
                {compound (str) : {'Ef' : formation energy (float),
                                   'Ed' : decomposition energy (float),
                                   'rxn' : decomposition reaction (str),
                                   'stability' : stable (True) or unstable (False)}}
                HullAnalysis.AnalyzeHull.hull_output_data
            els (list) - list of els (str) for triangle
                els go CC from right to top to left
                (S, Ag, Cr) would be right S, top Ag, left Cr
        
        Returns:
            input_data
            els
            hull space (str) 
        """
        self.input_data = input_data
        self.els = els
        self.space = '_'.join(sorted(els))
    
    def _make_triangle(self, tri_lw=1.5):
        """
        Args:
            tri_lw (float) - thickness for triangle border
        
        Returns:
            plots the triangle boundary (note this plots anyway, but allows user to specify thickness)
        """
        lines = triangle_boundary()
        for l in lines:
            ax = plt.plot(lines[l]['x'], lines[l]['y'],
                          color='black',
                          lw=tri_lw,
                          zorder=1)
        return ax
    
    def _remove_spines_and_ticks(self):
        """
        Args:
            ax (axis object)
        
        Returns:
            removes 2D spines and ticks
        """
        for spine in ['bottom', 'left', 'top', 'right']:
            ax = plt.gca().spines[spine].set_visible(False)
        ax = plt.gca().xaxis.set_ticklabels([])
        ax = plt.gca().yaxis.set_ticklabels([])
        ax = plt.gca().tick_params(bottom=False, top=False, left=False, right=False)
        return ax
    
    def _add_points(self, pts_for_els=True):
        """
        Args:
            pts_for_els (bool) - if True, put points on corners
        
        Returns:
            adds compound+element points to triangle
        """
        cmpd_params = params()
        data = self.input_data
        els = self.els
        for cmpd in data:
            if not pts_for_els and (len(CompAnalyzer(cmpd).els) == 1):
                continue
            pt = [CompAnalyzer(cmpd).fractional_amt_of_el(el) for el in els]
            pt = triangle_to_square(pt)
            stability = 'stable' if data[cmpd]['stability'] else 'unstable'
            color, marker = [cmpd_params[stability][k] for k in ['c', 'm']]
            ax = plt.scatter(pt[0], pt[1], 
                             color='white', 
                             marker=marker,
                             facecolor='white',
                             edgecolor=color,
                             zorder=2)
        return ax
    
    def _add_lines(self, tie_lw=1.5):
        """
        Args:
            tie_lw (float) - thickness of tie lines
        
        Returns:
            adds tie lines to triangle
        """
        els = self.els
        hull_data = GetHullInputData(self.input_data, 'Ef').hull_data(remake=True)
        obj = AnalyzeHull(hull_data, self.space)
        simplices = obj.hull_simplices
        lines = uniquelines(simplices)
        sorted_cmpds = obj.sorted_compounds
        stable_cmpds = obj.stable_compounds
        print(stable_cmpds)
        line_data = {}
        for l in lines:
            cmpds = (sorted_cmpds[l[0]], sorted_cmpds[l[1]])
            if (cmpds[0] not in stable_cmpds) or (cmpds[0] not in stable_cmpds):
                continue
            line_data[l] = {'cmpds' : cmpds}
            line_data[l]['pts'] = tuple([cmpd_to_pt(cmpd, els) for cmpd in cmpds])
            x = (line_data[l]['pts'][0][0], line_data[l]['pts'][1][0])
            y = (line_data[l]['pts'][0][1], line_data[l]['pts'][1][1])
            ax = plt.plot(x, y, zorder=1, lw=tie_lw, color='black')
            
    def _label_els(self, 
                   el_label_size=18,
                   left_el_pos=(-0.02, -0.05),
                   right_el_pos=(1.02, -0.05),
                   top_el_pos=(0.5, 0.89)):
        """
        Args:
            el_label_size (int) - font size for corner labels
            *el_pos (tuple) - (x, y) for where to put corner labels
        
        Returns:
            adds element labels on the corners
        """

        els = self.els
        ax = plt.text(right_el_pos[0], right_el_pos[1], els[0], fontsize=el_label_size, horizontalalignment='left')
        ax = plt.text(top_el_pos[0], top_el_pos[1], els[1], fontsize=el_label_size, horizontalalignment='center')
        ax = plt.text(left_el_pos[0], left_el_pos[1], els[2], fontsize=el_label_size, horizontalalignment='right')
        return ax
    
    def _label_pts(self,
                   el_order_for_label,
                   pt_label_size=16,
                   label_unstable=False,
                   specify_labels={},
                   only_certain_labels=[],
                   skip_certain_labels=[]):
        """
        Args:
            el_order_for_label (list) - ordered list of elements for labels
            pt_label_size (int) - font size for cmpd labels
            label_unstable (bool) - whether or not to label unstable points
            specify_labels (dict) - specific positions for certain compounds
                                    {cmpd : {'xpos' : ,
                                             'ypos' : ,
                                             'xalign' : ,
                                             'yalign' : }}
            only_certain_labels (list) - enforce only these compounds are labeled
            skip_certain_labels (list) - skip these labels
        
        Returns:
            labels the compounds
        """
        input_data = self.input_data
        stable_cmpds = [c for c in input_data if input_data[c]['stability'] if len(CompAnalyzer(c).els) > 1]
        unstable_cmpds = [c for c in input_data if not input_data[c]['stability']]
        if label_unstable:
            cmpds_to_label = stable_cmpds + unstable_cmpds
        else:
            cmpds_to_label = stable_cmpds
        if len(only_certain_labels) > 0:
            cmpds_to_label = only_certain_labels
        cmpds_to_label = [c for c in cmpds_to_label if c not in skip_certain_labels]
        count = 0
        for cmpd in cmpds_to_label:
            pt = cmpd_to_pt(cmpd, self.els)
            tri = [CompAnalyzer(cmpd).fractional_amt_of_el(el) for el in self.els]
            label = get_label(cmpd, el_order_for_label)
            if cmpd in specify_labels:
                xpos, xalign, ypos, yalign = [specify_labels[cmpd][k] for k in ['xpos', 'xalign', 'ypos', 'yalign']]
            elif tri[1] in (0, 1):
                xpos = pt[0]
                xalign = 'center'
                ypos = -0.02
                yalign = 'top'

            elif tri[0] in (0, 1):
                xpos = pt[0]-0.02
                xalign = 'right'
                ypos = pt[1]
                yalign = 'center'
                
            elif tri[2] in (0, 1):
                xpos = pt[0]+0.02
                xalign = 'left'
                ypos = pt[1]
                yalign = 'center'
            else:
                if count % 2:
                    x_sign = 1
                    y_sign = -1
                    xalign = 'left'
                    yalign = 'top'
                else:
                    x_sign, y_sign, xalign, yalign = -1, 1, 'right', 'bottom'
                xpos = pt[0]+x_sign*0.01
                ypos = pt[1]+y_sign*0.01
                count += 1
            if cmpd in stable_cmpds:
                color = params()['stable']['c']
            else:
                color = params()['unstable']['c']
            ax = plt.text(xpos, ypos, label, 
                          horizontalalignment=xalign, 
                          verticalalignment=yalign, 
                          fontsize=pt_label_size,
                          color=color, zorder=100)
            
    @property
    def _mask_outside(self):
        t1 = plt.Polygon(np.array([[0,0], [0, np.sqrt(3)/2], [0.5, np.sqrt(3)/2]]), color='white')
        ax = plt.gca().add_patch(t1)
        t1 = plt.Polygon(np.array([[1,0], [1, np.sqrt(3)/2], [0.5, np.sqrt(3)/2]]), color='white')
        ax = plt.gca().add_patch(t1)
        ax = plt.ylim([-0.02, np.sqrt(3)/2])
        return ax
    
    def ax3d(self,
             el_order_for_label=False,
             remove_spines=True,
             label_els=True,
             label_pts=True,
             label_unstable=True,
             tri_lw=1.5,
             tie_lw=1.0,
             el_label_size=18,
             left_el_pos=(-0.02, -0.05),
             right_el_pos=(1.02, -0.05),
             top_el_pos=(0.5, 0.9),
             pt_label_size=16,
             specify_labels={},
             only_certain_labels=[],
             skip_certain_labels=[],
             show_lines=True):
        """
        Args:
            remove_spines (bool) - remove x, y spines or not
            label_els (bool) - label corners or not
            label_pts (bool) - label compounds or not
            SEE methods for other Args (should be **kwargs???)
        
        Returns:
            axis object with triangular phase diagram
        """
            
        ax = self._make_triangle(tri_lw)
        if remove_spines:
            ax = self._remove_spines_and_ticks()
        ax = self._add_points()
        if show_lines:
            ax = self._add_lines(tie_lw)
        if label_els:
            ax = self._label_els( 
                             el_label_size,
                             left_el_pos,
                             right_el_pos,
                             top_el_pos)
        if label_pts:
            if not el_order_for_label:
                el_order_for_label = self.els[::-1]
            ax = self._label_pts(
                                 el_order_for_label,
                                 pt_label_size,
                                 label_unstable,
                                 specify_labels,
                                 only_certain_labels,
                                 skip_certain_labels)
        return ax
    
    def color_matrix(self, npts=20):
        input_data = self.input_data
        els = self.els
        xleft = np.linspace(0, 1, npts)
        xcenter, xright = xleft, xleft
        data = []
        count = 0
        for i in range(npts):
            for j in range(npts):
                for k in range(npts):
                    feed = {els[2] : xleft[i],
                            els[0] : xright[j],
                            els[1] : xcenter[k]}
                    if np.sum(list(feed.values())) != 1:
                        continue
                    value = _compute_energy(input_data, feed)
                    if math.isnan(value):
                        continue
                    data.append([i, j, k, value])
                    count += 1
        return np.array(data)
    
    def add_color(self, vmin=False, vmax=False,
                  cmap='plasma_r',
                  nlevels=100, npts=20,
                  alpha=1):
        input_data = self.input_data
        if not vmin:
            vmin = np.min([input_data[c]['Ef'] for c in input_data])
            vmin = np.round(vmin, 2)
        vmax = 0 if not vmax else vmax
        matrix = self.color_matrix(npts)
        a = matrix[:,0]
        b = matrix[:,1]
        c = matrix[:,2]
        v = matrix[:,-1]
        x = 0.5 * ( 2.*b+c ) / ( a+b+c )
        y = 0.5*np.sqrt(3) * c / (a+b+c)
        T = mpl.tri.Triangulation(x, y)    
        ax = plt.tricontourf(x,y,T.triangles,v,
                             levels=nlevels,
                             vmin=vmin, vmax=vmax,
                             cmap=cmap,
                             alpha=alpha)
        return ax, vmin, vmax, cmap, alpha
    
def _compute_energy(input_data, feed):
    data = {}
    for c in input_data:
        if c not in feed:
            amt = 0
        else:
            amt = feed[c]
        data[c] = {'dG' : input_data[c]['Ef'],
                   'amt' : amt,
                   'phase' : 'solid'}
    for el in feed:
        data[el] = {'dG' : 0,
                   'amt' : feed[el],
                   'phase' : 'solid'}
    obj = ThermoEq(data, 300)
    minG = obj.minimized_G
    return minG/96.485
    
def triangle_to_square(pt):
    """
    Args:
        pt (tuple) - (left, top, right) triangle point
                e.g., (1, 0, 0) is left corner
                e.g., (0, 1, 0) is top corner
    
    Returns:
        pt (tuple) - (x, y) same point in 2D space
    """
    converter = np.array([[1, 0], [0.5, np.sqrt(3) / 2]])
    new_pt = np.dot(np.array(pt[:2]), converter)
    return new_pt.transpose()

def square_to_triangle(x, y):
    """
    Args:
        pt (tuple) - (x, y) planar point in 2D space
    
    Returns:
        pt (tuple) - (left, top, right) triangle point
                e.g., (1, 0, 0) is left corner
                e.g., (0, 1, 0) is top corner
    """
    b = y*2/np.sqrt(3)
    a = x-0.5*b
    c = 1 - a - b
    return (a, b, c)

def triangle_boundary():
    """
    Args:
        
    Returns:
        the boundary of a triangle where each axis goes 0 to 1
    """
    corners = [(0,0,0), (0,1,0), (1,0,0)]
    corners = [triangle_to_square(pt) for pt in corners]
    data = dict(zip(['left', 'top', 'right'], corners))
    lines = {'bottom' : {'x' : (data['left'][0], data['right'][0]),
                         'y' : (data['left'][1], data['right'][1])},
             'left' : {'x' : (data['left'][0], data['top'][0]),
                       'y' : (data['left'][1], data['top'][1])},
             'right' : {'x' : (data['top'][0], data['right'][0]),
                       'y' : (data['top'][1], data['right'][1])} }
    return lines             

def params():
    """
    Args:
        
    Returns:
        just colors for stable and unstable points
    """
    return {'stable' : {'c' : tableau_colors()['blue'],
                        'm' : 'o'},
            'unstable' : {'c' : tableau_colors()['red'],
                          'm' : '^'}}    

def get_label(cmpd, els):
    """
    Args:
        cmpd (str) - chemical formula
        els (list) - ordered list of elements (str) as you want them to appear in label
    
    Returns:
        neatly formatted chemical formula label
    """
    label = r'$'
    for el in els:
        amt = CompAnalyzer(cmpd).amt_of_el(el)
        if amt == 0:
            continue
        label += el
        if amt == 1:
            continue
        label += '_{%s}' % amt
    label += '$'
    return label

def uniquelines(q):
    """
    Given all the facets, convert it into a set of unique lines.  Specifically
    used for converting convex hull facets into line pairs of coordinates.

    Args:
        q: A 2-dim sequence, where each row represents a facet. E.g.,
            [[1,2,3],[3,6,7],...]

    Returns:
        setoflines:
            A set of tuple of lines.  E.g., ((1,2), (1,3), (2,3), ....)
    """
    setoflines = set()
    for facets in q:
        for line in itertools.combinations(facets, 2):
            setoflines.add(tuple(sorted(line)))
    return setoflines

def cmpd_to_pt(cmpd, els):
    """
    Args:
        cmpd (str) - chemical formula
        els (list) - ordered list of elements (str) in triangle (right, top, left)
    
    Returns:
        (x, y) for compound
    """
    tri = [CompAnalyzer(cmpd).fractional_amt_of_el(el) for el in els]
    return triangle_to_square(tri)

def add_colorbar(fig, label, ticks, 
                 cmap, vmin, vmax, position, 
                 label_size, tick_size, tick_len, tick_width,
                 alpha=1):
    norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
    cax = fig.add_axes(position)    
    cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=norm, orientation='vertical', alpha=alpha)
    cb.set_label(label, fontsize=label_size)
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels(ticks)
    cb.ax.tick_params(labelsize=tick_size, length=tick_len, width=tick_width)
    return fig 