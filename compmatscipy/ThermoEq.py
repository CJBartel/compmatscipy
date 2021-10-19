#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:58:26 2019

@author: chrisbartel
"""

import os
import numpy as np
from scipy.optimize import fmin_slsqp, minimize
from compmatscipy.CompAnalyzer import CompAnalyzer
import math
import time

class ThermoEq(object):
    
    def __init__(self, input_data, temp, excluded_from_solution=[]):
        """
        Args:
            
            input_data (dict) - {formula (str) : {'dG' : Gibbs formation energy (eV/atom),
                                                  'amt' : molar feed (mol),
                                                  'phase' : 'solid' or 'nonsolid'}}
            temp (int) - temperature in Kelvin 
        Returns:
            dictionary of {formula (str) : formation energy (float)}
            
        FORMATION ENERGY IN eV/atom!
        """
        for formula in input_data:
            dG_eV = input_data[formula]['dG']
            dG_kJ = self._eV_to_kJ(formula, dG_eV)
            input_data[formula]['dG'] = dG_kJ
        self.input_data = input_data
        self.temp = temp
        self.excluded = excluded_from_solution
        
    @property
    def _sorted_formulas(self):
        input_data = self.input_data
        return sorted(list(input_data.keys()))
    
    @staticmethod
    def _eV_to_kJ(formula, eV_at):
        return 96.485 * eV_at * CompAnalyzer(formula).num_atoms_in_formula()
    
    @property
    def _relevant_els(self):
        cmpds = self.input_data.keys()
        els = [CompAnalyzer(c).els for c in cmpds]
        return sorted(list(set([j for i in els for j in i])))
    
    @property
    def A(self):
        input_data = self.input_data
        relevant_els = self._relevant_els
        sorted_formulas = self._sorted_formulas
        formulas = [f for f in sorted_formulas if f not in self.excluded]
        A = np.zeros((len(relevant_els), len(formulas)))
        for row in range(len(relevant_els)):
            el = relevant_els[row]
            for col in range(len(formulas)):
                formula = formulas[col]
                A[row,col] = CompAnalyzer(formula).amt_of_el(el)
        return A
    
    @property
    def b(self):
        input_data = self.input_data
        relevant_els = self._relevant_els
        b = np.zeros((len(relevant_els)))
        sorted_formulas = self._sorted_formulas
        for i in range(len(relevant_els)):
            el = relevant_els[i]
            amt = 0
            for formula in sorted_formulas:
                amt += input_data[formula]['amt'] * CompAnalyzer(formula).amt_of_el(el)
            b[i] = amt
        return b
    
    @property
    def Gjo(self):
        input_data = self.input_data
        sorted_formulas = self._sorted_formulas
        formulas = [f for f in sorted_formulas if f not in self.excluded]
        return [input_data[formula]['dG'] for formula in formulas]
    
    @property
    def solution(self):
        T = self.temp
        input_data = self.input_data
        Gjo, A, b = self.Gjo, self.A, self.b
        R = 0.008314 # kJ/mol/K
        sorted_formulas = self._sorted_formulas
        formulas = [f for f in sorted_formulas if f not in self.excluded]
        n0 = [input_data[formula]['amt']+1e-8 for formula in formulas]
        bounds = [(1e-12, 1e1*np.sum(n0)) for i in n0]
        def func(nj):
            nj = np.array(nj)
            Enj = np.sum([nj[i] for i in range(len(nj)) if input_data[formulas[i]]['phase'] == 'nonsolid'])
            if Enj != 0:
                Gj =  [(Gjo[i] + R*T*np.log(nj[i] / Enj)) if ((input_data[formulas[i]]['phase'] == 'nonsolid') and (nj[i]/Enj > 0)) else (Gjo[i]) for i in range(len(nj))]
            else:
                Gj =  [Gjo[i] for i in range(len(nj))]
    #        Gj = [(Gjo[i] / (0.008314 * T) + np.log(nj[i] / Enj)) for i in range(len(nj))]
            return np.dot(nj, Gj)
        
        def ec1(nj):
            return np.dot(A, nj) - b
        
        niter = 1000
        out, fx, its, imode, smode = fmin_slsqp(func, n0, f_eqcons=ec1, bounds=bounds, iter=niter, acc=1e-4, iprint=2, full_output=True)
        if imode != 0:
            print('trying softer\n')
            out, fx, its, imode, smode = fmin_slsqp(func, n0, f_eqcons=ec1, bounds=bounds, iter=niter, acc=1e-3, iprint=0, full_output=True)
            if imode != 0:
                print('trying softer\n')
                out, fx, its, imode, smode = fmin_slsqp(func, n0, f_eqcons=ec1, bounds=bounds, iter=niter, acc=5e-3, iprint=0, full_output=True)
                if imode != 0:
                    print('trying softer\n')
                    out, fx, its, imode, smode = fmin_slsqp(func, n0, f_eqcons=ec1, bounds=bounds, iter=1000, acc=1e-2, iprint=0, full_output=True)
                    if imode != 0:
                        print('trying softer\n')
                        out, fx, its, imode, smode = fmin_slsqp(func, n0, f_eqcons=ec1, bounds=bounds, iter=5000, acc=5e-2, iprint=0, full_output=True)                        
                        if imode != 0:
                            print('\n\n\n\n\nAHHHHHHHHHHH!!!!!\n\n\n\n\n\n')
                            return [np.nan for i in sorted_formulas]
        return out, fx
    
    @property
    def results(self):
        T = self.temp
        sorted_formulas = self._sorted_formulas
        formulas = [f for f in sorted_formulas if f not in self.excluded]
        N, fx = self.solution
        r = {}
        for s, n in zip(list(formulas), N):
            r[s] = n/np.sum(N)
        return r
    
    @property
    def minimized_G(self):
        return self.solution[1]
    
def main():
    data = {'Al2O3' : {'phase' : 'solid',
                       'amt' : 0,
                       'dG' : -3},
            'Al1N1' : {'phase' : 'solid',
                       'amt' : 3,
                       'dG' : -2},
            'Al1' : {'phase' : 'solid',
                     'amt' : 6,
                     'dG' : 0},
            'O2' : {'phase' : 'nonsolid',
                     'amt' : 2.5,
                     'dG' : 0}, 
            'H2' : {'phase': 'nonsolid',
                    'amt': 0,
                    'dG' : -1000}}   
                     
    obj = ThermoEq(data, 1000, [])
    print(obj.results)
    return obj

if __name__ == '__main__':
    obj = main()