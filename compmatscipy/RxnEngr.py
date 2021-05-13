#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:53:07 2021

@author: chrisbartel
"""

from compmatscipy.handy_functions import read_json, write_json
from compmatscipy.TrianglePlots import get_label
from compmatscipy.ThermoEq import ThermoEq
from compmatscipy.CompAnalyzer import CompAnalyzer
import numpy as np
import os

def _incorp_dmu(data, dmus):
    R = 0.008314
    for T in data:
        for c in data[T]:
            initial = 96.485*CompAnalyzer(c).num_atoms_in_formula()*data[str(T)][c]['Ef']
            dGf = initial
            for el in dmus:
                n = CompAnalyzer(c).amt_of_el(el)
                adj = initial - (1/dmus[el]['n']*n*R*int(T)*np.log(dmus[el]['p']))
                dGf = adj
            data[T][c]['dGf'] = dGf / CompAnalyzer(c).num_atoms_in_formula() / 96.485

    return data

def _eVat_kJmol(E, formula):
    """
    Args:
        E (float) - energy in eV/atom
        formula (str)
        
    Returns:
        E (float) in kJ/mol
    """
    natoms = CompAnalyzer(formula).num_atoms_in_formula()
    return E*natoms*96.485

def _kJmol_eVat(E, formula):
    """
    Args:
        E (float) - energy in kJ/mol
        formula (str)
        
    Returns:
        E (float) in eV/atom
    """
    natoms = CompAnalyzer(formula).num_atoms_in_formula()
    return E/natoms/96.485 

class RxnEngr(object):
    
    def __init__(self,
                 data,
                 interface,
                 temp,
                 open_to=[],
                 nonsolids=['H', 'N', 'O', 'F', 'Cl'],
                 dmus={},
                 eq_thresh=0.001,
                 norm_closed=False,
                 eq_hull_thresh=0.0,
                 el_order_for_rxns=None):
        
        """
        Args:
            data (dict) : {T (str) : 
                           {formula (str) : 
                               {'Ef' : formation energy (eV/atom),
                                'Ed' : decomposition energy (eV/atom),
                                'rxn' : decomposition rxn (str),
                                'stability' : bool}}}
            interface (str) : looks like MOLES_FORMULA|MOLES_FORMULA|...
            temp (int) : T (K)
            open_to (list) : open elements (str) in the system 
            nonsolids (list) : non-solid formulae (str) to consider
            dmus (dict) : {el : {'p' : partial pressure (float, atm)
                                 'n' : atoms per mole (int)}}
            eq_thresh (float) : min mole fraction from eq to retainn
            norm_closed (False or int) : # moles of "closed" elements to normalize by
            eq_hull_thresh (float) : max metastability to consnider for eq analysis
            el_order_for_rxns (list) : order of elements for labels
        """
        self.data = _incorp_dmu(data, dmus)
        self.interface = interface
        self.open_to = open_to
        self.nonsolids = nonsolids
        self.eq_thresh = eq_thresh
        self.eq_hull_thresh = eq_hull_thresh
        self.norm_closed = norm_closed
        self.temp = temp
        self.el_order_for_rxns = el_order_for_rxns
        return
    
    @property
    def feed(self):
        """
        Returns:
            {formula (str) : # moles in feed (float)}
        """
        interface = self.interface
        n_species = interface.count('|')+1
        
        items = interface.split('|')
        
        return {CompAnalyzer(item.split('_')[1]).std_formula() : float(item.split('_')[0]) for item in items}
    
    @property
    def eq_input(self):
        """
        """
        T = self.temp
        hull_thresh = self.eq_hull_thresh
        data = self.data
        feed = self.feed
        nonsolids = self.nonsolids
        input_data = {c : {'dG' : data[str(T)][c]['dGf'],
                           'amt' : 0 if c not in feed else feed[c],
                           'phase' : 'solid'} for c in data[str(T)] if data[str(T)][c]['Ed'] <= hull_thresh}
        els = [CompAnalyzer(c).els for c in feed if feed[c] > 0]
        els = [j for i in els for j in i]
        els = sorted(list(set(els)))
        for el in els:
            input_data[el] = {'dG' : 0,
                              'amt' : 0 if el not in feed else feed[el],
                              'phase' : 'solid' if el not in nonsolids else 'nonsolid'}
        input_data = {c : input_data[c] for c in input_data if set(CompAnalyzer(c).els).issubset(set(els))}
            
        return input_data
    
    @property
    def eq(self):
        T = self.temp
        input_data = self.eq_input
        feed = self.feed
        thresh = self.eq_thresh
        eq = ThermoEq(input_data, T)
        results = eq.results
        results = {c : results[c] for c in results if results[c] >= thresh}
        N = np.sum(eq.solution[0])
        return {c : results[c]*N for c in results}
    
    @property
    def species(self):
        T = self.temp
        feed = self.feed
        results = self.eq
        species = {}
        for c in feed:
            if c in results:
                net_amt = feed[c] - results[c]
            else:
                net_amt = feed[c]
            side = 'right' if net_amt < 0 else 'left'
            amt = -net_amt if side == 'right' else net_amt
            species[c] = {'side' : side,
                          'amt' : amt}
            
        for c in results:
            if c not in species:
                species[c] = {'side' : 'right',
                              'amt' : results[c]}
                
        species = {c : species[c] for c in species if species[c]['amt'] > 0}
        
        return species
    
    @property
    def dGrxn(self):
        T = self.temp
        species = self.species
        data = self.data
        norm_closed = self.norm_closed
        feed = self.feed
        dGrxn = 0
        for s in species:
            if CompAnalyzer(s).num_els_in_formula == 1:
                continue
            if 'cmpd' not in species[s]:
                formula = CompAnalyzer(s).std_formula()
            else:
                formula = species[s]['cmpd']
            if species[s]['side'] == 'left':
                sign = -1
            elif species[s]['side'] == 'right':
                sign = 1
            else:
                raise ValueError
            coef = species[s]['amt']
            Ef = data[str(T)][formula]['dGf'] 
            Ef = _eVat_kJmol(Ef, formula)
            dGrxn += sign*coef*Ef
        moles_of_atoms = np.sum([CompAnalyzer(c).num_atoms_in_formula()*feed[c] for c in feed])
        if not norm_closed:
            return {'kJ' : dGrxn, 'eV/at' : dGrxn/moles_of_atoms/96.485}
        else:
            raise NotImplementedError
        """
        metals = 0
        if int(T) == 300:
            els = [CompAnalyzer(s).els for s in species]
            els = list(set([j for i in els for j in i]))
            for el in els:
                left = np.sum([CompAnalyzer(s).amt_of_el(el)*species[s]['amt'] for s in species if species[s]['side'] == 'left'])
                right = np.sum([CompAnalyzer(s).amt_of_el(el)*species[s]['amt'] for s in species if species[s]['side'] == 'right'])
                if right > left:
                    print(species)
                    print('needs input of %.2f %s' % ((right - left), el))
                elif left > right:
                    if el != 'O':
                        print(species)
                        print('makes xs of %.2f %s' % ((left-right), el))
                if el not in ['O']:
                    metals += right
            if metals != :
                print(species)
                print('AHHHH')
                print(metals)
        """

    @property
    def rxn(self):
        order = self.el_order_for_rxns
        T = self.temp
        species = self.species
        rxn = r''
        reactants = [s for s in species if species[s]['side'] == 'left']
        products = [s for s in species if species[s]['side'] == 'right']
        if not order:
            order = [CompAnalyzer(c).els for c in reactants + products]
            order = [j for i in order for j in i]
            order = sorted(list(set(order)))
        count = 0
        for r in reactants:
            amt = species[r]['amt']
            if amt == 0:
                continue
            if amt != 1:
                rxn += str(np.round(amt, 2))
            rxn += get_label(r, order)
            count += 1
            if count < len(reactants):
                rxn += '+'
        rxn += ' --> '
        count = 0
        for p in products:
            amt = species[p]['amt']
            if amt == 0:
                continue
            if amt != 1:
                rxn += str(np.round(amt, 2))
            rxn += get_label(p, order)
            count += 1
            if count < len(products):
                rxn += '+'
        rxn += r''
        
        rxn = rxn.replace('$', '').replace('{', '').replace('}', '').replace('_', '').replace('+', ' + ')
        return rxn
    
def make_data(remake=False):
    fjson = '_data_for_RxnEngr.json'
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    data_file = '/Users/chrisbartel/Dropbox/postdoc/projects/synthesis/paperdb/data/mp/MP_stability.json'
    data = read_json(data_file)
    
    my_els = ['Li', 'Co', 'Ba', 'Ti', 'Y', 'Ba', 'Cu', 'C',  'O']
    
    cmpds = sorted(list(data['0'].keys()))
    relevant_cmpds = [c for c in cmpds if set(CompAnalyzer(c).els).issubset(set(sorted(my_els)))]
    
    d = {T : {} for T in data}
    for c in relevant_cmpds:
        for T in d:
            if c in data[T]:
                d[T][c] = data[T][c]
    
    return write_json(d, fjson)

def main():
    d = make_data(False)
    obj = RxnEngr(d, '1', 300)
    print('INTERFACE:')
    print(obj.interface)
    print('EQ:')
    print(obj.eq)
    print('RXN:')
    print(obj.rxn)
    print('dGr:')
    print('%.2f eV/at' % obj.dGrxn['eV/at'])
    return d, obj

def LiNaMnCO():
    
    d = read_json('/Users/chrisbartel/Dropbox/postdoc/projects/synthesis/LiNaMnCO/data/hulls.json')
    d = {T : {c : d[c]['stability'][T] for c in d} for T in d['Mn1O2']['stability']}
    obj = RxnEngr(d, '1_Na2O|2_MnO', 1000, el_order_for_rxns=['Li', 'Co', 'Ba', 'Ti', 'Y', 'Ba', 'Cu', 'C',  'O'])
    print('INTERFACE:')
    print(obj.interface)
    print('EQ:')
    print(obj.eq)
    print('RXN:')
    print(obj.rxn)
    print('dGr:')
    print('%.2f eV/at' % obj.dGrxn['eV/at'])
    return d, obj

if __name__ == '__main__':
    d, obj = main()
    