#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:53:36 2020

@author: chrisbartel
"""


from compmatscipy.CompAnalyzer import CompAnalyzer
import numpy as np
from scipy.optimize import nnls
from compmatscipy.TrianglePlots import get_label

class ReactionAnalysis(object):
    
    def __init__(self, reactants, products,
                 el_open, num_closed):
        self.reactants = list(set(reactants))
        self.products = list(set([p for p in products if p not in reactants]))
        self.el_open = el_open
        self.num_closed = num_closed
        
    @property
    def all_els(self):
        r, p = self.reactants, self.products
        els = [CompAnalyzer(c).els for c in r+p]
        els = [j for i in els for j in i]
        return sorted(list(set(els)))
    
    @property
    def balance_els(self):
        els = self.all_els
        el_open = self.el_open
        if not el_open:
            return els
        return [el for el in els if el != el_open]
    
    @property
    def A(self):
        r, p = self.reactants, self.products
        balance_els = self.balance_els
        A = np.zeros(shape=(len(r)+len(p), len(balance_els)+len(p)))
        A = A.T
        for i in range(len(balance_els)):
            count = 0
            for j in range(len(r)+len(p)):
                count += 1
                if count <= len(r):
                    sign = 1
                    cmpd = r[j]
                else:
                    sign = -1
                    cmpd = p[j-len(r)]
                A[i, j] = sign*CompAnalyzer(cmpd).amt_of_el(balance_els[i])
        line = [0 for i in range(len(r))]
        for j in range(len(p)):
            line.append(np.sum([CompAnalyzer(p[j]).amt_of_el(el) for el in balance_els]))
        A[-1] = line
        return np.array(A)
    
    @property
    def b(self):
        r, p = self.reactants, self.products
        balance_els = self.balance_els
        b = np.zeros(shape=(len(balance_els)+len(p), 1))
        b[-1] = self.num_closed
        b = [b[i][0] for i in range(len(b))]
        return np.array(b)
    
    @property
    def solution(self):
        A, b = self.A, self.b
        return nnls(A, b)
        
    @property
    def species(self):
        coefs = {}
        r, p = self.reactants, self.products
        components = r + p
        solution = self.solution
    
        for i in range(len(components)):
            coefs[components[i]] = solution[0][i]
        
        species = {reac : {'side' : 'left', 'amt' : coefs[reac]} for reac in r}
        for prod in p:
            species[prod] = {'side' : 'right',
                          'amt' : coefs[prod]} 
        return species
    
    def fancy_reaction_string(self, order):
        species = self.species
        rxn = r''
        reactants = [s for s in species if species[s]['side'] == 'left']
        products = [s for s in species if species[s]['side'] == 'right']
        
        count = 0
        for r in reactants:
            amt = species[r]['amt']
            if amt == 0:
                continue
            if amt != 1:
                rxn += str(amt)
            rxn += get_label(r, order)
            count += 1
            if count < len(reactants):
                rxn += '+'
        rxn += r'$\rightarrow$'
        count = 0
        for p in products:
            amt = species[p]['amt']
            if amt == 0:
                continue
            if amt != 1:
                rxn += str(amt)
            rxn += get_label(p, order)
            count += 1
            if count < len(products):
                rxn += '+'
        rxn += r''
        return rxn 
    
    def reaction_string(self, order):
        species = self.species
        rxn = r''
        reactants = [s for s in species if species[s]['side'] == 'left']
        products = [s for s in species if species[s]['side'] == 'right']
        
        count = 0
        for r in reactants:
            count += 1
            amt = species[r]['amt']
            amt = np.round(amt, 3)
            if amt < 1e-4:
                continue
            if amt != 1:
                rxn += str(amt)
            rxn += '_'
            rxn += r
            if count < len(reactants):
                rxn += ' + '
        count = 0
        rxn += ' ---> '
        for p in products:
            count += 1
            amt = species[p]['amt']
            amt = np.round(amt, 3)
            if amt <  1e-4:
                continue
            if amt != 1:
                rxn += str(amt)
            rxn += '_'
            rxn += p
            if count < len(products):
                rxn += ' + '
        return rxn 
    
    @property
    def diagnostics(self):
        species = self.species
        el_open = self.el_open
        num_closed = self.num_closed
        closed = 0
        els = [CompAnalyzer(s).els for s in species]
        els = list(set([j for i in els for j in i]))
        d = {'xs in' : [],
             'xs out' : [],
             'norm' : None,
             'bad' : False}
        for el in els:
            left = np.sum([CompAnalyzer(s).amt_of_el(el)*species[s]['amt'] for s in species if species[s]['side'] == 'left'])
            right = np.sum([CompAnalyzer(s).amt_of_el(el)*species[s]['amt'] for s in species if species[s]['side'] == 'right'])
            if right > left+0.02:
                if el != el_open:
                    d['bad'] = True
                d['xs in'].append({el : right-left})
            elif left > right+0.02:
                if el != el_open:
                    d['xs out'].append({el : left-right})
            if el not in [el_open]:
                closed += right
        norm = closed
        d['norm'] = norm
        if np.round(norm, 2) != np.round(num_closed, 2):
            d['bad'] = True
        return d