#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:28:19 2020

@author: chrisbartel
"""

from compmatscipy.CompAnalyzer import CompAnalyzer
from pymatgen.analysis.phase_diagram import PhaseDiagram, GrandPotentialPhaseDiagram, PDEntry
from pymatgen.core.composition import Element
import numpy as np

def _convert_decomp_to_rxn(decomp):
    data = {CompAnalyzer(k.original_comp.formula).std_formula() : decomp[k] for k in decomp}
    decomp_rxn = ['_'.join([str(np.round(data[k], 4)), k]) for k in data]
    decomp_rxn = ' + '.join(decomp_rxn)
    return decomp_rxn

class GrandPotentialAnalysis(object):
    
    def __init__(self, compound_to_total_energy_per_atom, wi, mu_wi):
        
        self.compound_to_total_energy_per_atom = compound_to_total_energy_per_atom
        self.wi = wi
        self.mu_wi = mu_wi
        
    @property
    def _entries(self):
        d = self.compound_to_total_energy_per_atom
        return [PDEntry(c, d[c]*CompAnalyzer(c).num_atoms_in_formula()) for c in d]
        
    @property
    def _pd(self):
        return PhaseDiagram(self._entries)
    
    @property
    def _gppd(self):
        chempots = {Element(self.wi) : self.mu_wi}
        return GrandPotentialPhaseDiagram(self._pd.all_entries, chempots)
    
    @property
    def hull_output_data(self):
        """
        Args:
            compound (str) - formula to get data for
            
        Returns:
            hull_output_data but only for single compound
        """
        
        gppd = self._gppd
        entries = gppd.all_entries
        stable_entries = list(gppd.stable_entries)
        el_refs = list(gppd.el_refs.values())
        data = {}
        for e in entries+el_refs:
            print(e)
            stability = True if ((e in stable_entries) or (e in el_refs)) else False
            original_cmpd = CompAnalyzer(e.original_comp.formula).std_formula()
            print(original_cmpd)
            cmpd = CompAnalyzer(e.composition.formula).std_formula()
            print(cmpd)
            if CompAnalyzer(cmpd).num_els_in_formula == 1:
                phid = None
                decomp = None
                rxn = None
            else:
                phid = gppd.get_equilibrium_reaction_energy(e) if stability else gppd.get_e_above_hull(e)
                decomp = gppd.get_decomposition(e.composition)
                rxn = _convert_decomp_to_rxn(decomp)
            data[original_cmpd] = {'stability' : stability,
                                   'phid' : phid,
                                   'rxn' : rxn,
                                   'entry' : e.as_dict()}
            
        return data 
        
def main():
    
    d = {'Li1P3': -4.18752313125,
 'Li1P1': -4.18375594,
 'Li1P7': -5.13038443078125,
 'Li1P5': -5.01959429125,
 'Li3P1': -3.48098072625,
 'Li3P7': -4.71979694025,
 'Li2S1': -3.98754871,
 'Li1S4': -3.901453009,
 'Li1S1': -3.74074228,
 'P2S7': -4.534166235277778,
 'P4S5': -4.860789709444444,
 'P2S3': -4.80380366,
 'P4S3': -5.008038418392857,
 'P2S1': -4.742814926666667,
 'P1S1': -4.906938485625,
 'P4S7': -4.7744978115909085,
 'P4S9': -4.693347632115384,
 'P2S5': -4.656138745,
 'Li7P3S11': -4.399364608809524,
 'Li2P1S3': -4.3241097491666665,
 'Li3P1S4': -4.3896048959375,
 'Li7P1S6': -4.185985969107143,
 'Li48P16S61': -4.3651921816,
 'Li1': -1.90797265,
 'P1': -5.409794347857143,
 'S1': -3.464418535937499}

    space = 'Li_P_S'
    obj = GrandPotentialAnalysis(d, 'Li', -4.05)
    out = obj.hull_output_data
    #return obj
    return out, obj

if __name__ == '__main__':
    out, obj = main()