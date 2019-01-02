# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 14:52:32 2018

@author: Chris
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:55:39 2018

@author: Chris
"""

import numpy as np
from itertools import combinations
import math
from compmatscipy.handy_functions import gcd
from compmatscipy.CompAnalyzer import CompAnalyzer
from compmatscipy.data import elemental_gibbs_energies_data, atomic_masses_data

class PredictGibbsEnergy(object):
    """
    Designed to take temperature-independent calculated or experimental data as input 
        and return the Gibbs formation energy at some temperature by applying
        a descriptor for the Gibbs energy of compounds
    """
    def __init__(self, 
                 formula,
                 H,
                 V):
        """
        Args:
            formula (str) - chemical formula (can be poorly formatted)
            H (float) - formation enthalpy at 0 or 298 K [eV/atom]
            V (float) - volume per atom of calculated structure [A**3/atom]
        """
        self.formula = formula        
        self.H = H
        self.V = V
        
    @property
    def els(self):
        """
        Returns:
            sorted list of elements (str)
        """
        return CompAnalyzer(self.formula).els
    
    @property
    def amts(self):
        """
        Returns:
            number of each element in the standardized formula (int) in the order of self.els
        """    
        return CompAnalyzer(self.formula).amts()
        
    @property
    def m(self):
        """
        Returns:
            reduced mass (float)
        """
        names = self.els
        nums = self.amts
        num_els = len(names)
        num_atoms = np.sum(nums)
        mass_d = atomic_masses_data()
        denom = (num_els - 1) * num_atoms
        if denom <= 0:
            print('descriptor should not be applied to unary compounds (elements)')
            return np.nan
        masses = [mass_d[el] for el in names]
        good_masses = [m for m in masses if not math.isnan(m)]
        if len(good_masses) != len(masses):
            for el in names:
                if math.isnan(mass_d[el]):
                    print('I dont have a mass for %s...' % el)
                    return np.nan
        else:
            pairs = list(combinations(names, 2))
            pair_red_lst = []
            for i in range(len(pairs)):
                first_elem = names.index(pairs[i][0])
                second_elem = names.index(pairs[i][1])
                pair_coeff = nums[first_elem] + nums[second_elem]
                pair_prod = masses[first_elem] * masses[second_elem]
                pair_sum = masses[first_elem] + masses[second_elem]
                pair_red = pair_coeff * pair_prod / pair_sum
                pair_red_lst.append(pair_red)
            return np.sum(pair_red_lst) / denom
            
    def Gd_sisso(self, T):
        """
        Args:
            T (int) - temperature [K]
        Returns:
            G^delta as predicted by SISSO-learned descriptor (float) [eV/atom]
        """
        if len(self.els) == 1:
            return 0
        else:
            m = self.m
            V = self.V
            return (-2.48e-4*np.log(V) - 8.94e-5*m/V)*T + 0.181*np.log(T) - 0.882
    
    def summed_Gi(self, T):
        """
        Args:
            T (int) - temperature [K]
        Returns:
            sum of the stoichiometrically weighted chemical potentials of the elements at T (float) [eV/atom]
        """
        Gi_d = elemental_gibbs_energies_data()
        names, nums = self.els, self.amts
        els_sum = 0
        for i in range(len(names)):
            el = names[i]
            if el not in Gi_d[str(T)]:
                return np.nan
            num = nums[i]
            Gi = Gi_d[str(T)][el]
            els_sum += num*Gi
        return els_sum
    
    def G(self, T, vol_per_atom=False):
        """
        Args:
            T (int) - temperature [K]
        Returns:
            Absolute Gibbs energy at T using SISSO-learned descriptor for G^delta (float) [eV/atom]
        """
        if len(self.els) == 1:
            Gi_d = elemental_gibbs_energies_data()            
            el = self.els[0]
            return Gi_d[str(T)][el]
        else:
            return self.H + self.Gd_sisso(T)
    
    def dG(self, T):
        """
        Args:
            T (int) - temperature [K]
        Returns:
            Gibbs formation energy at T using SISSO-learned descriptor for G^delta (float) [eV/atom]
        """
        if len(self.els) == 1:
            return 0.
        else:
            num_atoms_in_fu = np.sum(self.amts)
            return ((self.H + self.Gd_sisso(T))*num_atoms_in_fu - self.summed_Gi(T)) / num_atoms_in_fu

def main():
    return

if __name__ == '__main__':
    main()