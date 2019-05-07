# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 21:45:30 2018

@author: Chris
"""

import os, unittest
from compmatscipy.HullAnalysis import GetHullInputData, AnalyzeHull
from compmatscipy.handy_functions import read_json, write_json
from compmatscipy.CompAnalyzer import CompAnalyzer
import numpy as np

test_dir = os.path.join(os.path.dirname(__file__))
test_data_dir = os.path.join(test_dir, 'test_data')

class UnitTestHullAnalysis(unittest.TestCase):
    
    def test_hull_analysis_against_MP(self):
        """
        Uses an MP query of the Al-Ca-Mg-O-Si space
        Compares my decomp energies to theirs
            Changes mine to 0 for stable compounds to force agreement with theirs
        """        
        d = read_json(os.path.join(test_data_dir, 'MP_Ca-Mg-Al-Si-O.json'))
        data_for_hulls = {CompAnalyzer(d[k]['pretty_formula']).std_formula() : {'E' : d[k]['formation_energy_per_atom']} for k in d if d[k]['is_min_ID'] == True}
        obj = GetHullInputData(data_for_hulls, 'E')
        fjson = os.path.join(test_data_dir, '_'.join(['hulls', 'Ca-Mg-Al-Si-O.json']))
        hull_data = obj.hull_data(fjson, True)
        for chemical_space in hull_data:
            obj = AnalyzeHull(hull_data, chemical_space)
            hull_results = obj.hull_output_data
        for ID in d:
            if d[ID]['is_min_ID'] == 1:
                cmpd = CompAnalyzer(d[ID]['pretty_formula']).std_formula()
                Ed_MP = d[ID]['e_above_hull']
                Ed_me = hull_results[cmpd]['Ed']
                # MP shows all stables as Hd = 0
                if Ed_me < 0:
                    Ed_me = 0
                self.assertAlmostEqual(Ed_me, Ed_MP, places=3)

    def test_hull_analysis_against_old_analyzer(self):
        """
        Compares my decomp energies to those I used for npj paper
            Changes mine to Ed = Ef if Ef > 0 because this was approach for npj
        """
        d = read_json(os.path.join(test_data_dir, 'SCAN_Hs_from_npj.json'))
        d['Br3Cr1'] = {'H' : -1.128072912,	
                       'Hd' : -0.35554559}
        obj = GetHullInputData(d, 'H')
        fjson = os.path.join(test_data_dir, '_'.join(['hulls', 'npj.json']))
        hull_data = obj.hull_data(fjson, True)
        counted_cmpds = []
        for chemical_space in hull_data:
            obj = AnalyzeHull(hull_data, chemical_space)
            hull_results = obj.hull_output_data
            for cmpd in hull_results:
                if cmpd in d:
                    Ed_current = hull_results[cmpd]['Ed']
                    # Old script treated Ef > 0 as decomp into elements
                    Ef_current = hull_results[cmpd]['Ef']
                    if Ef_current > 0:
                        Ed_current = Ef_current
                    Ed_old = d[cmpd]['Hd']
                    self.assertAlmostEqual(Ed_current, Ed_old, places=3)
                    counted_cmpds.append(cmpd)
        self.assertEqual(len(set(counted_cmpds)), len(d))
                
if __name__ == '__main__':
    unittest.main()                
