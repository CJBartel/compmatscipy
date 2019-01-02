# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:11:59 2018

@author: Chris
"""

import unittest
from compmatscipy.PullMP import PullMP

class UnitTestPullMP(unittest.TestCase):
    
    def dtest_groundstate_finder_and_compound_query(self):
        obj = PullMP('')
        fjson, tag, props, remove_polys, remake, write = '', 'Al2O3', ['pretty_formula', 'formation_energy_per_atom'], True, True, False
        data = obj.specific_query(fjson, tag, props, remove_polys, remake, write)
        self.assertEqual(len(data), 1)
        self.assertEqual(list(data.keys())[0], 'mp-1143')
        self.assertAlmostEqual(data['mp-1143']['formation_energy_per_atom'], -3.442, places=3)
        
    def dtest_space_query(self):
        obj = PullMP('')
        fjson, tag, props, remove_polys, remake, write = '', 'Cs-Br-F', ['pretty_formula', 'formation_energy_per_atom', 'band_gap'], True, True, False
        data = obj.specific_query(fjson, tag, props, remove_polys, remake, write)
        self.assertEqual(len(data), 4)
        
    def dtest_hull_query(self):
        obj = PullMP('')
        fjson, elements, props, remove_polys, remake, write = '', ['Cs', 'Br', 'F'], ['pretty_formula', 'formation_energy_per_atom', 'band_gap'], True, True, False
        data = obj.specific_hull_query(fjson, elements, props, remove_polys, remake, write)
        self.assertEqual(len(data), 9)
        
    def test_big_query(self):
        obj = PullMP('')
        fjson, criteria, props, remove_polys, remake, write = '', {'elements' : {'$in' : ['Si', 'O'], '$all' : ['Si', 'O']}, 'nelements' : {'$in' : [2]}}, ['pretty_formula', 'formation_energy_per_atom', 'band_gap'], True, True, False
        data = obj.big_query(fjson, criteria, props, remove_polys, remake, write)
        self.assertEqual(len(data), 8)
        
if __name__ == '__main__':
    unittest.main()