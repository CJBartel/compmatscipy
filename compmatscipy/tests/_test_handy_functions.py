# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:35:33 2018

@author: Chris
"""

import unittest, sys, os
from compmatscipy.handy_functions import gcd, list_of_dicts_to_dict, H_from_E

class UnitTestHandyFunctions(unittest.TestCase):
    
    def test_gcd(self):
        a, b = (9, 60)
        answer = 3
        self.assertEqual(gcd(a,b), answer)
        
    def test_gcd_reverse(self):
        a, b = (60, 9)
        answer = 3
        self.assertEqual(gcd(a,b), answer)    
    
    def test_list_to_dict(self):
        l = [{'age' : 27,
              'name' : 'Chris',
              'species' : 'human'},
             {'age' : 5,
              'name' : 'Keema',
              'species' : 'dog'}]
        major_key, other_keys = 'name', ['age', 'species']
        answer = {'Keema' : {'age' : 5,
                             'species' : 'dog'},
                  'Chris' : {'age' : 27,
                             'species' : 'human'}}
        self.assertEqual(list_of_dicts_to_dict(l, major_key, other_keys), answer)
        
    def test_H_from_E(self):
        els_to_amts = {'Al' : 2,
                       'Fe' : 1,
                       'O' : 4}
        mus = {'Al' : -7.7075,
               'Fe' : -18.0782,
               'O' : -6.1543}
        E = -11.134
        answer = -2.8325
        self.assertAlmostEqual(H_from_E(els_to_amts, E, mus), answer, places=3)
        
if __name__ == '__main__':
    unittest.main()