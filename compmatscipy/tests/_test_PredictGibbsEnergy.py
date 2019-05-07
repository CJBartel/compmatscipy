# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:29:41 2018

@author: Chris
"""

import os, unittest
from compmatscipy.PredictGibbsEnergy import PredictGibbsEnergy

test_dir = os.path.join(os.path.dirname(__file__))
test_data_dir = os.path.join(test_dir, 'test_data')

class UnitTestPredictGibbsEnergies(unittest.TestCase):
    
    def test_dGpredictor(self):
        formula = 'MgSiO3'
        H, V = -1548.917/5/96.485, 10.8101
        obj = PredictGibbsEnergy(formula, H, V)
        T1 = 300
        T1_pred = obj.dG(T1)
        T1_answer = -1462.023/5/96.485
        self.assertAlmostEqual(T1_pred, T1_answer, places=1)
        
if __name__ == '__main__':
    unittest.main()