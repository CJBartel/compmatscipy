# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:53:53 2018

@author: Chris
"""

import os, unittest
from compmatscipy.PerovskiteStability import SinglePerovskiteStability, DoublePerovskiteStability
from compmatscipy.handy_functions import read_json, write_json
from compmatscipy.data import calibrated_tau_prob_clf

clf = calibrated_tau_prob_clf()

test_dir = os.path.join(os.path.dirname(__file__))

def make_json(remake=False):
    fjson = os.path.join(test_dir, 'test_data', 'perovskite_test_data.json')
    return read_json(fjson)

class UnitTestPerovskiteStability(unittest.TestCase):
    
    def test_single_against_sci_adv(self):
        test_data = make_json(False)
        single_formulas = [k for k in test_data if '3' in k]
        for formula in single_formulas:
            t, tau = test_data[formula]['t'], test_data[formula]['tau']
            obj = SinglePerovskiteStability(formula)
            t_pred, tau_pred = obj.t, obj.tau
            self.assertAlmostEqual(t, t_pred, 2)
            self.assertAlmostEqual(tau, tau_pred, 2)
            
    def test_single_dict_input(self):
        test_data = make_json(False)
        t, tau = test_data['CaTiO3']['t'], test_data['CaTiO3']['tau']
        user_input = {'A' : 'Ca', 'B' : 'Ti', 'X' : 'O'}
        obj = SinglePerovskiteStability(user_input)        
        t_pred, tau_pred = obj.t, obj.tau
        self.assertAlmostEqual(t, t_pred, 2)
        self.assertAlmostEqual(tau, tau_pred, 2)
            
    def test_double_against_sci_adv(self):
        test_data = make_json(False)
        double_formulas = [k for k in test_data if '3' not in k]
        correct = 0
        for formula in double_formulas:
            A, B1, B2, X = test_data[formula]['A'], test_data[formula]['B1'], test_data[formula]['B2'], test_data[formula]['X']
            user_input = dict(zip(['A', 'B1', 'B2', 'X'], [A, B1, B2, X]))
            tau_pred = DoublePerovskiteStability(user_input).tau
            if tau_pred <= 4.18:
                pred = 1
            else:
                pred = -1
            if pred == test_data[formula]['exp_label']:
                correct += 1
        self.assertGreaterEqual(correct, 0.91*len(double_formulas))
        
if __name__ == '__main__':
    unittest.main()