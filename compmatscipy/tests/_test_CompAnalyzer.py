# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 13:33:18 2018

@author: Chris
"""
import unittest
from compmatscipy.CompAnalyzer import CompAnalyzer

class UnitTestCompAnalyzer(unittest.TestCase):
    
    def test_basic(self):
        formula = 'Al2O3'
        answer = 'Al2O3'
        self.assertEqual(CompAnalyzer(formula).std_formula(), answer)
    
    def test_alphabetize(self):
        formula = 'O3Al2'
        answer = 'Al2O3'
        self.assertEqual(CompAnalyzer(formula).std_formula(), answer)
        
    def test_spaces(self):
        formula = 'O3 Al2'
        answer = 'Al2O3'
        self.assertEqual(CompAnalyzer(formula).std_formula(), answer)        
    
    def test_repeat_element(self):
        formula = 'Al2O3Al2'
        answer = 'Al4O3'
        self.assertEqual(CompAnalyzer(formula).std_formula(), answer)
        
    def test_missing_one(self):
        formula = 'TiO2'
        answer = 'O2Ti1'
        self.assertEqual(CompAnalyzer(formula).std_formula(), answer)
        
    def test_single_parentheses(self):
        formula = 'Al(OH)3'
        answer = 'Al1H3O3'
        self.assertEqual(CompAnalyzer(formula).std_formula(), answer)

    def test_missing_one_parentheses(self):
        formula = 'Al(OH)'
        answer = 'Al1H1O1'
        self.assertEqual(CompAnalyzer(formula).std_formula(), answer)
        
    def test_many_parentheses(self):
        formula = 'Al(OH)3(CO2)4Ti'
        answer = 'Al1C4H3O11Ti1'
        self.assertEqual(CompAnalyzer(formula).std_formula(), answer)
    
    def test_reduce_formula(self):
        formula = 'Ti4O12'
        answer = 'O3Ti1'
        self.assertEqual(CompAnalyzer(formula).std_formula(), answer)
        
    def test_dont_reduce_formula(self):
        formula = 'Ti4O12'
        answer = 'O12Ti4'
        self.assertEqual(CompAnalyzer(formula).std_formula(False), answer)
        
    def test_els(self):
        formula = 'Al(OH)3(CO2)4Ti'
        answer = ['Al', 'C', 'H', 'O', 'Ti']
        self.assertEqual(CompAnalyzer(formula).els, answer)
        
    def test_amts(self):
        formula = 'O2 Ti'
        answer = [2, 1]
        self.assertEqual(CompAnalyzer(formula).amts(), answer)
        
    def test_num_els_in_formula(self):
        formula = 'Al(OH)3(CO2)4Ti'
        answer = 5
        self.assertEqual(CompAnalyzer(formula).num_els_in_formula, answer)        
    
    def test_num_atoms_in_formula(self):
        formula = 'O2 Ti'
        answer = 3  
        self.assertEqual(CompAnalyzer(formula).num_atoms_in_formula(), answer)
        
    def test_fractional_amts(self):
        formula = 'O2 Ti O'
        answer = [3/4, 1/4]
        self.assertEqual(CompAnalyzer(formula).fractional_amts, answer)
        
    def test_amt_of_el_when_el_not_there(self):
        formula = 'Al2O3'
        el = 'Ti'
        answer = 0
        self.assertEqual(CompAnalyzer(formula).amt_of_el(el=el), answer)
        
    def test_amt_of_el_when_el_is_there(self):
        formula = 'Al2O3'
        el = 'Al'
        answer = 2
        self.assertEqual(CompAnalyzer(formula).amt_of_el(el=el), answer)
    
    def test_frac_amt_of_el_when_el_is_there(self):
        formula = 'Al2O3'
        el = 'Al'
        answer = 0.4
        self.assertEqual(CompAnalyzer(formula).fractional_amt_of_el(el=el), answer)          
        
if __name__ == '__main__':
    unittest.main()