# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 19:07:33 2018

@author: Chris
"""

import os, unittest
from compmatscipy.HelpWithVASP import VASPSetUp, VASPBasicAnalysis, VASPDOSAnalysis, LOBSTERAnalysis, ProcessDOS

test_dir = os.path.join(os.path.dirname(__file__))
test_data_dir = os.path.join(test_dir, 'test_data')

def calcs():
    names = ['PBE_sp', 'SCAN_geometry', 'HSE_dielectric', 'COHP']
    return dict(zip(names, [os.path.join(test_data_dir, name) for name in names]))

class UnitTestHelpWithVASP(unittest.TestCase):
    
    def test_set_up_els_from_poscar(self):
        calc = calcs()['PBE_sp']
        els = VASPSetUp(calc).ordered_els_from_poscar(True)
        answer = ['O', 'Fe', 'As']
        self.assertEqual(els, answer)
        
    def test_basic_is_converged(self):
        for calc in calcs():
            check = VASPBasicAnalysis(calcs()[calc]).is_converged
            answer = True
            self.assertEqual(check, answer)
            
    def test_basic_els_from_outcar(self):
        calc = calcs()['PBE_sp']
        els = VASPBasicAnalysis(calc).ordered_els_from_outcar
        answer = ['O', 'Fe', 'As']
        self.assertEqual(els, answer)
    
    def test_basic_idxs_to_els(self):    
        calc = calcs()['SCAN_geometry']
        check = VASPSetUp(calc).idxs_to_els[13]
        answer = 'Ag'
        self.assertEqual(check, answer)
        
    def test_basic_formula(self):
        calc = calcs()['COHP']
        formula = VASPBasicAnalysis(calc).formula(True)
        answer = 'Mo1N4Zn3'
        self.assertEqual(formula, answer)
        
    def test_basic_Etot(self):
        calc = calcs()['HSE_dielectric']
        Etot = VASPBasicAnalysis(calc).Etot
        answer = -.71601536E+02/20
        self.assertEqual(Etot, answer)
        
    def test_basic_Efermi(self):
        calc = calcs()['SCAN_geometry']
        Efermi = VASPBasicAnalysis(calc).Efermi
        answer = 0.3567
        self.assertEqual(Efermi, answer)
        
    def __test_tdos_detailed(self):
        calc = calcs()['SCAN_geometry']
        d = VASPDOSAnalysis(calc).detailed_dos_dict(fjson=os.path.join(test_data_dir, 'test_dos.json'), remake=True)
        total_line = '-6.203  0.1125E-04  0.1126E-04  0.1600E-01  0.1600E-01'
        total_line = [float(v) for v in total_line.split(' ') if v != '']
        energy, up, down, intup, intdown = total_line
        self.assertIn(energy, d)
        self.assertEqual(d[energy]['total']['up'], up)
        self.assertEqual(d[energy]['total']['down'], down)
        
        e_to_pop = VASPDOSAnalysis(calc).energies_to_populations(element='total', orbital='all', spin='up', fjson='test_dos.json', remake=False)
        self.assertEqual(e_to_pop[energy], up)
        
        
if __name__ == '__main__':
    unittest.main()