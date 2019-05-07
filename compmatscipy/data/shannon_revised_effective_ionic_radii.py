# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os, json

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data", "shannon_revised_effective_ionic_radii.json")


def shannon_revised_effective_ionic_radii_data():
    with open(DATA_PATH) as f:
        d = json.load(f)
#        
#        d['N']['-3']['6'] = 1.54
#        d['P']['-3'] = {'6' : 1.96}
#        d['As']['-3'] = {'6' : 2.07}
#        d['Sb']['-3'] = {'6' : 2.32}
        return d