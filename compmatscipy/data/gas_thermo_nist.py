# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os, json

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data", "gas_thermo_data.json")

def gas_thermo_nist_data():
    with open(DATA_PATH) as f:
        return json.load(f)