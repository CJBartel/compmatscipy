# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os, json

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data", "MP_pseudos.json")


def MP_pseudos_data():
    with open(DATA_PATH) as f:
        return json.load(f)
