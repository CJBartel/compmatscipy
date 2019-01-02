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
        return json.load(f)