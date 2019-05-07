# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os, pickle

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data", "calibrated_tau_prob.p")


def calibrated_tau_prob_clf():
    with open(DATA_PATH, 'rb') as f:
        return pickle.load(f)