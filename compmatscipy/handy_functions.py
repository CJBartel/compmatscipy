# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 22:04:40 2018

@author: Chris
"""

import json, os
import numpy as np

def make_directory_tree(path_to_make, sep='/'):
    path_pieces = path_to_make.split(sep)
    for i in range(len(path_pieces)):
        parent = sep.join(path_pieces[:i+1])
        if not os.path.exists(parent):
            os.mkdir(parent)

def read_json(fjson):
    """
    Args:
        fjson (str) - file name of json to read
    Returns:
        dictionary stored in fjson
    """
    with open(fjson) as f:
        return json.load(f)

def write_json(d, fjson):
    """
    Args:
        d (dict) - dictionary to write
        fjson (str) - file name of json to write
    Returns:
        written dictionary
    """        
    with open(fjson, 'w') as f:
        json.dump(d, f)
    return d  

def gcd(a,b):
    """
    Args:
        a (float, int) - some number
        b (float, int) - another number
    Returns:
        greatest common denominator (int) of a and b
    """
    while b:
        a, b = b, a%b
    return a    

def list_of_dicts_to_dict(l, major_key, other_keys):
    """
    Args:
        l (list) - list of dictionaries
        major_key - key to orient output dictionary on
        other_keys (list) - list of keys to include in output dictionary
    Returns:
        dictionary of information in l
    """
    return {d[major_key] : {other_key : d[other_key] for other_key in other_keys} for d in l}

def H_from_E(els_to_amts, E, mus):
    """
    Args:
        els_to_amts (dict) - {element (str) : amount of element in formula (int) for element in formula}        
        formula (str) - chemical formula
        E (float) - total energy per atom
        mus (dict) - {el (str) : elemental energy (float)}
    Returns:
        formation energy per atom (float)
    """
    atoms_in_fu = np.sum(list(els_to_amts.values()))
    stoich_weighted_elemental_energies = np.sum([mus[el]*els_to_amts[el] for el in els_to_amts])
    E_per_fu = E*atoms_in_fu
    Ef_per_fu = E_per_fu - stoich_weighted_elemental_energies
    return Ef_per_fu / atoms_in_fu

def get_q(f_qstat='qstat.txt', f_jobs='qjobs.txt', username='cbartel'):
    """
    Args:
        f_qstat (str) - path to write detailed queue information
        f_jobs (str) - path to write job IDs
    Returns:
        list of job IDs in the queue (str)
    """
    from subprocess import call
    if os.path.exists(f_qstat):
        os.remove(f_qstat)
    if os.path.exists(f_jobs):
        os.remove(f_jobs)
    with open(f_qstat, 'wb') as f:
        call(['qstat', '-f', '-u', username], stdout=f)
    with open(f_jobs, 'wb') as f:
        call(['grep', 'Job_Name', f_qstat], stdout=f)
    with open(f_jobs) as f:
        jobs_in_q = [line.split(' = ')[1][:-1] for line in f]
    return jobs_in_q
