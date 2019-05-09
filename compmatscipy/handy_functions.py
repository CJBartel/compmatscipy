import json, os
import numpy as np
from subprocess import call

def make_directory_tree(path_to_make, sep='/'):
    """
    Args:
        path_to_make (str) - relative path of directory to make
        sep (str) - os-dependent path separator
    
    Returns:
        None (makes directory of interest)
    """
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
        major_key (tuple, str, float, int) - key to orient output dictionary on
        other_keys (list) - list of keys (tuple, str, float, int) to include in output dictionary
    
    Returns:
        dictionary representation of information in l
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

def get_pbs_q(f_qstat='qstat.txt', f_jobs='qjobs.txt', username='cbartel'):
    """
    Args:
        f_qstat (str) - path to write detailed queue information
        f_jobs (str) - path to write job IDs
        username (str) - user name on HPC
    
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

def is_slurm_job_in_queue(job_name, user_name='tg857781', fqueue='q.out'):
    with open(fqueue, 'w') as f:
        call(['squeue','-u', user_name, '--name=%s' % job_name], stdout=f)
    names_in_q = []
    with open(fqueue) as f:
        for line in f:
            if 'PARTITION' not in line:
                names_in_q.append([v for v in line.split(' ') if len(v) > 0][2])
    if len(names_in_q) == 0:
        return False
    else:
        return True

def get_stampede2_queue_counts(fqueue):
    with open(fqueue, 'w') as f:
        call(['squeue','-u', 'tg857781'], stdout=f)
    with open(fqueue) as f:
        normal = 0
        skx = 0
        for line in f:
            if 'PARTITION' not in line:
                queue = [v for v in line.split(' ') if len(v) > 0][1]
                if queue == 'normal':
                    normal += 1
                elif queue == 'skx':
                    skx += 1
    total = normal + skx
    return {'normal' : normal,
            'skx' : skx,
            'total' : total}
