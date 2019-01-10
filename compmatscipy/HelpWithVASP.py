# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:31:56 2018

@author: Chris
"""

import os
from shutil import copyfile
import numpy as np
from compmatscipy.CompAnalyzer import CompAnalyzer
from compmatscipy.handy_functions import read_json, write_json
from scipy.integrate import simps

class VASPSetUp(object):
    
    def __init__(self, calc_dir):
        """
        Args:
            calc_dir (str) - path to run VASP calculation
            
        Returns:
            calc_dir
        """
        self.calc_dir = calc_dir
        
    def incar(self, is_geometry_opt=False, functional='pbe', dos=False, dielectric=False, standard={'EDIFF' : 1e-6,
                                                                                                    'ISMEAR' : 0,
                                                                                                    'SIGMA' : 0.01,
                                                                                                    'ENCUT' : 520,
                                                                                                    'PREC' : 'Accurate',
                                                                                                    'LORBIT' : 11,
                                                                                                    'LASPH' : 'TRUE',
                                                                                                    'ISIF' : 3}, additional={}):
        """
        Args:
            is_geometry_opt (bool) - True if geometry is optimized; else False
            functional (str) - 'pbe', 'scan', or 'hse'
            dos (bool) - True if accurate DOS desired
            dielectric (bool) - True if LOPTICS on
            standard (dict) - dictionary of parameters for starting point in INCAR
            additional (dict) - dictionary of parameters to enforce in INCAR
        
        Returns:
            writes INCAR file
        """
        d = {}
        
        if is_geometry_opt == True:
            d['IBRION'] = 2
            d['NSW'] = 200
            d['NELM'] = 200
        else:
            d['IBRION'] = -1
            d['NSW'] = 0
            d['NELM'] = 5000
        
        if functional == 'pbe':
            d['GGA'] = 'PE'
        elif functional == 'scan':
            d['METAGGA'] = 'SCAN'
        elif functional == 'hse':
            d['GGA'] = 'PE'
            d['LHFCALC'] = 'TRUE'
            d['ALGO'] = 'Damped'
            d['TIME'] = 0.4
            
        if dos == True:
            d['NEDOS'] = 2500
        
        if dielectric == True:
            d['LOPTICS'] = 'TRUE'
            d['NEDOS'] = 2500
            
        for k in standard:
            d[k] = standard[k]
        
        for k in additional:
            d[k] = additional[k]
            
        fincar = os.path.join(self.calc_dir, 'INCAR')
        with open(fincar, 'w') as f:
            for k in d:
                f.write(' = '.join([k, str(d[k])])+'\n')
        return d
    
    def kpoints(self, discretizations=False, grid=False):
        """
        Args:
            discretizations (int) - False if grid to be specified; else number of discretizations for auto grid
            grid (int) - False if discretizations to be specified; else tuple of ints for grid
            
        Returns:
            writes KPOINTS file
        """
        fkpoints = os.path.join(self.calc_dir, 'KPOINTS')
        with open(fkpoints, 'w') as f:
            if grid != False:
                f.write('Auto\n')
                f.write('0\n')
                f.write('Gamma\n')
                f.write(' '.join([str(g) for g in grid])+'\n')
                f.write('0 0 0')
            elif discretizations != False:
                f.write('Auto\n')
                f.write('0\n')
                f.write('Auto\n')
                f.write(str(discretizations)+'\n') 
            else:
                print('youn need to specify how to write the KPOINTS file')
                
    def poscar(self, copy_contcar=False):
        """
        Args:
            copy_contcar (bool) - if True, copies CONTCAR to POSCAR is CONTCAR not empty
            
        Returns:
            POSCAR
        """
        poscar = os.path.join(self.calc_dir, 'POSCAR')
        if not os.path.exists(poscar):
            copy_contcar = True
        if not copy_contcar:
            return poscar
        contcar = os.path.join(self.calc_dir, 'CONTCAR')
        if os.path.exists(contcar):
            with open(contcar) as f:
                contents = f.read()
            if '0' in contents:
                copyfile(contcar, poscar)
            return poscar
                
    def ordered_els_from_poscar(self, copy_contcar=False):
        """
        Args:
            copy_contcar (bool) - if True, copies CONTCAR to POSCAR is CONTCAR not empty 
            
        Returns:
            dictionary of {element (str) : number in calculated structure (int)}
        """
        poscar = self.poscar(copy_contcar)
        with open(poscar) as f:
            count = 0
            for line in f:
                if count <= 5:
                    count += 1
                if count == 6:
                    els = [v for v in line[:-1].split(' ') if v != '']
                    try:
                        num_check = [int(el) for el in els]
                        print('elements not provided in POSCAR')
                        return np.nan
                    except:
                        return els
                
    def potcar(self, els_in_poscar=False, specific_pots=False, path_to_pots='/projects/thermochem/rs_perovs/potpaw_PBE.54'):
        """
        Args:
            els_in_poscar (list or False) - ordered list of elements (str) in POSCAR; if FALSE, read POSCAR
            which_pot (bool or dict) - False to use VASP defaults; else dict of {el : which POTCAR (str)}
            
        Returns:
            writes POTCAR file
        """
        if not els_in_poscar:
            els_in_poscar = self.ordered_els_from_poscar
        fpotcar = os.path.join(self.calc_dir, 'POTCAR')
        with open(fpotcar, 'w') as f:
            for el in els_in_poscar:
                if specific_pots == False:
                    pot_to_add = os.path.join(path_to_pots, el, 'POTCAR')
                else:
                    pot_to_add = os.path.join(path_to_pots, el+'_'+specific_pots[el])
                with open(pot_to_add) as g:
                    for line in g:
                        f.write(line)
    
    def sub(self, log_name, nodes, ppn, queue, walltime, allocation, 
                  priority='low', feature=False, 
                  username='cbartel', bash='.bashrc', 
                  mpi_command='/nopt/intel/psxe2017u2/compilers_and_libraries_2017.5.239/linux/mpi/intel64/bin/mpiexec', 
                  vasp='/projects/thermochem/apps/vasp.5.4.4/impi_centos/bin/vasp_std',
                  out_file='vasp.out', sub_file='sub.sh'):
        """
        Args:
            log_name (str) - log name for calculation
            nodes (int)- number of nodes
            ppn (int) - cores per node
            queue (str) - which queue to run on
            walltime (int) - walltime in hours
            allocation (str) - allocation to charge against            
            priority (str) - 'low' or 'high'
            feature (str) - feature to specify for certain queue (e.g., 256GB)
            username (str) - user name
            bash (str) - bashrc to execute
            mpi_command (str) - mpi command to execute
            vasp (str) - path to VASP
            out_file (str) - VASP output file
            sub_file (str) - submission script file name
            
        Returns:
            writes sub_file to calc_dir
        """
        fsub = os.path.join(self.calc_dir, sub_file)
        long_calc_dir = os.path.join(os.getcwd(), self.calc_dir)
        with open(fsub, 'w', newline='\n') as f:
            f.write('#!/bin/sh\n')
            f.write('#PBS -l nodes=%s:ppn=%s -q %s -l walltime=%s:00:00\n' % (nodes, ppn, queue, walltime))
            if priority == 'high':
                f.write('#PBS -l qos=high\n')
            if feature != False:
                f.write('#PBS -l feature=%s\n' % feature)
            f.write('#PBS -e %s -o %s\n' % (long_calc_dir, long_calc_dir))
            f.write('#PBS -k eo -m n -e %s -o %s\n' % (long_calc_dir, long_calc_dir))
            f.write('#PBS -N %s\n' % log_name)
            f.write('#PBS -A %s\n' % allocation)
            f.write('source /home/%s/%s\n' % (username, bash))
            f.write('export OMP_NUM_THREADS=1\n')
            f.write('%s -np %s %s > %s\n' % (mpi_command, str(int(nodes) * int(ppn)), vasp, out_file))
            f.write('exit 0\n')
            
class VASPBasicAnalysis(object):
    """
    Parses VASP inputs and outputs for quick analysis
    Preferred files: INCAR, OSZICAR, OUTCAR, CONTCAR
    """
    
    def __init__(self, calc_dir):
        """
        Args:
            calc_dir (str) - path to analyze VASP calculation
        Returns:
            calc_dir
        """
        self.calc_dir = calc_dir
        
    def params_from_outcar(self, num_params=['NKPTS', 'NKDIM', 'NBANDS', 'NEDOS',
                                             'ISPIN', 'ENCUT', 'NELM', 'EDIFF',
                                             'EDIFFG', 'NSW', 'ISIF', 'ISYM', 
                                             'NELECT', 'NUPDOWN', 'EMIN', 'EMAX', 
                                             'ISMEAR', 'SIGMA', 'AEXX'],
                                 str_params = ['PREC', 'METAGGA', 'LHFCALC', 'LEPSILON',
                                              'LRPA']):
        """
        Args:
            num_params (list) - list of numerical parameters to retrieve (str)
            str_params (list) - list of True/False or string parameters to retrieve (str)
            
        Returns:
            dictionary of {paramater (str) : value (str or float) for each parameter provided}
        """
        outcar = os.path.join(self.calc_dir, 'OUTCAR')
        data = {}        
        params = num_params + str_params
        with open(outcar) as f:
            for param in params:
                f.seek(0)
                for line in f:
                    if (param in line) and ('LMODELHF' not in line):
                        val = line.split(param)[1].split('=')[1].strip().split(' ')[0].replace(';', '')
                        break
                if param in num_params:
                    data[param] = float(val)
                elif param in str_params:
                    data[param] = str(val)
        return data
    
    @property
    def nelect(self):
        return self.params_from_outcar(num_params=['NELECT'], str_params=[])['NELECT']
    
    @property
    def params_from_incar(self):
        """
        Args:

        Returns:
            dictionary of {paramater (str) : value (str or float) for each parameter set in INCAR}
        """        
        incar = os.path.join(self.calc_dir, 'INCAR')
        data = {}
        with open(incar) as f:
            for line in f:
                if ('=' in line) and (line[0] != '!'):
                    line = line[:-1].split(' = ')
                    data[line[0].strip()] = line[1].strip()
        return data
        
    @property
    def is_converged(self):
        """
        Args:
        
        Returns:
            True if calculation converged; else False
        """
        outcar = os.path.join(self.calc_dir, 'OUTCAR')
        oszicar = os.path.join(self.calc_dir, 'OSZICAR')
        if not os.path.exists(outcar):
            return False
        with open(outcar) as f:
            contents = f.read()
            if 'Elaps' not in contents:
                return False
        params = self.params_from_outcar(num_params=['NELM', 'NSW'], str_params=[])
        nelm, nsw = params['NELM'], params['NSW']
        if nsw == 0:    
            max_ionic_step = 1
        else:
            if os.path.exists(oszicar):
                with open(oszicar) as f:
                    for line in f:
                        if 'F' in line:
                            step = line.split('F')[0].strip()
                            if ' ' in step:
                                step = step.split(' ')[-1]
                            step = int(step)
            else:
                with open(outcar) as f:
                    f.seek(0)
                    for line in f:
                        if ('Iteration' in line) and ('(' in line): #)
                            step = line.split('Iteration')[1].split('(')[0].strip() #)
                            step = int(step)
            max_ionic_step = step
            if max_ionic_step == nsw:
                return False
        with open(outcar) as f:
            f.seek(0)
            for line in f:
                if ('Iteration' in line) and (str(max_ionic_step)+'(') in line: #)
                    step = line.split(str(max_ionic_step)+'(')[1].split(')')[0].strip()
                    if int(step) == nelm:
                        return False
        return True
    
    @property
    def structure(self):
        """
        Args:
            
        Returns:
            CONTCAR if it exists; else POSCAR
        """
        fposcar = os.path.join(self.calc_dir, 'POSCAR')
        if 'CONTCAR' in os.listdir(self.calc_dir):
            fcontcar = os.path.join(self.calc_dir, 'CONTCAR')
            with open(fcontcar) as f:
                contents = f.read()
                if len(contents) > 0:
                    return fcontcar
                else:
                    return fposcar
        else:
            return fposcar
        
    @property
    def ordered_els_from_outcar(self):
        """
        Args:
            
        Returns:
            list of elements (str) as they appear in POTCAR
        """
        outcar = os.path.join(self.calc_dir, 'OUTCAR')
        els = []
        with open(outcar) as f:
            for line in f:
                if 'VRHFIN' in line:
                    el = line.split('=')[1].split(':')[0]
                    els.append(el.strip())
                if 'POSCAR' in line:
                    break
        return els
        
    @property
    def els_to_amts(self):
        """
        Args:
        
        Returns:
            dictionary of {element (str) : number in calculated structure (int)}
        """
        els = self.ordered_els_from_outcar
        with open(self.structure) as f:
            count = 0
            for line in f:
                if count <= 4:
                    count += 1
                if count == 5:
                    stuff = [v for v in line.split(' ') if v != '']
                    try:
                        amts = [int(v) for v in stuff]
                        els_to_amts = dict(zip(els, amts))
                        break
                    except:
                        continue
                if count == 6:
                    stuff = [v for v in line.split(' ') if v != '']
                    amts = [int(v) for v in stuff]
                    els_to_amts = dict(zip(els, amts))   
        return els_to_amts
    
    @property
    def idxs_to_els(self):
        """
        Args:
            
        Returns:
            dictionary of index in structure (int) to the element (str) at that index
        """
        els = self.ordered_els_from_outcar
        els_to_amts = self.els_to_amts
        els_to_sites = {}
        idx = 0
        for el in els:
            start, stop = idx, els_to_amts[el]+idx
            els_to_sites[el] = (start, stop)
            idx = stop        
        nsites = self.nsites
        els_to_idxs = {el : range(els_to_sites[el][0], els_to_sites[el][1]) for el in els_to_sites}
        idxs_to_els = {}
        for idx in range(nsites):
            for el in els_to_idxs:
                if idx in els_to_idxs[el]:
                    idxs_to_els[idx] = el
        return idxs_to_els
                
    def formula(self, reduce=False):
        """
        Args:
            reduce (bool) - if True; reduce formula to unit cell
            
        Returns:
            standardized chemical formula of calculated structure
        """
        els_to_amts = self.els_to_amts
        initial = ''.join([el+str(els_to_amts[el]) for el in els_to_amts])
        return CompAnalyzer(initial).std_formula(reduce)
                
    @property
    def nsites(self):
        """
        Args:
            
        Returns:
            number (int) of ions in calculated structure
        """
        return np.sum(list(self.els_to_amts.values()))
              
    @property
    def Etot(self):
        """
        Args:
            
        Returns:
            energy per atom (float) of calculated structure if converged
        """
        if not self.is_converged:
            return np.nan
        oszicar = os.path.join(self.calc_dir, 'OSZICAR')
        if os.path.exists(oszicar):
            with open(oszicar) as f:
                for line in f:
                    if 'F' in line:
                        E = float(line.split('F=')[1].split('E0')[0].strip())
        else:
            outcar = os.path.join(self.calc_dir, 'OUTCAR')
            with open(outcar) as f:
                for line in reversed(list(f)):
                    if 'TOTEN' in line:
                        line = line.split('=')[1]
                        E = float(''.join([c for c in line if c not in [' ', '\n', 'e', 'V']]))
                        break
        return float(E)/self.nsites
    
    @property
    def Efermi(self):
        """
        Args:
            
        Returns:
            Fermi energy outputted by VASP (float)
        """
        outcar = os.path.join(self.calc_dir, 'OUTCAR')
        with open(outcar) as f:
            for line in f:
                if 'E-fermi' in line:
                    return float(line.split(':')[1].split('XC')[0].strip())
                
    def gaps(self, fgaps):
        """
        Args:
            fgaps (str) - file with $ grepgap OUTCAR output
                            NOTE: grepgap is a custom function...
                            
        Returns:
            dictionary of {'Eg' : overall band gap (float), 'Egd' : direct band gap (float)}
        """
        calc_dir = self.calc_dir
        fgaps = os.path.join(calc_dir, fgaps)
        if os.path.exists(fgaps):
            with open(fgaps) as f:
                contents = f.read()
                if 'metal' in contents:
                    Eg, Egd = 0, 0
                elif 'Eg' not in contents:
                    Eg, Egd = np.nan, np.nan
                else:
                    f.seek(0)
                    for line in f:
                        if ('Eg' in line) and ('Egd' not in line):
                            Eg = float(line.split('Eg')[0].strip())
                        elif 'Egd' in line:
                            Egd = float(line.split('Egd')[0].strip())
        else:
            Eg, Egd = np.nan, np.nan
        return {'Eg' : Eg, 'Egd' : Egd}                
                
class VASPDOSAnalysis(object):
    """
    Convert DOSCAR to useful dictionary
    """
    
    def __init__(self, calc_dir, doscar='DOSCAR'):
        """
        Args:
            calc_dir (str) - path to VASP calculation
            
        Returns:
            path to DOSCAR to analyze
        """
        self.calc_dir = calc_dir
        self.doscar = os.path.join(calc_dir, doscar)
    
    def detailed_dos_dict(self, fjson=False, remake=False):
        """
        Args:
            fjson (str or False) - path to json to write; if False, writes to calc_dir/DOS.json
            remake (bool) - if True, regenerate json; else read json
            
        Returns:
            dictionary of DOS information
                first_key = energy (float)
                next_keys = ['total'] + els
                for total, keys are ['down', 'up']
                for els, keys are orbital_up, orbital_down, level_up, level_down, all_up, all_down
                    populations are as generated by vasp
        """
        if not fjson:
            if 'lobster' not in self.doscar:
                fjson = os.path.join(self.calc_dir, 'DOS.json')
            else:
                fjson = os.path.join(self.calc_dir, 'lob_DOS.json')
        if remake or not os.path.exists(fjson) or (read_json(fjson) == {}):
            basic_obj = VASPBasicAnalysis(self.calc_dir)
            idxs_to_els = basic_obj.idxs_to_els
            num_params = ['ISPIN', 'NEDOS']
            d_params = basic_obj.params_from_outcar(num_params=num_params, str_params=[])
            spin, nedos = d_params['ISPIN'], int(d_params['NEDOS'])

            if 'lobster' in self.doscar:
                with open(self.doscar) as f:
                    count = 0
                    for line in f:
                        count += 1
                        if count == 6:
                            nedos = int([v for v in line.split(' ') if v != ''][2])
                            break
            if spin == 2:
                spins = ['up', 'down']
                orbital_keys = 'E,s_up,s_down,py_up,py_down,pz_up,pz_down,px_up,px_down,dxy_up,dxy_down,dyz_up,dyz_down,dzz_up,dzz_down,dxz_up,dxz_down,dxxyy_up,dxxyy_down'.split(',')
                total_keys = 'E,up,down'.split(',')
            else:
                spins = ['up']
                orbital_keys = 'E,s_up,py_up,pz_up,px_up,dxy_up,dyz_up,dzz_up,dxz_up,dxxyy_up'.split(',')
                total_keys = 'E,up'.split(',')
            dos_count = 0
            count = 0
            data = {}
            with open(self.doscar) as f:
                for line in f:
                    count += 1
                    if count == 6+(dos_count*nedos)+dos_count:
                       dos_count += 1
                       continue
                    if dos_count == 1:
                        values = [v for v in line[:-1].split(' ') if v != '']
                        tmp = dict(zip(total_keys, values))
                        data[float(tmp['E'])] = {'total' : {k : float(tmp[k]) for k in tmp if k != 'E'}}
                    elif dos_count == 0:
                        continue
                    else:
                        values = [v for v in line[:-1].split(' ') if v != '']                        
                        el = idxs_to_els[dos_count-2]
                        tmp = dict(zip(orbital_keys, values))
                        if el not in data[float(tmp['E'])]:
                            data[float(tmp['E'])][el] = {k : float(tmp[k]) for k in tmp if k != 'E'}
                        else:
                            for k in data[float(tmp['E'])][el]:
                                if k != 'E':
                                    data[float(tmp['E'])][el][k] += float(tmp[k])
            for E in data:
                for el in data[E]:
                    if el != 'total':
                        for orb in ['p', 'd']:
                            for spin in spins:
                                data[E][el]['_'.join([orb, spin])] = np.sum([data[E][el][k] for k in data[E][el] if k.split('_')[0][0] == orb if k.split('_')[1] == spin])
                        for spin in spins:
                            data[E][el]['_'.join(['all', spin])] = np.sum([data[E][el]['_'.join([orb, spin])] for orb in ['s', 'p', 'd']])
            for E in data:
                data[E]['total'] = {spin : 0 for spin in spins}
                for el in data[E]:
                    if el != 'total':
                        for spin in spins:
                            data[E]['total'][spin] += data[E][el]['_'.join(['all', spin])]
            return write_json(data, fjson)
        else:
            data = read_json(fjson)
            return {float(k) : data[k] for k in data}
        
    def energies_to_populations(self, element='total', orbital='all', spin='summed', fjson=False, remake=False):
        """
        Args:
            element (str) - element or 'total' to analyze
            orbital (str) - orbital or 'all' to analyze
            spin (str) - 'up', 'down', or 'summed'
            fjson (str or False) - path to json to write; if False, writes to calc_dir/DOS.json
            remake (bool) - if True, regenerate dos_dict json; else read json         
            
        Returns:
            dictionary of {energies (float) : populations (float)}
        """
        dos_dict = self.detailed_dos_dict(fjson, remake)
        energies = sorted(list(dos_dict.keys()))
        ispin = VASPBasicAnalysis(self.calc_dir).params_from_outcar(num_params=['ISPIN'], str_params=[])['ISPIN']
        if ispin == 1:
            spins = ['up']
        elif ispin == 2:
            spins = ['up', 'down']
        if element != 'total':
            if spin != 'summed':
                populations = [dos_dict[E][element]['_'.join([orbital, spin])] for E in energies]
            else:
                populations = [0 for E in energies]
                for s in range(len(spins)):
                    spin = spins[s]
                    for i in range(len(energies)):
                        E = energies[i]
                        populations[i] += dos_dict[E][element]['_'.join([orbital, spin])]
        else:
            if spin != 'summed':
                populations = [dos_dict[E][element][spin] for E in energies]
            else:
                populations = [np.sum([dos_dict[E][element][spin] for spin in spins]) for E in energies]
        return dict(zip(energies, populations))
        
class LOBSTERAnalysis(object):
    """
    Convert COHPCAR or COOPCAR to useful dictionary
    """
    
    def __init__(self, calc_dir, lobster='COHPCAR.lobster'):
        """
        Args:
            calc_dir (str) - path to calculation with LOBSTER output
            lobster (str) - either 'COHPCAR.lobster' or 'COOPCAR.lobster'
            
        Returns:
            calc_dir
            path to LOBSTER output
        """
        self.calc_dir = calc_dir
        self.lobster = os.path.join(calc_dir, lobster)
    
    @property
    def pair_dict(self):
        """
        Args:
            
        Returns:
            dictionary of {pair index (int) : {'els' : (el1, el1) (str), 
                                               'sites' : (structure index for el1, structure index for el2) (int),
                                               'orbitals' : (orbital for el1, orbital for el2) (str) ('all' if all orbitals summed),
                                               'dist' : distance in Ang (float)}
                                               'energies' : [],
                                               'populations' : []}
        """
        lobster = self.lobster
        data = {}
        with open(lobster) as f:
            count = 0
            idx_count = 0
            for line in f:
                if count < 3:
                    count += 1
                    continue
                elif line[:3] == 'No.':
                    idx_count += 1
                    pair_idx = idx_count
                    if '[' not in line: #]
                        el_site1 = line.split(':')[1].split('->')[0]
                        el_site2 = line.split(':')[1].split('->')[1].split('(')[0] #)
                        orb1, orb2 = 'all', 'all' 
                    else:
                        el_site1 = line.split(':')[1].split('->')[0].split('[')[0] #]
                        el_site2 = line.split(':')[1].split('->')[1].split('(')[0].split('[')[0] #)]
                        orb1, orb2 = line.split('[')[1].split(']')[0], line.split('->')[1].split('[')[1].split(']')[0]  
                    dist = float(line.split('(')[1].split(')')[0])
                    el1, el2 = CompAnalyzer(el_site1).els[0], CompAnalyzer(el_site2).els[0]
                    site1, site2 = int(el_site1.split(el1)[1]), int(el_site2.split(el2)[1]) 
                    data[pair_idx] = {'els' : (el1, el2),
                                      'sites' : (site1, site2),
                                      'orbitals' : (orb1, orb2),
                                      'dist' : dist,
                                      'energies' : [],
                                      'populations' : []}
                else:
                    return data        

    def detailed_dos_dict(self, fjson=False, remake=False):
        """
        Args:
            fjson (str) - path to json to write; if False, writes to calc_dir/COHP.json or COOP.json
            remake (bool) - if True, regenerate json; else read json
            
        Returns:
            dictionary of COHP/COOP information
                first_key = energy (float)
                next_keys = each unique el1_el2 interaction and also total
                for total, value is the total population
                for each sorted(el1_el2) interaction, keys are each specific sorted(site1_site2) interaction for those elements and total
                    populations are as generated by LOBSTER
        """
        if not fjson:
            fjson = os.path.join(self.lobster.replace('CAR.lobster', '.json'))
        if remake or not os.path.exists(fjson) or (read_json(fjson) == {}):
            lobster = self.lobster
            data = self.pair_dict
            with open(lobster) as f:
                count = 0
                for line in f:
                    if (count < 3) or (line[:3] == 'No.'):
                        count += 1
                        continue
                    else:
                        line = line.split(' ')
                        values = [v for v in line if v != '']
                        energy = float(values[0])
                        for pair in data:
                            idx = 2+pair*2-1
                            population = float(values[idx])     
                            data[pair]['energies'].append(energy)
                            data[pair]['populations'].append(population)
            new = {}
            energies = data[1]['energies']
            element_combinations = list(set(['_'.join(sorted(data[pair]['els'])) for pair in data]))
            for i in range(len(energies)):
                overall_total = 0
                tmp1 = {}
                for el_combo in element_combinations:
                    combo_total = 0
                    pairs = [pair for pair in data if (tuple(el_combo.split('_')) == data[pair]['els'])]
                    if len(pairs) == 0:
                        pairs = [pair for pair in data if (tuple(el_combo.split('_')[::-1]) == data[pair]['els'])]
                        sites = [data[pair]['sites'][::-1] for pair in pairs]
                        orbitals = [data[pair]['orbitals'][::-1] for pair in pairs]
                    else:
                        sites = [data[pair]['sites'] for pair in pairs]
                        orbitals = [data[pair]['orbitals'] for pair in pairs]                        
                    tmp2 = {'_'.join([str(s) for s in site]) : {} for site in list(set(sites))}
                    for j in range(len(pairs)):
                        pair, site, orbital = pairs[j], sites[j], orbitals[j]
                        population = data[pair]['populations'][i]
                        if orbital == ('all', 'all'):
                            combo_total += population
                        site_key = '_'.join([str(s) for s in site])
                        orb_key = '-'.join(orbital)
                        tmp2[site_key][orb_key] = population
                    tmp2['total'] = combo_total
                    tmp1[el_combo] = tmp2
                    overall_total += combo_total
                tmp1['total'] = overall_total
                new[energies[i]] = tmp1
            return write_json(new, fjson)
        else:
            data = read_json(fjson)
            return {float(k) : data[k] for k in data}
        
    def energies_to_populations(self, element_pair='total', site_pair='total', orb_pair='all-all', fjson=False, remake=False):
        """
        Args:
            element_pair (str) - el1_el2 (alphabetical) or 'total'
            site_pair (str) - site1_site2 (order corresponds with el1_el2) or 'total'
            orb_pair (str) - orb1-orb2 (order corresponds with el1_el2) or 'all-all' for all orbitals
            fjson (str or False) - path to json to write; if False, writes to calc_dir/DOS.json
            remake (bool) - if True, regenerate dos_dict json; else read json            
        Returns:
            dictionary of {energies (float) : populations (float)} for specified subset
        """        
        dos_dict = self.detailed_dos_dict(fjson, remake)
        energies = sorted(list(dos_dict.keys()))
        if element_pair != 'total':
            if site_pair != 'total':
                populations = [dos_dict[E][element_pair][site_pair][orb_pair] for E in energies]
            else:
                populations = [dos_dict[E][element_pair][site_pair] for E in energies]
        else:
            populations = [dos_dict[E][element_pair] for E in energies]
        return dict(zip(energies, populations))
        
class ProcessDOS(object):
    """
    Handles generic dictionary of {energies : states}
    Used for manipulating density of states (or equivalent) and retrieving summary statistics
    """    
    
    def __init__(self, energies_to_populations, 
                       shift=False,
                       energy_limits=False, 
                       flip_sign=False,
                       min_population=False, 
                       max_population=False, 
                       abs_population=False,
                       normalization=False):
        """
        Args:
            energies_to_populations (dict) - dictionary of {energy (float) : population (float) for all energies}
            populations (array) - number of states at each energy (float)
            shift (float or False) - shift all energies
                e.g., shift = -E_Fermi would make E_fermi = 0 eV
            energy_limits (list or False) - get data only for energies between (including) energy_limits[0] and energy_limits[1]
                e.g., energy_limits = [-1000, E_Fermi] would return only occupied states                
            flip_sign (True or False) - change sign of all populations
            min_population (float or False) - get data only when the population is greater than some value
                e.g., min_population = 0 would return only bonding states in the COHP
            max_population (float or False) - get data only when the population is less than some value
            abs_population (True or False) - make all populations >= 0
            normalization (foat or False) - divide all populations by some value
            
        Returns:
            dictionary of {energy (float) : population (float)} for specified data
        """
        energies = sorted(list(energies_to_populations.keys()))
        populations = [energies_to_populations[E] for E in energies]
        indices = range(len(energies))
        if shift:
            energies = [E+shift for E in energies]
        if flip_sign:
            populations = [-p for p in populations]
        if energy_limits:
            Emin, Emax = energy_limits
            indices = [i for i in indices if energies[i] >= Emin if energies[i] <= Emax]
        if min_population:
            indices = [i for i in indices if populations[i] >= min_population]
        if max_population:
            indices = [i for i in indices if populations[i] <= max_population]
        if abs_population:
            populations = [abs(populations[i]) for i in indices]
        energies, populations = [energies[i] for i in indices], [populations[i] for i in indices]
        if normalization:
            populations = [population/normalization for population in populations]
        self.energies_to_populations = dict(zip(energies, populations))
  
    def stats(self, area=True, energy_weighted_area=True, center=True, width=False, skewness=False, kurtosis=False):
        """
        Args:
            area (bool) - if True, compute integral of absolute populations
            energy_weighted_area (bool) - if True, compute energy-weighted integral
            center (bool) - if True, compute center
            width (bool) - if True, compute width
            skewness (bool) - if True, compute skewness
            kurtosis (bool) if True, compute kurtosis
            
        Returns:
            dictionary of {property (str) : value (float) for specified property in args}
        """
        if center:
            area, energy_weighted_area = True, True
        if width or skewness or kurtosis:
            area = True
        energies_to_populations = self.energies_to_populations
        energies = np.array(sorted(list(energies_to_populations.keys())))
        populations = np.array([energies_to_populations[E] for E in energies])
        summary = {}
        if area:
            summary['area'] = simps(abs(populations), energies)
        if energy_weighted_area:
            summary['energy_weighted_area'] = simps(populations*energies, energies)
        if center:  
            summary['center'] = summary['energy_weighted_area'] / summary['area']
        if width:
            summary['width'] = simps(populations*energies**2, energies) / summary['area']
        if skewness:
            summary['skewness'] = simps(populations*energies**3, energies) / summary['area']
        if kurtosis:
            summary['kurtosis'] = simps(populations*energies**4, energies) / summary['area']
        return summary
        
class VASPDielectricAnalysis(object):
    
    def __init__(self, calc_dir):
        """
        Args:
            calc_dir (str) - path to VASP calculation
            
        Returns:
            calc_dir
        """
        self.calc_dir = calc_dir
        
    @property
    def dielectric_constant(self):
        """
        Args:

        Returns:
            dielectric constant (eps_inf) from LOPTICS=TRUE run (float)
        """    
        outcar = os.path.join(self.calc_dir, 'OUTCAR')
        with open(outcar) as f:
            count = 0
            real_count = 1e6
            for line in f:
                count += 1
                if ('REAL DIELECTRIC FUNCTION' in line) and ('current-current' in line):
                    real_count = count
                if count == real_count + 3:
                    line = line[:-1].split(' ')
                    line = [float(v) for v in line if len(v) > 0]
                    X, Y, Z = line[1:4]
                    eps = np.mean([X, Y, Z])
                    return eps
            if real_count == 1e6:
                f.seek(0)
                count = 0
                for line in f:
                    count += 1
                    if ('REAL DIELECTRIC FUNCTION' in line) and ('density-density' in line):
                        real_count = count
                    if count == real_count + 3:
                        line = line[:-1].split(' ')
                        line = [float(v) for v in line if len(v) > 0]
                        X, Y, Z = line[1:4]
                        eps = np.mean([X, Y, Z])
                        return eps
            if real_count == 1e6:
                f.seek(0)
                count = 0
                for line in f:
                    count += 1
                    if ('REAL DIELECTRIC FUNCTION' in line) and ('no local' in line):
                        real_count = count
                    if count == real_count + 3:
                        line = line[:-1].split(' ')
                        line = [float(v) for v in line if len(v) > 0]
                        X, Y, Z = line[1:4]
                        eps = np.mean([X, Y, Z])
                        return eps
                    
class VASPAbsorptionAnalysis(object):
    
    def __init__(self, calc_dir):
        """
        Args:
            calc_dir (str) - path to VASP calculation
            
        Returns:
            calc_dir
        """        
        self.calc_dir = calc_dir
        
    def absorption_dict(self, fjson=False, remake=False):
        """
        Args:
            fjson (str or False) - where to write dictionary; if False, ABS.json in calc_dir
            remake (bool) - rewrite json (True) or not (False)
                NOTE: 'ABSORB.dat' comes from vaspkit
                
        Returns:
            dictionary of {energy (float) : {components of absorption spectra and their average (alpha) (float)}}
        """        
        fabs = os.path.join(self.calc_dir, 'ABSORB.dat')
        if not os.path.exists(fabs):
            return {}
        if not fjson:
            fjson = os.path.join(self.calc_dir, 'ABS.json')
        if not os.path.exists(fjson) or remake:
            data = {}
            with open(fabs) as f:
                count = 0
                for line in f:
                    count += 1
                    if count == 1:
                        continue
                    E, xx, yy, zz, xy, yz, zx = [float(v) for v in line[:-1].split(' ') if v != '']
                    data[E] = {'xx' : xx,
                               'yy' : yy,
                               'zz' : zz,
                               'xy' : xy,
                               'yz' : yz,
                               'zx' : zx,
                               'alpha' : np.mean([xx, yy, zz])}
            return write_json(data, fjson)
        else:
            return read_json(fjson)
        
def main():
#    calc_dir = os.path.join('tests', 'test_data', 'SCAN_geometry')
#    calc_dir = os.path.join('..', '..', 'misc', 'calcium_nitride_cohps', 'Ca2N1')
#    return VASPDOSAnalysis(calc_dir, doscar='DOSCAR.lobster').detailed_dos_dict(remake=True)
    return

if __name__ == '__main__':
    d = main()
