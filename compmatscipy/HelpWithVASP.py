import os, sys
from shutil import copyfile
import numpy as np
from compmatscipy.CompAnalyzer import CompAnalyzer
from compmatscipy.handy_functions import read_json, write_json
from scipy.integrate import simps
from compmatscipy.data import atomic_valences_data
from shutil import copyfile

def magnetic_els():
    return ['Ti', 'Zr', 'Hf',
            'V', 'Nb', 'Ta',
            'Cr', 'Mo', 'W',
            'Mn', 'Tc', 'Re',
            'Fe', 'Ru', 'Os',
            'Co', 'Rh', 'Ir',
            'Ni', 'Pd', 'Pt',
            'Cu', 'Ag', 'Au',
            'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

def make_magmom(ordered_els, els_to_sites, spin, config='fm'):
    magmom = []
    if len([el for el in ordered_els if el not in magnetic_els()]) == len(ordered_els):
        print('no magnetic elements')
        return 'CRASH THE JOB'
    for el in ordered_els:
        sites = els_to_sites[el]
        if el not in magnetic_els():
            magmom.append((len(sites), 0))
        else:
            if config == 'fm':
                magmom.append((len(sites), spin))
            elif 'afm' in config:
                i = int(config.split('-')[1])
                if len(sites) % 2:
                    print('cannot do afm for odd-numbered magnetic elements!')
                    return 'CRASH THE JOB'
                elif len(sites) == 2:
                    magmom.append((1, spin))
                    magmom.append((1, -spin))
                else:
                    if i == 1:
                        for s in range(len(sites)):
                            if s % 2:
                                magmom.append((1, spin))
                            else:
                                magmom.append((1, -spin))
                    elif i == 2:
                        for s in range(len(sites)):
                            if s < len(sites)/2:
                                magmom.append((1, spin))
                            else:
                                magmom.append((1, -spin))                          
                    elif i >= 3:
                        print('only sampling 2 configs for now')
                        return 'CRASH THE JOB'
    magmom = ['*'.join([str(v) for v in m]) for m in magmom]
    magmom = ' '.join(magmom)
    return magmom
    
def els_to_amts(ordered_els, fstructure):
    """
    Args:
    ordered_els (list) - list of els (str) in the order they appear in the structure
    fstructure (path) - path to structure file to parse
    Returns:
        dictionary of {element (str) : number in calculated structure (int)}
    """
    els = ordered_els
    with open(fstructure) as f:
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

def nsites(els_to_amts):
    """
    Args:
        els_to_amts (dict) - {el (str) : number of that el in structure (int)}
    Returns:
        number (int) of ions in calculated structure
    """
    return np.sum(list(els_to_amts.values()))
    
def idxs_to_els(ordered_els, els_to_amts, nsites):
    """
    Args:
        ordered_els (list) - list of els (str) in the order they apepar in a structure
    els_to_amts (dict) - {el (str) : number of that el in structure (int)}
    nsites (int) - number (int) of ions in calculated structure
    Returns:
        dictionary of index in structure (int) to the element (str) at that index
    """
    els = ordered_els
    els_to_sites = {}
    idx = 0
    for el in els:
        start, stop = idx, els_to_amts[el]+idx
        els_to_sites[el] = (start, stop)
        idx = stop        
    els_to_idxs = {el : range(els_to_sites[el][0], els_to_sites[el][1]) for el in els_to_sites}
    idxs_to_els = {}
    for idx in range(nsites):
        for el in els_to_idxs:
            if idx in els_to_idxs[el]:
                idxs_to_els[idx] = el
    return idxs_to_els
    
def els_to_idxs(idxs_to_els):
    """
    Args:
    idxs_to_els (dict) - {index in structure (int) : el on that site (str)}
    Returns:
    {el (str) : [idxs (int) with that el]}
    """
    els = sorted(list(set(idxs_to_els.values())))
    els_to_idxs = {el : [] for el in els}
    for idx in idxs_to_els:
        els_to_idxs[idxs_to_els[idx]].append(idx)
    return els_to_idxs

class VASPSetUp(object):
    """
    Helps set up VASP calculations
    """
    
    def __init__(self, calc_dir):
        """
        Args:
            calc_dir (str) - path to run VASP calculation
            
        Returns:
            calc_dir
        """
        self.calc_dir = calc_dir

    @property
    def mag_info(self):
        """
        IN PROGRESS
        """
        els_to_sites = self.els_to_idxs
        ordered_els = self.ordered_els_from_poscar()
        if len([el for el in ordered_els if el not in magnetic_els()]) == len(ordered_els):
            print('no magnetic elements')
            return 'nm'
        for el in ordered_els:
            sites = els_to_sites[el]
            if len(sites) % 2:
                print('cannot do afm for odd-numbered magnetic elements!')
                return 'fm'
        return 'afm'

    def incar(self, is_geometry_opt=False, functional='pbe', dos=False, dielectric=False, mag=False, piezo=False,
                    standard={'EDIFF' : 1e-6,
                              'ISMEAR' : 0,
                              'SIGMA' : 0.01,
                              'ENCUT' : 520,
                              'PREC' : 'Accurate',
                              'LORBIT' : 11,
                              'LASPH' : 'TRUE',
                              'ISIF' : 3,
                              'ISYM' : 0,
                              'NPAR' : 4}, 
                    additional={},
                    skip=[],
              MP=False,
              vdW=False,
              U=False,
              neb=False):
        """
        Args:
            is_geometry_opt (bool) - True if geometry is to be optimized
            functional (str) - 'pbe' (PBE), 'scan' (SCAN), or 'hse' (HSE06)
            dos (bool) - True if accurate DOS desired
            dielectric (bool) - True if LOPTICS on
            mag (bool or str) - False = non-magnetic; 'fm' = hs-ferro; 'afm-1' = hs-afm alternate; 'afm-2' = hs-afm blocksT
            standard (dict) - dictionary of parameters for starting point in INCAR
            additional (dict) - dictionary of parameters to enforce in INCAR
        skip (list) - parameters to exclude from INCAR
        MP (bool or str) - False or "Relax" to impose MP Relax parameters
        Returns:
            writes INCAR file to calc_dir; returns dictionary of INCAR settings
        if U not False, should be in {el : {'L' : LDAUL value, 'U' : LDAUU avlue} for all els in POSCAR}
        """
        d = {}
        
        if is_geometry_opt:
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
        elif functional == 'rtpss':
            d['METAGGA'] = 'RTPSS'
        elif functional == 'hse':
            d['GGA'] = 'PE'
            d['LHFCALC'] = 'TRUE'
            d['ALGO'] = 'Damped'
            d['TIME'] = 0.4
            d['HFSCREEN'] = 0.2
        else:
            print('are you sure you want that functional?')
            d = np.nan
            
        if functional in ['scan', 'rtpss']:
            d['LASPH'] = 'TRUE'
            d['ALGO'] = 'All'
            d['ADDGRID'] = 'TRUE'
            d['ISMEAR'] = 0
            
        if dos:
            d['NEDOS'] = 2500
        
        if dielectric:
            d['LOPTICS'] = 'TRUE'
            d['NEDOS'] = 2500
            
        if piezo:
            d['IBRION'] = 6
            d['LCALCEPS'] = 'TRUE'
            
        if vdW:
            if functional == 'pbe':
                if vdW == 'D3':
                    d['IVDW'] = 11
            if functional == 'scan':
                if vdW == 'rVV10':
                    d['LUSE_VDW'] = 'TRUE'
                if vdW == 'rVV10_2':
                    d['BPARAM'] = 15.7
        if U:
            ordered_els = self.ordered_els_from_poscar()
            nels = len(ordered_els)
            d['LDAU'] = 'TRUE'
            d['LDAUTYPE'] = 2
            d['LDAUPRINT'] = 1
            d['LDAUJ'] = ' '.join([str(0) for v in ordered_els])
            d['LDAUL'] = ' '.join([str(U[el]['L']) for el in U])
            d['LDAUU'] = ' '.join([str(U[el]['U']) for el in U])

        for k in standard:
            d[k] = standard[k]
        
        if mag:
            ordered_els = self.ordered_els_from_poscar()
            all_mag_els = magnetic_els()
            nmagnetic_els = len([el for el in ordered_els if el in all_mag_els])
            if nmagnetic_els == 0:
                d['ISPIN'] = 1
                if 'MAGMOM' not in skip:
                    skip.append('MAGMOM')
            else:
                if 'ISPIN' not in d:
                    d['ISPIN'] = 2
                if 'MAGMOM' not in d:
                    magmom = make_magmom(ordered_els, self.els_to_idxs, spin=5, config=mag)            
                    d['MAGMOM'] = magmom

        fincar = os.path.join(self.calc_dir, 'INCAR')
        if MP == 'Relax':
            from pymatgen.core.structure import Structure
            from pymatgen.io.vasp.sets import MPRelaxSet
            poscar = self.poscar()
            s = Structure.from_file(poscar)
            obj = MPRelaxSet(s)
            obj.incar.write_file(fincar)
            params = VASPBasicAnalysis(self.calc_dir).params_from_incar
            d = params
            if functional == 'rtpss':
                d['METAGGA'] = 'RTPSS'
            elif functional == 'scan':
                d['METAGGA'] = 'SCAN'
            if functional in ['rtpss', 'scan']:
                skip += ['GGA']
                d['LASPH'] = 'TRUE'
                d['ALGO'] = 'All'
                d['ADDGRID'] = 'TRUE'
                d['ISMEAR'] = 0
        
        if neb:
            defaults = {'EDIFFG' : -0.1,
                        'IBRION' : 3,
                        'ISIF' : 2,
                        'IMAGES' : 5,
                        'NELMIN' : 4,
                        'ICHAIN' : 0,
                        'IOPT' : 7,
                        'LCLIMB' : 'TRUE',
                        'LCHARG' : 'FALSE',
                        'POTIM' : 0,
                        'SPRING' : -5,
                        }
            if isinstance(neb, bool):
                neb = {}
            for param in defaults:
                d[param] = defaults[param] if param not in neb else neb[param]
                
        for k in additional:
            d[k] = additional[k]

        with open(fincar, 'w') as f:
            for k in d:
                if k not in skip:
                    f.write(' = '.join([k, str(d[k])])+'\n')
        return d

    def modify_incar(self, enforce={}):
        incar = os.path.join(self.calc_dir, 'INCAR')
        old_incar = os.path.join(self.calc_dir, 'old_INCAR')
        copyfile(incar, old_incar)
        with open(incar, 'w') as new:
            with open(old_incar) as old:
                for line in old:
                    skip = False
                    for key in enforce:
                        if key in line:
                            skip = True
                    if not skip:
                        new.write(line)
                for key in enforce:
                    line = ' = '.join([str(key), str(enforce[key])])+'\n'
                    new.write(line)
    
    def kpoints(self, discretizations=False, grid=False, kppa=False):
        """
        Args:
            discretizations (int) - False if grid to be specified; else number of discretizations for auto grid
            grid (int) - False if discretizations to be specified; else tuple of ints for grid
            kppa (int) - auto-generated Gamma or Monkhorst-pack grid with kppa / atom ^-1
            
        Returns:
            writes KPOINTS file to calc_dir
        """
        nsites = self.nsites
        els = self.ordered_els_from_poscar
        fkpoints = os.path.join(self.calc_dir, 'KPOINTS')
        with open(fkpoints, 'w') as f:
            if nsites == 1:
                f.write('Auto\n')
                f.write('0\n')
                f.write('Gamma\n')
                f.write('6 6 6\n')
                f.write('0 0 0')
                return
            elif nsites == 2:
                if els in [['O'], ['H'], ['Cl'], ['F'], ['N']]:
                    f.write('Auto\n')
                    f.write('0\n')
                    f.write('Gamma\n')
                    f.write('1 1 1\n')
                    f.write('0 0 0\n')
                    return
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
            elif kppa != False:
                from pymatgen.io.vasp.inputs import Kpoints, Poscar
                poscar = Poscar.from_file(self.poscar())
                s = poscar.structure
                Kpoints().automatic_density(s, kppa=kppa).write_file(fkpoints)
                
            else:
                print('you need to specify how to write the KPOINTS file')
                
    def poscar(self, copy_contcar=False):
        """
        Args:
            copy_contcar (bool) - if True, copies CONTCAR to POSCAR if CONTCAR not empty
            
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

    def scale_poscar(self, scaling):
        """
        Args:
        scaling (float) - how much to scale lattice vectors (1 = no scaling)

        Returns:
        scaled POSCAR
        """
        poscar = self.poscar()
        old_poscar = os.path.join(self.calc_dir, 'unscaled_POSCAR')
        copyfile(poscar, old_poscar)
        with open(old_poscar) as old:
            count = 0
            with open(poscar, 'w') as new:
                for line in old:
                    count += 1
                    if count == 2:
                        new.write('%.3f\n' % scaling)
                    else:
                        new.write(line)
        return poscar
    
    def perturb_poscar(self, perturbation):
        """
        Args:
        perturbation (float) - distance in Ang to randomly perturb ions
        
        Returns:
        POSCAR with random displacements
        """
        from pymatgen.core.structure import Structure
        
        poscar = self.poscar()
        s = Structure.from_file(poscar)
        s.perturb(perturbation)
        s.to(fmt='poscar', filename=poscar)
        return poscar

    def make_supercell(self, config):
        """
        Args:
        config (tuple) - (a, b, c) to expand poscar

        Returns: 
        supercell of POSCAR
        """
        from pymatgen.core.structure import Structure

        poscar = self.poscar()
        s = Structure.from_file(poscar)
        s.make_supercell(config)
        s.to(fmt='poscar', filename=poscar)
        return poscar
                
    def ordered_els_from_poscar(self, copy_contcar=False):
        """
        Args:
            copy_contcar (bool) - if True, copies CONTCAR to POSCAR if CONTCAR not empty 
            
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
                    
    @property
    def els_to_amts(self):
        """
        Args:
        
        Returns:
            dictionary of {element (str) : number in calculated structure (int)}
        """
        return els_to_amts(self.ordered_els_from_poscar(), self.poscar())
    
    @property
    def idxs_to_els(self):
        """
        Args:
            
        Returns:
            dictionary of index in structure (int) to the element (str) at that index
        """
        return idxs_to_els(self.ordered_els_from_poscar(), self.els_to_amts, self.nsites)

    
    @property
    def els_to_idxs(self):
        """
        """
        return els_to_idxs(self.idxs_to_els)
    
    @property
    def nsites(self):
        """
        Args:
            
        Returns:
            number (int) of ions in calculated structure
        """
        return nsites(self.els_to_amts)                      
                
    def potcar(self, els_in_poscar=False, specific_pots=False, machine='eagle', src='gga_54', MP=False):
        """
        Args:
            els_in_poscar (list or False) - ordered list of elements (str) in POSCAR; if FALSE, read POSCAR
            specific_pots (bool or dict) - False to use VASP defaults; else dict of {el : which POTCAR (str)}
            machine (str) - which computer or the path to your potcars
            src (str) - 'potpaw' implies SCAN-able POTs and configuration like ELEMENT_MOD/POTCAR
        MP (bool) - if True, use MP pseudopotentials
        Returns:
            writes POTCAR file to calc_dir
        """
        if not els_in_poscar:
            els_in_poscar = self.ordered_els_from_poscar()
        fpotcar = os.path.join(self.calc_dir, 'POTCAR')
        if machine in ['eagle', 'bridges']:
            path_to_pots = '/home/cbartel/bin/pp'
        elif machine == 'stampede2':
            path_to_pots = '/home1/06479/tg857781/bin/pp'
        elif machine == 'cori':
            path_to_pots = '/global/homes/c/cbartel/bin/pp'
        elif machine in ['savio', 'lrc']:
            path_to_pots = '/global/home/users/cbartel/bin/pp'
        if src == 'gga_54':
            pot_dir = 'POT_GGA_PAW_PBE_54'
        elif src == 'gga_52':
            pot_dir = 'POT_GGA_PAW_PBE_52'
        elif src == 'gga':
            pot_dir = 'POT_GGA_PAW_PBE'
        if MP:
            from compmatscipy.data.MP_pseudos import MP_pseudos_data
            specific_pots = MP_pseudos_data()
            pot_dir = 'POT_GGA_PAW_PBE'
            if src != 'gga':
                print('using GGA pots bc MP = TRUE')
        pots = os.path.join(path_to_pots, pot_dir)
#        if machine == 'ginar':
#            path_to_pots = '/home/cbartel/apps/pp/potpaw_PBE.54/'
#        elif machine == 'stampede2':
#            path_to_pots = '/home1/06479/tg857781/bin/VASP_PSP/POT_GGA_PAW_PBE_52'
#        if src != 'potpaw':
#            print('havent configured this yet')
#            return
        with open(fpotcar, 'w') as f:
            for el in els_in_poscar:
                if (specific_pots == False) or (el not in specific_pots):
                    pot_to_add = os.path.join(pots, el, 'POTCAR')
                else:
                    pot_to_add = os.path.join(pots, specific_pots[el], 'POTCAR')
                with open(pot_to_add) as g:
                    for line in g:
                        f.write(line)

    @property
    def lobster_orbitals(self):
        potcar = os.path.join(self.calc_dir, 'POTCAR')
        orbital_d = {}
        l_to_orb = {0 : 's',
                    1 : 'p',
                    2 : 'd',
                    3 : 'f'}
        with open(potcar) as f:
            count = 0
            overall_count = 0
            done = False
            for line in f:
                overall_count += 1
                if 'VRHFIN' in line:
                    el = line.split('=')[1].split(':')[0].strip()
                    data = {}
                    count = 0
                    done = False
                if overall_count == 2:
                    nelec = float(''.join([char for char in line[:-1] if char != ' ']))
                    overall_count = 1000
                if 'Atomic configuration' in line:
                    count += 1
                if (count > 0) and (count < 4):
                    count += 1
                elif (count >= 4) and not done:
                    if 'Description' in line:
                        done = True
                        continue
                    line = line[:-1].split(' ')
                    vals = [v for v in line if v != '']
                    n = int(vals[0])
                    l = int(vals[1])
                    occ = int(float(vals[-1]))
                    E = float(vals[3])
                    data[(n, l)] = {'n' : n,
                                    'l' : l,
                                    'occ' : occ,
                                    'E' : E,
                                    'nelec' : nelec}
                if done:
                    Es = [data[k]['E'] for k in data]
                    Es = np.sort(Es)[::-1]
                    sum_occ = 0
                    included_keys = []
                    for E in Es:
                        curr_keys = [k for k in data if (data[k]['E'] == E) and (data[k]['occ'] > 0)]
                        included_keys.extend(curr_keys)
                        sum_occ += np.sum([data[k]['occ'] for k in curr_keys])
                        if sum_occ >= nelec:
                            break
                    nls = []
                    for k in included_keys:
                        nl = ''.join([str(k[0]), l_to_orb[k[1]]])
                        nls.append(nl)
                    orbital_d[el] = nls
                    continue
                if 'End of Dataset' in line:
                    count = 0
                    overall_count = 0
        return orbital_d

class JobSubmission(object):
    
    def __init__(self, launch_dir, machine,
                 sub_file='sub.sh',
                 err_file='log.e',
                 out_file='log.o',
                 job_name='cb_job',
                 nodes=1,
                 walltime='24:00:00',
                 account=None,
                 partition=None,
                 priority=None,
                 mem=None,
                 vasp='vasp_std',
                 xcs=['pbe', 'scan'],
                 calcs=['opt', 'sp'],
                 command=False,
                 status_file='status.o',
                 postprocess={}):
        self.launch_dir = launch_dir
        self.machine = machine
        self.sub_file = sub_file
        self.err_file = err_file
        self.out_file = out_file
        self.job_name = job_name
        self.nodes = nodes
        self.walltime = walltime
        self.account = account
        self.partition = partition
        self.priority = priority
        self.mem = mem
        self.vasp = vasp
        self.xcs = xcs
        self.command = command
        self.status_file = status_file
        self.calcs = calcs
        self.postprocess = postprocess
        """
        In launch dir, generate POSCAR, KPOINTS, INCAR, POTCAR for pbe calculation


        eagle
        -partitions
        ---standard (48 hr)
        ---debug (1 node, 24 hr)
        ---short (4 hr)
        ---long (240 hr)
        ---bigmem (mem=200000)
        -vasps
        ---vasp_std, ...
        
        stampede2
        -partitions
        ---development (knl, 2 hr, 1 job)
        ---normal (knl, 48 hr, 50 jobs)
        ---large (knl, 48 hr, 5 jobs)
        ---long (knl, 120 hr, 2 jobs)
        ---skx-dev (skx, 2 hr, 1 job)
        ---skx-normal (skx, 48 hr, 25 jobs)
        ---skx-large (skx, 48 hr, 3 jobs)
        -vasps
        ---
        ---

        cori
        -partitions
        ---regular (haswell/knl, 48 hr)
        ---debug (haswell/knl, 30 min, 2 jobs)
        -vasps
        ---??
        ---??

        savio
        -partitions
        ---savio (20 cores/node)
        ---savio2 (24 cores/node)
        -qos
        ---savio_debug (4 nodes, 30 minutes)
        ---savio_normal (72 hr)

        bridges
        -partitions
        ---RM (28 cores/node)
        ---RM-small (<= 2 nodes, <= 8 hrs)
        
        lawrencium
        -partitions
        ---lr5-free --> partition=lr5, qos=condo_ceder, account=lr_ceder
        ---lr5-pc --> partition=lr5, qos=lr_normal, account=pc_ceder
        ---lr5-ac --> partition=lr5, qos=lr_normal, account=ac_ceder
        ---qos=condo_ceder
        ---time=72:00:00
        ---partition=lr5
        ---ntasks_per_node=28

        """

    @property
    def manager(self):
        machine = self.machine
        if machine in ['eagle', 'cori', 'stampede2', 'savio', 'bridges', 'lrc']:
            return '#SBATCH'
        else:
            raise ValueError

    @property
    def options(self):
        account, machine = self.account, self.machine
        if not account:
            if machine == 'eagle':
                account = 'synpred'
            elif machine == 'cori':
                account = 'm1268'
            elif machine == 'stampede2':
                account = 'TG-DMR970008S'
            elif machine == 'savio':
                account = 'fc_ceder'
            elif machine == 'bridges':
                account = 'mr7tu0p'
            elif machine == 'lrc':
                account='lr_ceder'
            else:
                raise ValueError
        partition = self.partition
        if machine == 'cori':
            qos, constraint = partition.split('-')
            partition = None
            if constraint == 'hsw':
                tasks_per_node = 32
            elif constraint == 'knl':
                tasks_per_node = 64
        else:
            qos, constraint = None, None
        if machine == 'eagle':
            tasks_per_node = 36
        elif machine == 'stampede2':
            if 'skx' in partition:
                tasks_per_node = 48
            else:
                tasks_per_node = 64
        elif machine == 'savio':
            if '2' in partition:
                tasks_per_node = 24
            else:
                tasks_per_node = 20
        elif machine == 'bridges':
            tasks_per_node = 28
        elif machine == 'lrc':
            tasks_per_node = 28
        priority = self.priority
        if priority == 'low':
            qos = 'low'
        if machine in ['savio', 'cori', 'eagle']:
            mpi_command = 'srun'
        elif machine == 'stampede2':
            mpi_command = 'ibrun'
        elif machine in ['bridges', 'lrc']:
            mpi_command = 'mpirun'
        job_name, mem, err_file, out_file, walltime, nodes = self.job_name, self.mem, self.err_file, self.out_file, self.walltime, self.nodes
        ntasks = int(nodes*tasks_per_node)
        nodes = None if machine != 'stampede2' else nodes
        if machine == 'savio':
            if (':' not in walltime) or (walltime.split(':')[1] == '30'):
                qos = 'savio_debug'
            else:
                qos = 'savio_normal'
        if machine == 'lrc':
            if not partition or (partition == 'lr5-free') or (partition == 'lr5'):
                account = 'lr_ceder'
                qos = 'condo_ceder'
                partition = 'lr5'
            elif partition == 'lr5-pc':
                account = 'pc_ceder'
                qos = 'lr_normal'
                partition = 'lr5'
            elif partition == 'lr5-ac':
                account = 'ac_ceder'
                qos = 'lr_normal'
                partition = 'lr5'
                
        slurm_options = {'account' : account,
                         'constraint' : constraint if constraint != 'hsw' else 'haswell',
                         'error' : err_file,
                         'job-name' : job_name,
                         'mem' : mem,
                         'ntasks' : ntasks,
                         'output' : out_file,
                         'partition' : partition,
                         'qos' : qos,
                         'time' : walltime,
                         'nodes' : nodes}
        return slurm_options

    @property
    def vasp_dir(self):
        machine = self.machine
        if machine == 'stampede2':
            home_dir = '/home1/06479/tg857781'
        elif machine == 'eagle':
            home_dir = '/home/cbartel'
        elif machine == 'cori':
            home_dir = '/global/homes/c/cbartel'
        elif machine == 'savio':
            home_dir = '/global/home/users/cbartel'
        elif machine == 'bridges':
            home_dir = '/home/cbartel'
        elif machine == 'lrc':
            home_dir = '/global/home/users/cbartel'
        else:
            raise ValueError
        vasp_dir = os.path.join(home_dir, 'bin', 'vasp')
        return vasp_dir

    @property
    def mpi_command(self):
        machine = self.machine
        if machine == 'stampede2':
            return 'ibrun'
        elif machine in ['cori', 'eagle']:
            return 'srun'
        elif machine in ['savio', 'bridges', 'lrc']:
            return 'mpirun'
        else:
            raise ValueError

    @property
    def vasp_command_modifier(self):
        machine, partition = self.machine, self.partition
        if (machine == 'cori') and ('knl' in partition):
            return '-c4 --cpu_bind=cores'
        elif (machine == 'cori') and ('hsw' in partition):
            return '-c2 --cpu_bind=cores'
        else:
            return ''
    
    @property
    def vasp_modifier_lines(self):
        machine, partition = self.machine, self.partition
        """
        if (machine == 'cori') and (partition == 'knl'):
            return ['\nexport OMP_PROC_BIND=true\n',
                    'export OMP_PLACES=threads\n',
                    'export OMP_NUM_THREADS=4\n']
        else:
            return ['\n']
        """
        return ['\n']
    
    @property
    def vasp_command(self):
        modifier = self.vasp_command_modifier
        mpi_command = self.mpi_command
        vasp_dir = self.vasp_dir
        vasp = self.vasp
        vasp = os.path.join(vasp_dir, vasp)
        options = self.options
        ntasks = options['ntasks']
        return '\n%s -n %s %s %s > vasp.out\n' % (mpi_command, str(ntasks), modifier, vasp)

    @property
    def bader_command(self):
        machine = self.machine
        if machine == 'eagle':
            home_dir = '/home/cbartel'
        elif machine == 'savio':
            home_dir = '/global/home/users/cbartel'
        elif machine == 'bridges':
            home_dir = '/home/cbartel'
        elif machine == 'cori':
            home_dir = '/global/homes/c/cbartel'
        elif machine == 'lrc':
            home_dir = '/global/home/users/cbartel'
        return '\n%s/bin/chgsum.pl AECCAR0 AECCAR2\n%s/bin/bader CHGCAR -ref CHGCAR_sum\n' % (home_dir, home_dir)

    def write_lobster_orbs(self, calc_dir):
        vsu = VASPSetUp(calc_dir)
        orbs = vsu.lobster_orbitals
        forbs = os.path.join(calc_dir, 'orbitals.txt')
        with open(forbs, 'w') as f:
            for el in orbs:
                f.write(' '.join([str(val) for val in ['basisfunctions', el, ' '.join(orbs[el])]])+'\n')

    @property
    def lobster_command(self):
        machine = self.machine
        if machine == 'eagle':
            home_dir = '/home/cbartel'
        elif machine == 'savio':
            home_dir = '/global/home/users/cbartel'
        elif machine == 'bridges':
            home_dir = '/home/cbartel'
        elif machine == 'cori':
            home_dir = '/global/homes/c/cbartel'
        elif machine == 'lrc':
            home_dir = '/global/home/users/cbartel'
        return '\n%s/bin/lobster-3.2.0\n' % home_dir

    def write_lobsterin(self, calc_dir, min_d=1.0, max_d=5.0, min_E=-60, max_E=20, basis='pbeVaspFit2015', orbitalwise=False):
        flob = os.path.join(calc_dir, 'lobsterin')
        with open(flob, 'w') as f:
            f.write(' '.join(['cohpGenerator', 'from', str(min_d), 'to', str(max_d)])+'\n')
#            if orbitalwise:
#                f.write('orbitalwise\n')
            f.write(' '.join(['COHPstartEnergy', str(min_E)])+'\n')
            f.write(' '.join(['COHPendEnergy', str(max_E)])+'\n')
            f.write(' '.join(['basisSet', basis])+'\n')
            vsu = VASPSetUp(calc_dir)
#            orbs = vsu.lobster_orbitals
#            for el in orbs:
#                f.write(' '.join([str(v) for v in ['basisfunctions', el, ' '.join(orbs[el])]])+'\n')
            vba = VASPBasicAnalysis(calc_dir)
            sigma = vba.params_from_incar
            if 'SIGMA' in sigma:
                sigma = sigma['SIGMA']
                f.write(' '.join(['gaussianSmearingWidth', str(sigma)])+'\n')
            f.write('userecommendedbasisfunctions\n')
            f.write('DensityOfEnergy\n')
            f.write('BWDF\n')
            f.write('BWDFCOHP\n')

    @property
    def calc_dirs(self):
        calcs = self.calcs
        launch_dir = self.launch_dir
        xcs = self.xcs
        info = {xc : {calc : {'dir' : os.path.join(launch_dir, xc, calc)} for calc in calcs} for xc in xcs}
        for xc in xcs:
            xc_dir = os.path.join(launch_dir, xc)
            if not os.path.exists(xc_dir):
                os.mkdir(xc_dir)
            for calc in calcs:
                if calc == 'neb':
                    calc_dir = launch_dir
                else:
                    calc_dir = os.path.join(xc_dir, calc)
                if not os.path.exists(calc_dir):
                    os.mkdir(calc_dir)
                outcar = os.path.join(calc_dir, 'OUTCAR')
                if not os.path.exists(outcar):
                    convergence = False
                else:
                    convergence = VASPBasicAnalysis(calc_dir).is_converged
                info[xc][calc]['convergence'] = convergence
            if calcs != ['neb']:
                if not info[xc]['opt']['convergence']:
                    for calc in calcs:
                        info[xc][calc]['convergence'] = False
        return info

    def copy_files(self, xc, calc, overwrite=False):
        if calc == 'neb':
            return
        calc_dirs = self.calc_dirs
        base_files = ['KPOINTS', 'INCAR', 'POTCAR', 'POSCAR']
        continue_files = ['WAVECAR', 'CONTCAR']
        if calc == 'sp':
            src_dir = calc_dirs[xc]['opt']['dir']
            files = continue_files + base_files
        elif calc == 'resp':
            src_dir = calc_dirs[xc]['sp']['dir']
            files = continue_files + base_files
        elif xc == 'pbe':
            src_dir = self.launch_dir
            files = base_files
        elif xc == 'scan':
            src_dir = calc_dirs['pbe'][calc]['dir']
            files = continue_files + base_files
        elif xc == 'd3':
            src_dir = calc_dirs['pbe'][calc]['dir']
            files = continue_files + base_files
        elif xc == 'rVV10':
            src_dir = calc_dirs['scan'][calc]['dir']
            files = continue_files + base_files
        elif xc == 'rVV10_2':
            src_dir = calc_dirs['scan'][calc]['dir']
            files = continue_files + base_files
        elif xc == 'pbeu':
            src_dir = calc_dirs['pbe'][calc]['dir']
            files = base_files
        elif xc == 'scan_sppbe':
            src_dir = calc_dirs['scan']['sp']['dir']
            files = ['KPOINTS', 'POSCAR', 'POTCAR']
            incar_src_dir = calc_dirs['pbe']['sp']['dir']
            incar_dst_dir = calc_dirs[xc][calc]['dir']
            incar_src = os.path.join(incar_src_dir, 'INCAR')
            incar_dst = os.path.join(incar_dst_dir, 'INCAR')
            if os.path.exists(incar_src):
                copyfile(incar_src, incar_dst)
        elif xc == 'scan_sphse':
            src_dir = calc_dirs['scan_sppbe']['opt']['dir']
            files = continue_files + base_files
        dst_dir = calc_dirs[xc][calc]['dir']
        for f in files:
            src = os.path.join(src_dir, f)
            dst = os.path.join(dst_dir, f)
            if os.path.exists(src):
                if not os.path.exists(dst) or overwrite:
                    copyfile(src, dst)

    def write_sub(self, fresh_restart=False, sp_params={}, opt_params={}, resp_params={}, copy_contcar=True, U=False):
        fstatus = self.status_file
        machine = self.machine
        sub_file = self.sub_file
        fsub = os.path.join(self.launch_dir, sub_file)
        allowed_machines = ['stampede2', 'eagle', 'cori', 'savio', 'bridges', 'lrc']
        if machine not in allowed_machines:
            raise ValueError
        line1 = '#!/bin/bash\n'
        options = self.options
        manager = self.manager
        vasp_command = self.vasp_command
        vasp = self.vasp
        xcs = self.xcs
        calcs = self.calcs
        postprocess = self.postprocess
        postprocess = {calc : [] if calc not in postprocess else postprocess[calc] for calc in calcs}
        if 'sp' in postprocess:
            if 'bader' in postprocess['sp']:
                sp_params['LAECHG'] = 'TRUE'
        if 'resp' in postprocess:
            if 'bader' in postprocess['resp']:
                resp_params['LAECHG'] = 'TRUE'
        with open(fsub, 'w') as f:
            f.write(line1)
            for tag in options:
                option = options[tag]
                if option:
                    option = str(option)
                    f.write('%s --%s=%s\n' % (manager, tag, option))
            """
            if not vasp:
                if not command:
                    raise ValueError
                f.write('\n%s\n' % command)
                return
            """
            if (machine == 'cori') and ('knl' in self.partition):
                lines_to_add = self.vasp_modifier_lines
                for l in lines_to_add:
                    f.write(l)
                mod_vasp = self.vasp_command_modifier
            f.write('\n')
            calc_dirs = self.calc_dirs
            if not U:
                xcs = [xc for xc in xcs if xc != 'pbeu']
            for xc in xcs:
                for calc in calcs:
                    if (xc == 'scan_sphse') and (calc == 'sp'):
                        continue
                    if (xc == 'scan_sppbe') and (calc == 'sp'):
                        continue
                    convergence = calc_dirs[xc][calc]['convergence']
                    calc_dir = calc_dirs[xc][calc]['dir']
                    if calc == 'resp':
                        if 'LELF' in resp_params:
                            if resp_params['LELF'] == 'TRUE':
                                if not os.path.exists(os.path.join(calc_dir, 'ELFCAR')):
                                    convergence = False
                    cohpcar = os.path.join(calc_dir, 'COHPCAR.lobster')
                    obj = VASPSetUp(calc_dir)
                    if not convergence:
                        eh = ErrorHandler(calc_dir)
                        self.copy_files(xc, calc, overwrite=fresh_restart)
                        if (xc == 'pbe') and (calc == 'opt'):
                            obj.modify_incar(enforce=opt_params)
                            obj.poscar(copy_contcar)
                        elif (xc == 'scan') and (calc == 'opt'):
                            obj.modify_incar(enforce={'METAGGA' : 'SCAN',
                                                      'ALGO' : 'All',
                                                      'ADDGRID' : 'TRUE',
                                                      'ISMEAR' : 0,
                                                      'AMIX' : 0.1,
                                                      'BMIX' : 0.0001,
                                                      'AMIX_MAG' : 0.2,
                                                      'BMIX_MAG' : 0.0001,
                                                      **opt_params})
                            obj.poscar(copy_contcar)
                        elif (xc == 'd3') and (calc == 'opt'):
                            obj.modify_incar(enforce={'IVDW' : 11, **opt_params})
                            obj.poscar(copy_contcar)
                        elif (xc == 'rVV10') and (calc == 'opt'):
                            obj.modify_incar(enforce={'LUSE_VDW' : 'TRUE', **opt_params})
                            obj.poscar(copy_contcar)
                        elif (xc == 'rVV10_2') and (calc == 'opt'):
                            obj.modify_incar(enforce={'LUSE_VDW' : 'TRUE',
                                                      'BPARAM' : 15.7,
                                                      **opt_params})
                        elif (xc == 'pbeu') and (calc == 'opt'):
                            ordered_els = VASPSetUp(calc_dir).ordered_els_from_poscar()
                            obj.modify_incar(enforce={'LDAU' : 'TRUE',
                                                      'LDAUTYPE' : 2,
                                                      'LDAUPRINT' : 1,
                                                      'LDAUJ' : ' '.join([str(0) for v in ordered_els]),
                                                      'LDAUL' : ' '.join([str(U[el]['L']) for el in ordered_els]),
                                                      'LDAUU' : ' '.join([str(U[el]['U']) for el in ordered_els])})
                        elif xc == 'scan_sppbe':
                            obj.modify_incar(enforce={'GGA' : 'PE',
                                                      'ALGO' : 'All'})
                        elif xc == 'scan_sphse':
                            obj.modify_incar(enforce={'GGA' : 'PE',
                                                      'LHFCALC' : 'TRUE',
                                                      'ALGO' : 'Damped',
                                                      'TIME' : 0.1,
                                                      'HFSCREEN' : 0.2,
                                                      'IBRION' : -1,
                                                      'NSW' : 0,
                                                      'NELM' : 1000,
                                                      'KPAR' : 2,
                                                      **sp_params})
                            
                        elif calc == 'sp':
                            obj = VASPSetUp(calc_dir)
                            obj.modify_incar(enforce={'IBRION' : -1,
                                                      'NSW' : 0,
                                                      'NELM' : 1000,
                                                      **sp_params})
                        elif calc == 'resp':
                            obj = VASPSetUp(calc_dir)
                            if 'lobster' in postprocess[calc]:
                                old_calc_dir = calc_dirs[xc]['opt']['dir']
                                vba = VASPBasicAnalysis(old_calc_dir)
                                if not vba.is_converged or os.path.exists(cohpcar):
                                    do_lobster = False
                                else:
                                    do_lobster = True
                                if do_lobster:
                                    old_nbands = vba.nbands
                                    resp_params['NBANDS'] = int(2*old_nbands)
                                    resp_params['ISYM'] = -1
                                    obj.modify_incar(enforce={'IBRION' : -1,
                                                              'NSW' : 0,
                                                              'NELM' : 1000,
                                                              **resp_params})
                        if not eh.is_clean:
                            eh.handle_errors
                        if calc in ['sp', 'resp']:
                            f.write('\ncp %s %s' % (os.path.join(calc_dirs[xc]['opt']['dir'], 'CONTCAR'), os.path.join(calc_dir, 'POSCAR')))
                        elif xc == 'scan':
                            f.write('\ncp %s %s' % (os.path.join(calc_dirs['pbe']['opt']['dir'], 'WAVECAR'), os.path.join(calc_dir, 'WAVECAR')))
                            f.write('\ncp %s %s' % (os.path.join(calc_dirs['pbe']['opt']['dir'], 'CONTCAR'), os.path.join(calc_dir, 'POSCAR')))
                        elif xc == 'scan_sphse':
                            f.write('\ncp %s %s' % (os.path.join(calc_dirs['scan_sppbe']['opt']['dir'], 'WAVECAR'), os.path.join(calc_dir, 'WAVECAR')))
                        elif xc == 'scan_sppbe':
                            f.write('\ncp %s %s' % (os.path.join(calc_dirs['scan']['sp']['dir'], 'CONTCAR'), os.path.join(calc_dir, 'POSCAR')))
                        if calc != 'neb':
                            f.write('\ncd %s' % calc_dir)
                        f.write(vasp_command)
                        f.write('\ncd %s\n' % self.launch_dir)
                        f.write('\necho launched %s-%s >> %s\n' % (xc, calc, fstatus))
                        if postprocess[calc]:
                            f.write('\ncd %s' % calc_dir)
                            if 'bader' in postprocess[calc]:
                                if not os.path.exists(os.path.join(calc_dir, 'ACF.dat')):
                                    f.write(self.bader_command)
                            if 'lobster' in postprocess[calc]:
                                old_calc_dir = calc_dirs[xc]['opt']['dir']
                                vba = VASPBasicAnalysis(old_calc_dir)
                                if not vba.is_converged or os.path.exists(cohpcar):
                                    do_lobster = False
                                else:
                                    do_lobster = True
                                if do_lobster:
                                    old_nbands = vba.nbands
                                    resp_params['NBANDS'] = int(2*old_nbands)
                                    resp_params['ISYM'] = -1
                                    obj.modify_incar(enforce={'IBRION' : -1,
                                                              'NSW' : 0,
                                                              'NELM' : 1000,
                                                              **resp_params})
                                if do_lobster:
                                    self.write_lobsterin(calc_dir)
                                    f.write(self.lobster_command)
                            f.write('\ncd %s\n' % self.launch_dir)
                    else:
                        f.write('\necho %s-%s converged >> %s\n' % (xc, calc, fstatus))
                        if postprocess[calc]:
                            f.write('\ncd %s' % calc_dir)
                            if 'bader' in postprocess[calc]:
                                if not os.path.exists(os.path.join(calc_dir, 'ACF.dat')):
                                    f.write(self.bader_command)
                            if 'lobster' in postprocess[calc]:
                                obj = VASPSetUp(calc_dir)
                                old_calc_dir = calc_dirs[xc]['opt']['dir']
                                vba = VASPBasicAnalysis(old_calc_dir)
                                if not vba.is_converged or os.path.exists(cohpcar):
                                    do_lobster = False
                                    continue
                                do_lobster = True
#                                old_nbands = vba.nbands
#                                resp_params['NBANDS'] = int(2*old_nbands)
#                                resp_params['ISYM'] = -1
#                                obj.modify_incar(enforce={'IBRION' : -1,
#                                                          'NSW' : 0,
#                                                          'NELM' : 1000,
#                                                          **resp_params})
                                if do_lobster:
                                    self.write_lobsterin(calc_dir)
                                    f.write(self.lobster_command)
                            f.write('\ncd %s\n' % self.launch_dir)

class NEBSubmission(object):
    
    def __init__(self, launch_dir, machine,
                 sub_file='sub.sh',
                 err_file='log.e',
                 out_file='log.o',
                 job_name='cb_job',
                 nodes_for_endpts=2,
                 nodes_per_img=2,
                 walltime='48:00:00',
                 account=None,
                 partition=None,
                 priority=None,
                 mem=None,
                 vasp='vasp_std',
                 xcs=['pbe'],
                 calcs=['initial_ISIF3',
                        'initial_ISIF2',
                        'final_ISIF3',
                        'final_ISIF2',
                        'ts-10-5'],
                 command=False,
                 status_file='status.o',
                 postprocess={},
                 min_ts_tol=10,
                 max_ts_tol=1,
                 min_ts_imgs=5,
                 max_ts_imgs=9):
        self.launch_dir = launch_dir
        self.machine = machine
        self.sub_file = sub_file
        self.err_file = err_file
        self.out_file = out_file
        self.job_name = job_name
        self.nodes_for_endpts = nodes_for_endpts
        self.nodes_per_img = nodes_per_img
        self.walltime = walltime
        self.account = account
        self.partition = partition
        self.priority = priority
        self.mem = mem
        self.vasp = vasp
        self.xcs = xcs
        self.command = command
        self.status_file = status_file
        self.calcs = calcs
        self.postprocess = postprocess
        self.job_submission = JobSubmission(launch_dir, machine)
        js = self.job_submission
        self.manager = js.manager
        self.options = js.options
        self.vasp_dir = js.vasp_dir
        self.mpi_command = js.mpi_command
        self.vasp_command_modifier = js.vasp_command_modifier
        self.vasp_modifier_lines = js.vasp_modifier_lines
        self.vasp_command = js.vasp_command
        self.min_ts_tol = min_ts_tol
        self.min_ts_imgs = min_ts_imgs
        self.max_ts_tol = max_ts_tol
        self.max_ts_imgs = max_ts_imgs
        
    @property
    def calc_dirs(self):
        calcs = self.calcs
        launch_dir = self.launch_dir
        xcs = self.xcs
        info = {xc : {calc : {'dir' : os.path.join(launch_dir, xc, calc)} for calc in calcs} for xc in xcs}
        for xc in xcs:
            xc_dir = os.path.join(launch_dir, xc)
            if not os.path.exists(xc_dir):
                os.mkdir(xc_dir)
            for calc in calcs:
                calc_dir = os.path.join(xc_dir, calc)
                if not os.path.exists(calc_dir):
                    os.mkdir(calc_dir)
                outcar = os.path.join(calc_dir, 'OUTCAR')
                convergence_dir = calc_dir if calc != 'ts' else os.path.join(calc_dir, '01')
                convergence = VASPBasicAnalysis(convergence_dir).is_converged
                info[xc][calc]['convergence'] = convergence
            if not info[xc]['initial_ISIF3']['convergence'] or not info[xc]['final_ISIF3']['convergence']:
                    info[xc]['initial_ISIF2']['convergence'] = False
                    info[xc]['final_ISIF2']['convergence'] = False
            if not info[xc]['initial_ISIF2']['convergence'] or not info[xc]['final_ISIF2']['convergence']:
                for calc in calcs:
                    if 'ts' in calc:
                        info[xc][calc]['convergence'] = False
        return info
    
    def copy_files(self, xc, calc, overwrite=False):
        calc_dirs = self.calc_dirs
        base_files = ['KPOINTS', 'INCAR', 'POTCAR', 'POSCAR']
        continue_files = ['WAVECAR', 'CONTCAR']
        if 'ISIF2' in calc:
            src_dir = calc_dirs[xc][calc.replace('2', '3')]['dir']
            files = continue_files + base_files
            src_dir = calc_dirs[xc]['opt']['dir']
        if 'ts' in calc:
            tol, nimgs = calc.split('-')[1:]
            src_dir = calc_dirs[xc]['initial_ISIF2']['dir']
            files = base_files
        dst_dir = calc_dirs[xc][calc]['dir']
        for f in files:
            src = os.path.join(src_dir, f)
            dst = os.path.join(dst_dir, f)
            if os.path.exists(src):
                if not os.path.exists(dst) or overwrite:
                    copyfile(src, dst)
                    
    def write_sub(self, fresh_restart=False, endpt_params={}, neb_params={}, copy_contcar=True):
        fstatus = self.status_file
        machine = self.machine
        sub_file = self.sub_file
        fsub = os.path.join(self.launch_dir, sub_file)
        allowed_machines = ['stampede2', 'eagle', 'cori', 'savio', 'bridges', 'lrc']
        if machine not in allowed_machines:
            raise ValueError
        line1 = '#!/bin/bash\n'
        options = self.options
        manager = self.manager
        vasp_command = self.vasp_command
        vasp = self.vasp
        xcs = self.xcs
        calcs = self.calcs
        with open(fsub, 'w') as f:
            f.write(line1)
            for tag in options:
                option = options[tag]
                if option:
                    option = str(option)
                    f.write('%s --%s=%s\n' % (manager, tag, option))
            if (machine == 'cori') and ('knl' in self.partition):
                lines_to_add = self.vasp_modifier_lines
                for l in lines_to_add:
                    f.write(l)
                mod_vasp = self.vasp_command_modifier
            f.write('\n')
            calc_dirs = self.calc_dirs
            for xc in xcs:
                for calc in calcs:

                    convergence = calc_dirs[xc][calc]['convergence']
                    calc_dir = calc_dirs[xc][calc]['dir']
                    obj = VASPSetUp(calc_dir)
                    if not convergence:
                        eh = ErrorHandler(calc_dir)
                        self.copy_files(xc, calc, overwrite=fresh_restart)
                        if 'ISIF3' in calc:
                            obj.modify_incar(enforce=endpt_params)
                            obj.poscar(copy_contcar)
                        elif 'ISIF2' in calc:
                            obj.modify_incar(enforce={**endpt_params, **{'ISIF' : 2}})
                            obj.poscar(copy_contcar)
                            initial_isif3 = calc_dirs[xc]['initial_ISIF3']['dir']
                            final_isif3 = calc_dirs[xc]['final_ISIF3']['dir']
                            E_initial, E_final = VASPBasicAnalysis(initial_isif3).Etot, VASPBasicAnalysis(final_isif3).Etot
                            if E_initial and E_final:
                                lattice_src_dir = initial_isif3 if E_initial <= E_final else final_isif3
                            else:
                                continue
                            fpos_lattice = os.path.join(lattice_src_dir, 'CONTCAR')
                            fpos = os.path.join(calc_dir, 'POSCAR')
                            fpos_coords = os.path.join(calc_dir, 'old_POSCAR')
                            copyfile(fpos, fpos_coords)
                            with open(fpos, 'w') as f:
                                with open(fpos_lattice) as lat:
                                    count = 0
                                    for line in lat:
                                        count += 1
                                        if count <= 8:
                                            f.write(line)
                                with open(fpos_coords) as coords:
                                    count = 0
                                    for line in coords:
                                        count += 1
                                        if count > 8:
                                            f.write(line)
                            ### PICK UP HERE!

                        elif (xc == 'scan') and (calc == 'opt'):
                            obj.modify_incar(enforce={'METAGGA' : 'SCAN',
                                                      'ALGO' : 'All',
                                                      'ADDGRID' : 'TRUE',
                                                      'ISMEAR' : 0,
                                                      'AMIX' : 0.1,
                                                      'BMIX' : 0.0001,
                                                      'AMIX_MAG' : 0.2,
                                                      'BMIX_MAG' : 0.0001,
                                                      **opt_params})
                            obj.poscar(copy_contcar)
                        elif (xc == 'd3') and (calc == 'opt'):
                            obj.modify_incar(enforce={'IVDW' : 11, **opt_params})
                            obj.poscar(copy_contcar)
                        elif (xc == 'rVV10') and (calc == 'opt'):
                            obj.modify_incar(enforce={'LUSE_VDW' : 'TRUE', **opt_params})
                            obj.poscar(copy_contcar)
                        elif (xc == 'rVV10_2') and (calc == 'opt'):
                            obj.modify_incar(enforce={'LUSE_VDW' : 'TRUE',
                                                      'BPARAM' : 15.7,
                                                      **opt_params})
                        elif (xc == 'pbeu') and (calc == 'opt'):
                            ordered_els = VASPSetUp(calc_dir).ordered_els_from_poscar()
                            obj.modify_incar(enforce={'LDAU' : 'TRUE',
                                                      'LDAUTYPE' : 2,
                                                      'LDAUPRINT' : 1,
                                                      'LDAUJ' : ' '.join([str(0) for v in ordered_els]),
                                                      'LDAUL' : ' '.join([str(U[el]['L']) for el in ordered_els]),
                                                      'LDAUU' : ' '.join([str(U[el]['U']) for el in ordered_els])})
                        elif xc == 'scan_sppbe':
                            obj.modify_incar(enforce={'GGA' : 'PE',
                                                      'ALGO' : 'All'})
                        elif xc == 'scan_sphse':
                            obj.modify_incar(enforce={'GGA' : 'PE',
                                                      'LHFCALC' : 'TRUE',
                                                      'ALGO' : 'Damped',
                                                      'TIME' : 0.1,
                                                      'HFSCREEN' : 0.2,
                                                      'IBRION' : -1,
                                                      'NSW' : 0,
                                                      'NELM' : 1000,
                                                      'KPAR' : 2,
                                                      **sp_params})
                            
                        elif calc == 'sp':
                            obj = VASPSetUp(calc_dir)
                            obj.modify_incar(enforce={'IBRION' : -1,
                                                      'NSW' : 0,
                                                      'NELM' : 1000,
                                                      **sp_params})
                        elif calc == 'resp':
                            obj = VASPSetUp(calc_dir)
                            if 'lobster' in postprocess[calc]:
                                old_calc_dir = calc_dirs[xc]['opt']['dir']
                                vba = VASPBasicAnalysis(old_calc_dir)
                                if not vba.is_converged or os.path.exists(cohpcar):
                                    do_lobster = False
                                else:
                                    do_lobster = True
                                if do_lobster:
                                    old_nbands = vba.nbands
                                    resp_params['NBANDS'] = int(2*old_nbands)
                                    resp_params['ISYM'] = -1
                                    obj.modify_incar(enforce={'IBRION' : -1,
                                                              'NSW' : 0,
                                                              'NELM' : 1000,
                                                              **resp_params})
                        if not eh.is_clean:
                            eh.handle_errors
                        if calc in ['sp', 'resp']:
                            f.write('\ncp %s %s' % (os.path.join(calc_dirs[xc]['opt']['dir'], 'CONTCAR'), os.path.join(calc_dir, 'POSCAR')))
                        elif xc == 'scan':
                            f.write('\ncp %s %s' % (os.path.join(calc_dirs['pbe']['opt']['dir'], 'WAVECAR'), os.path.join(calc_dir, 'WAVECAR')))
                            f.write('\ncp %s %s' % (os.path.join(calc_dirs['pbe']['opt']['dir'], 'CONTCAR'), os.path.join(calc_dir, 'POSCAR')))
                        elif xc == 'scan_sphse':
                            f.write('\ncp %s %s' % (os.path.join(calc_dirs['scan_sppbe']['opt']['dir'], 'WAVECAR'), os.path.join(calc_dir, 'WAVECAR')))
                        elif xc == 'scan_sppbe':
                            f.write('\ncp %s %s' % (os.path.join(calc_dirs['scan']['sp']['dir'], 'CONTCAR'), os.path.join(calc_dir, 'POSCAR')))
                        f.write('\ncd %s' % calc_dir)
                        f.write(vasp_command)
                        f.write('\ncd %s\n' % self.launch_dir)
                        f.write('\necho launched %s-%s >> %s\n' % (xc, calc, fstatus))
                        if postprocess[calc]:
                            f.write('\ncd %s' % calc_dir)
                            if 'bader' in postprocess[calc]:
                                if not os.path.exists(os.path.join(calc_dir, 'ACF.dat')):
                                    f.write(self.bader_command)
                            if 'lobster' in postprocess[calc]:
                                old_calc_dir = calc_dirs[xc]['opt']['dir']
                                vba = VASPBasicAnalysis(old_calc_dir)
                                if not vba.is_converged or os.path.exists(cohpcar):
                                    do_lobster = False
                                else:
                                    do_lobster = True
                                if do_lobster:
                                    old_nbands = vba.nbands
                                    resp_params['NBANDS'] = int(2*old_nbands)
                                    resp_params['ISYM'] = -1
                                    obj.modify_incar(enforce={'IBRION' : -1,
                                                              'NSW' : 0,
                                                              'NELM' : 1000,
                                                              **resp_params})
                                if do_lobster:
                                    self.write_lobsterin(calc_dir)
                                    f.write(self.lobster_command)
                            f.write('\ncd %s\n' % self.launch_dir)
                    else:
                        f.write('\necho %s-%s converged >> %s\n' % (xc, calc, fstatus))
                        if postprocess[calc]:
                            f.write('\ncd %s' % calc_dir)
                            if 'bader' in postprocess[calc]:
                                if not os.path.exists(os.path.join(calc_dir, 'ACF.dat')):
                                    f.write(self.bader_command)
                            if 'lobster' in postprocess[calc]:
                                obj = VASPSetUp(calc_dir)
                                old_calc_dir = calc_dirs[xc]['opt']['dir']
                                vba = VASPBasicAnalysis(old_calc_dir)
                                if not vba.is_converged or os.path.exists(cohpcar):
                                    do_lobster = False
                                    continue
                                do_lobster = True
#                                old_nbands = vba.nbands
#                                resp_params['NBANDS'] = int(2*old_nbands)
#                                resp_params['ISYM'] = -1
#                                obj.modify_incar(enforce={'IBRION' : -1,
#                                                          'NSW' : 0,
#                                                          'NELM' : 1000,
#                                                          **resp_params})
                                if do_lobster:
                                    self.write_lobsterin(calc_dir)
                                    f.write(self.lobster_command)
                            f.write('\ncd %s\n' % self.launch_dir)

class ErrorHandler(object):

    def __init__(self, calc_dir, out_file='vasp.out'):
        error_msgs = {
            "tet": ["Tetrahedron method fails for NKPT<4",
                    "Fatal error detecting k-mesh",
                    "Fatal error: unable to match k-point",
                    "Routine TETIRR needs special values",
                    "Tetrahedron method fails (number of k-points < 4)"],
            "inv_rot_mat": ["inverse of rotation matrix was not found (increase "
                            "SYMPREC)"],
            "brmix": ["BRMIX: very serious problems"],
            "subspacematrix": ["WARNING: Sub-Space-Matrix is not hermitian in "
                               "DAV"],
            "tetirr": ["Routine TETIRR needs special values"],
            "incorrect_shift": ["Could not get correct shifts"],
            "real_optlay": ["REAL_OPTLAY: internal error",
                            "REAL_OPT: internal ERROR"],
            "rspher": ["ERROR RSPHER"],
            "dentet": ["DENTET"],
            "too_few_bands": ["TOO FEW BANDS"],
            "triple_product": ["ERROR: the triple product of the basis vectors"],
            "rot_matrix": ["Found some non-integer element in rotation matrix"],
            "brions": ["BRIONS problems: POTIM should be increased"],
            "pricel": ["internal error in subroutine PRICEL"],
            "zpotrf": ["LAPACK: Routine ZPOTRF failed"],
            "amin": ["One of the lattice vectors is very long (>50 A), but AMIN"],
            "zbrent": ["ZBRENT: fatal internal in",
                       "ZBRENT: fatal error in bracketing"],
            "pssyevx": ["ERROR in subspace rotation PSSYEVX"],
            "eddrmm": ["WARNING in EDDRMM: call to ZHEGV failed"],
            "edddav": ["Error EDDDAV: Call to ZHEGV failed"],
            "grad_not_orth": [
                "EDWAV: internal error, the gradient is not orthogonal"],
            "nicht_konv": ["ERROR: SBESSELITER : nicht konvergent"],
            "zheev": ["ERROR EDDIAG: Call to routine ZHEEV failed!"],
            "elf_kpar": ["ELF: KPAR>1 not implemented"],
            "elf_ncl": ["WARNING: ELF not implemented for non collinear case"],
            "rhosyg": ["RHOSYG internal error"],
            "posmap": ["POSMAP internal error: symmetry equivalent atom not found"],
            "point_group": ["Error: point group operation missing"]
        }
        self.error_msgs = error_msgs
        self.out_file = os.path.join(calc_dir, out_file)
        self.calc_dir = calc_dir
        
    @property
    def error_log(self):
        error_msgs = self.error_msgs
        out_file = self.out_file
        errors = []
        with open(out_file) as f:
            contents = f.read()
        for e in error_msgs:
            for t in error_msgs[e]:
                if t in contents:
                    errors.append(e)
        return errors

    @property
    def is_clean(self):
        if VASPBasicAnalysis(self.calc_dir).is_converged:
            return True
        if not os.path.exists(self.out_file):
            return True
        errors = self.error_log
        if len(errors) == 0:
            return True
        with open(os.path.join(self.calc_dir, 'errors'), 'w') as f:
            for e in errors:
                f.write(e+'\n')
        return False
    
    @property
    def handle_errors(self):
        errors = self.error_log
        vsu = VASPSetUp(self.calc_dir)
        chgcar = os.path.join(self.calc_dir, 'CHGCAR')
        wavecar = os.path.join(self.calc_dir, 'WAVECAR')
        
        enforce = {}
        if 'grad_not_orth' in errors:
            enforce['SIGMA'] = 0.05
            if os.path.exists(wavecar):
                os.remove(wavecar)
            enforce['ALGO'] = 'Exact'
        if 'edddav' in errors:
            enforce['ALGO'] = 'All'
            if os.path.exists(chgcar):
                os.remove(chgcar)
        if 'eddrmm' in errors:
            if os.path.exists(wavecar):
                os.remove(wavecar)
        if 'subspacematrix' in errors:
            enforce['LREAL'] = 'FALSE'
            enforce['PREC'] = 'Accurate'
        if 'inv_rot_mat' in errors:
            enforce['SYMPREC'] = 1e-8
        if 'zheev' in errors:
            enforce['ALGO'] = 'Exact'
        if 'zpotrf' in errors:
            enforce['ISYM'] = -1
        if 'zbrent' in errors:
            enforce['IBRION'] = 1
        if 'brmix' in errors:
            enforce['IMIX'] = 1
        vsu.modify_incar(enforce=enforce)

class DiffusionSetUp(object):
    """
    Set up NEB calculations of diffusion migration barriers
    """
    def __init__(self, pristine_contcar, path_to_sites, migrating_ion, head_calc_dir):
        """
        Args:
        pristine_contcar (str) - path to optimized bulk structure
        path_to_sites (dict) - {path_idx (int) : {'path' : (site1, site2, site3, ...),
        'subpaths' : [(site1, site2), (site2, site3), ...]
        -----for now, path_to_sites is set manually by inspection of structure
        -----'path' : (site1, site2, site3, ...) means the ion moves from site1 -> site2 -> site3 -> ... in a full migration path
        migrating_ion (str) - ion that is diffusing
        head_calc_dir (str) - the parent directory where you want calculations to run
        Returns:
        
        """
        self.pristine_dir = pristine_contcar
        self.path_to_sites = path_to_sites
        self.migrating_ion = migrating_ion
        self.head_calc_dir = head_calc_dir

class SubstitutionSetUp(object):
    """
    replace ions and select low electrostatic structures
    """
    def __init__(self, template, poscar_dir, oxidation_states, host_ion, new_ion, fraction):
        from pymatgen.core.structure import Structure
        fposcar = os.path.join(poscar_dir, 'POSCAR')
        copyfile(template, fposcar)
        self.template = Structure.from_file(fposcar)
        self.oxidation_states = oxidation_states
        self.host_ion = host_ion
        self.new_ion = new_ion
        self.fraction = fraction
    
    @property
    def _partial_occ(self):
        from pymatgen.transformations.standard_transformations import SubstitutionTransformation, OxidationStateDecorationTransformation
        species_map = {self.host_ion : {self.host_ion : 1-self.fraction, self.new_ion : self.fraction}}
        obj = SubstitutionTransformation(species_map)
        new = obj.apply_transformation(self.template)
        obj = OxidationStateDecorationTransformation(self.oxidation_states)
        return obj.apply_transformation(new)
    
    def _ordered_strucs(self, n_ordered, algo=0, symmetrized_structures=False, no_oxi_states=False):
        from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
        obj = OrderDisorderedStructureTransformation(algo=algo, 
                                                     symmetrized_structures=symmetrized_structures,
                                                     no_oxi_states=no_oxi_states)
        return obj.apply_transformation(self._partial_occ, n_ordered)

    def unique_strucs(self, n_unique, 
                      n_ordered, 
                      algo=0, symmetrized_structures=False, no_oxi_states=False,
                      ltol=0.005, stol=0.005, angle_tol=0.005,
                      primitive_cell=False,
                      scale=False,
                      attempt_supercell=False):
        from pymatgen.analysis.structure_matcher import StructureMatcher
        strucs = self._ordered_strucs(n_ordered, algo=algo,
                                      symmetrized_structures=symmetrized_structures,
                                      no_oxi_states=no_oxi_states)
        checks = {i : [] for i in range(len(strucs))}
        for i in range(len(strucs)):
            for j in range(len(strucs)):
                if i == j:
                    continue
                s1, s2 = strucs[i]['structure'], strucs[j]['structure']
                test = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol, primitive_cell=primitive_cell, scale=scale, attempt_supercell=attempt_supercell).fit(s1, s2)
                if test:
                    checks[i].append(j)
        pairs = [tuple(sorted((i, j))) for i in checks for j in checks[i]]
        pairs = set(pairs)
        good_ones = []
        for i in checks:
            if len(checks[i]) == 0:
                good_ones.append(i)
        for p in pairs:
            if p[0] not in good_ones:
                good_ones.append(p[0])
        if len(good_ones) > n_unique:
            return [strucs[i] for i in good_ones][:n_unique]
        else:
            return [strucs[i] for i in good_ones]


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
        """
        Args:
            
        Returns:
            number of electrons in calculation (int)
        """
        return self.params_from_outcar(num_params=['NELECT'], str_params=[])['NELECT']
    
    @property
    def nbands(self):
        """
        Args:
            
        Returns:
            number of bands in calculation (int)
        """
        return self.params_from_outcar(num_params=['NBANDS'], str_params=[])['NBANDS']
    
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
#            print('no OUTCAR file')
            return False
        try:
            with open(outcar) as f:
                contents = f.read()
                if 'Elaps' not in contents:
                    return False
        except UnicodeDecodeError:
            with open(outcar, 'rb') as f:
                contents = f.read()
                contents = str(contents)
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
        return els_to_amts(self.ordered_els_from_outcar, self.structure)
    
    @property
    def idxs_to_els(self):
        """
        Args:
            
        Returns:
            dictionary of index in structure (int) to the element (str) at that index
        """
        return idxs_to_els(self.ordered_els_from_outcar, self.els_to_amts, self.nsites)

    
    @property
    def els_to_idxs(self):
        """
        """
        return els_to_idxs(self.idxs_to_els)
    
    @property
    def nsites(self):
        """
        Args:
            
        Returns:
            number (int) of ions in calculated structure
        """
        return nsites(self.els_to_amts)    
                
    def formula(self, reduce=False):
        """
        Args:
            reduce (bool) - if True; reduce formula to unit cell
            
        Returns:
            standardized chemical formula of calculated structure (str)
        """
        els_to_amts = self.els_to_amts
        initial = ''.join([el+str(els_to_amts[el]) for el in els_to_amts])
        return CompAnalyzer(initial).std_formula(reduce)
    
    @property
    def pseudopotentials(self):
        """
        returns dictionary of {el : {'pp' : pseudo (str), 
                                     'name' : type of pseudo (str), 
                                     'date' : date of pseudo (str), 
                                     'nval' : number of valence electrons considered (int)}}
        """
        potcar = os.path.join(self.calc_dir, 'POTCAR')
        pseudo_dict = {}
        with open(potcar) as f:
            count = 0
            for line in f:
                if 'VRHFIN' in line:
                    el = line.split('=')[1].split(':')[0].strip()
                if 'TITEL' in line:
                    tmp_dict = {}
                    line = line.split('=')[1].split(' ')
                    tmp_dict['name'] = line[1]
                    tmp_dict['pp'] = line[2]
                    tmp_dict['date'] = line[3][:-1]
                if 'ZVAL' in line:
                    line = line.split(';')[1].split('=')[1].split('mass')[0]
                    tmp_dict['nval'] = int(float(''.join([val for val in line if val != ' '])))
                    pseudo_dict[el] = tmp_dict
                    count += 1
        return pseudo_dict    
              
    @property
    def Etot(self):
        """
        Args:
            
        Returns:
            energy per atom (float) of calculated structure if converged
        """
        if not self.is_converged:
            print('calcuation is not converged')
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
                          NOTE: grepgap is a custom function
                            
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
        
    @property
    def mg_gaps(self):
        """
        """
        from pymatgen.io.vasp.outputs import Eigenval
        calc_dir = self.calc_dir
        feig = os.path.join(calc_dir, 'EIGENVAL')
        if os.path.exists(feig):
            e = Eigenval(feig)
            props = ['Eg', 'cbm', 'vbm', 'is_direct']
            out = e.eigenvalue_band_properties
            return dict(zip(props, out))
        else:
            return None
        
    @property
    def core_level_shift(self):
        """
        Args:

        Returns:
        alpha+bet in OUTCAR (float)
        """
        calc_dir = self.calc_dir
        outcar = os.path.join(calc_dir, 'OUTCAR')
        if not os.path.exists(outcar):
            print('no outcar!')
            return np.nan
        with open(outcar) as f:
            for line in f:
                if 'alpha+bet' in line:
                    shift = float(line[:-1].split('alpha+bet')[1].split(':')[1].strip())
                    return shift
                
class VASPDOSAnalysis(object):
    """
    Convert DOSCAR to useful dictionary
    TO DO: f-electrons...
    """
    
    def __init__(self, calc_dir, doscar='DOSCAR'):
        """
        Args:
            calc_dir (str) - path to VASP calculation
            doscar (str) - name of DOSCAR file to analyze (presumably 'DOSCAR' or 'DOSCAR.lobster')
            
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
                fjson = os.path.join(self.calc_dir, 'lobDOS.json')
        if not os.path.exists(self.doscar) and not os.path.exists(fjson):
            print('%s doesnt exist' % self.doscar)
            return np.nan                
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
        if isinstance(dos_dict, float):
            print('hmmm dos_dict is not a dict...')
            return np.nan
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

    def min_valence_energy(self, tol=0.01, electrons='valence', fjson=False, remake=False):
        if electrons == 'all':
            nelect = VASPBasicAnalysis(self.calc_dir).nelect
        elif electrons == 'valence':
            valence_data = atomic_valences_data()
            els_to_idxs = VASPBasicAnalysis(self.calc_dir).els_to_idxs
            nelect = 0
            for el in els_to_idxs:
                nelect += valence_data[el]*len(els_to_idxs[el])
        dos = self.energies_to_populations(fjson=fjson, remake=remake)
        if isinstance(dos, float):
            print('hmmm energies_to_populations is not a dict...')
            return np.nan
        sorted_Es = sorted(list(dos.keys()))
        if 'lobster' in self.doscar:
            Efermi = 0
        else:
            Efermi = VASPBasicAnalysis(self.calc_dir).Efermi
        occ_Es = [E for E in sorted_Es if E <= Efermi][::-1]
        for i in range(2, len(occ_Es)):
            int_Es = occ_Es[:i]
            int_doss = [dos[E] for E in int_Es]
            sum_dos = abs(simps(int_doss, int_Es))
            if sum_dos >= (1-tol)*nelect:
                return occ_Es[i]
        print('DOS doesnt integrate to %.2f percent of %i electrons' % (100*(1-tol), nelect))
        return np.nan

class DOEAnalysis(object):
    """
    Convert DensityOfEnergy.lobster into useful dictionary
    """

    def __init__(self, calc_dir):
        """
        Args:
            calc_dir (str) - path to LOBSTER calculation

        Returns:
            path to DOE to analyze
        """
        self.calc_dir = calc_dir
        self.doscar = os.path.join(calc_dir, 'DensityOfEnergy.lobster')

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
            fjson = os.path.join(self.calc_dir, 'DOE.json')        
        if not os.path.exists(self.doscar) and not os.path.exists(fjson):
            print('%s doesnt exist' % self.doscar)
            return np.nan
        if remake or not os.path.exists(fjson) or (read_json(fjson) == {}):
            basic_obj = VASPBasicAnalysis(self.calc_dir)
            num_params = ['ISPIN']
            d_params = basic_obj.params_from_outcar(num_params=num_params, str_params=[])
            spin = d_params['ISPIN']
            if spin == 2:
                spins = ['up', 'down']
                total_keys = 'E,up,down'.split(',')
            else:
                spins = ['up']
                total_keys = 'E,up'.split(',')
            with open(self.doscar) as f:
                count = 0
                for line in f:
                    count += 1
                    if count == 6:
                        nedos = int([v for v in line.split(' ') if v != ''][2])
                        break
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
                    else:
                        continue
            return write_json(data, fjson)
        else:
            data = read_json(fjson)
            return {float(k) : data[k] for k in data}

    def energies_to_populations(self, spin='summed', fjson=False, remake=False):
        """
        Args:
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
        if spin != 'summed':
            populations = [dos_dict[E]['total'][spin] for E in energies]
        else:
            populations = [np.sum([dos_dict[E]['total'][spin] for spin in spins]) for E in energies]
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
    
    def pair_dict(self, fjson=False, remake=False):
        """
        Args:
            
        Returns:
            dictionary of {pair index (int) : {'els' : (el1, el1) (str), 
                                               'sites' : (structure index for el1, structure index for el2) (int),
                                               'orbitals' : (orbital for el1, orbital for el2) (str) ('all' if all orbitals summed),
                                               'dist' : distance in Ang (float)}
                                               'energies' : [] (placeholder),
                                               'populations' : [] (placeholder)}
        """
        if not fjson:
            fjson = os.path.join(self.calc_dir, 'lobPAIRS.json')
        if remake or not os.path.exists(fjson) or (read_json(fjson) == {}):        
            lobster = self.lobster
            if not os.path.exists(lobster):
                print('%s doesnt exist' % lobster)
                return np.nan
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
                        return write_json(data, fjson)  
                    
        else:
            return read_json(fjson)

    def detailed_dos_dict(self, fjson=False, remake=False, fjson_pairs=False):
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
            if not fjson_pairs:
                fjson_pairs = os.path.join(self.calc_dir, 'lobPAIRS.json')
            data = self.pair_dict(fjson=fjson_pairs, remake=False)
            if not isinstance(data, dict):
                return np.nan
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

class BWDFAnalysis(object):
    """
    Analyzes bond-weighted distribution functions from LOBSTER code
    """

    def __init__(self, calc_dir, lobster='BWDFCOHP.lobster'):
        """
        Args:
        calc_dir (str) - path to calculation with LOBSTER output 
        lobster (str) - either 'BWDF.lobster' or 'BWDFCOHP.lobster'

        Returns:
        calc_dir
        path to LOBSTER output
        """
        self.calc_dir = calc_dir
        self.lobster = os.path.join(calc_dir, lobster)

    def bwdf(self, fjson, remake=False):
        """
        Args:
        fjson (str) - path to write; if False writes to calc_dir/BWDFcohp.json or calc_dir/BWDFcoop.json
        remake (bool) - if True, regenerate json; else read json
        
        Returns:
        dictionary of {dist (Ang) : BWDF (arb u)}
        """
        if not os.path.exists(self.lobster):
            print('%s doesnt exist' % self.lobster)
            return np.nan
        if not fjson:
            if 'COHP' in self.lobster:
                fjson = os.path.join(self.calc_dir, 'BWDFcohp.json')
            else:
                fjson = os.path.join(self.calc_dir, 'BWDFcoop.json')
        if remake or not os.path.exists(fjson) or (read_json(fjson) == {}):
            data = {}
            with open(self.lobster) as f:
                for line in f:
                    items = [float(v.strip()) for v in line[:-1].split(' ') if v != '']
                    if len(items) == 3:
                        dist, bwdf_spin1, bwdf_spin2 = items
                        data[dist] = {'spin1' : bwdf_spin1, 'spin2' : bwdf_spin2, 'bwdf' : bwdf_spin1+bwdf_spin2}
                    elif len(items) == 2:
                        dist, bwdf = items
                        data[dist] = {'bwdf' : bwdf}
                    else:
                        continue
            return write_json(data, fjson)
        else:
            return read_json(fjson)


class ProcessDOS(object):
    """
    Handles generic dictionary of {energies : states}
    Used for manipulating density of states (or equivalent) and retrieving summary statistics
    """    
    def __init__(self, energies_to_populations, 
                       shift=False,
                       cb_shift=False,
                       vb_shift=False,
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
            cb_shift (tuple or False) - shift all energies >= cb_shift[0] (float) by cb_shift[1] (float)
            vb_shift (tuple or False) - shift all energies <= vb_shift[0] (float) by vb_shift[1] (float)            
            energy_limits (list or False) - get data only for energies between (including) energy_limits[0] and energy_limits[1]
                e.g., energy_limits = [-1000, E_Fermi] would return only occupied states                
            flip_sign (True or False) - change sign of all populations
            min_population (float or False) - get data only when the population is greater than some value
                e.g., min_population = 0 would return only bonding states in the COHP (presuming flip_sign)
            max_population (float or False) - get data only when the population is less than some value
            abs_population (True or False) - make all populations >= 0
            normalization (float or False) - divide all populations by some value
            
        Returns:
            dictionary of {energy (float) : population (float)} for specified data
        """
        if isinstance(shift, float):
            energies_to_populations = {E+shift : energies_to_populations[E] for E in energies_to_populations}
        if cb_shift:
            energies_to_populations = {E+cb_shift[1] if E >= cb_shift[0] else E : energies_to_populations[E] for E in energies_to_populations}
        if vb_shift:
            energies_to_populations = {E+vb_shift[1] if E <= vb_shift[0] else E : energies_to_populations[E] for E in energies_to_populations}
        if flip_sign:
            energies_to_populations = {E : -energies_to_populations[E] for E in energies_to_populations}
        if energy_limits:
            Emin, Emax = energy_limits
            energies = [E for E in energies_to_populations if E >= Emin if E <= Emax]
            energies_to_populations = {E : energies_to_populations[E] for E in energies}
        if isinstance(min_population, float):
            energies_to_populations = {E : energies_to_populations[E] if energies_to_populations[E] >= min_population else 0. for E in energies_to_populations}
        if isinstance(max_population, float):
            energies_to_populations = {E : energies_to_populations[E] if energies_to_populations[E] <= max_population else 0. for E in energies_to_populations}
        if abs_population:
            energies_to_populations = {E : abs(energies_to_populations[E]) for E in energies_to_populations}
        if normalization:
            energies_to_populations = {E : energies_to_populations[E]/normalization for E in energies_to_populations}
        self.energies_to_populations = energies_to_populations

    def stats(self, area=True, net_area=True, energy_weighted_area=True, center=True, width=False, skewness=False, kurtosis=False):
        """
        Args:
            area (bool) - if True, compute integral of absolute populations
        net_area (bool) - if True, compute integral of populations
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
        if net_area:
            summary['net_area'] = simps(populations, energies)
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
    """
    Analyze LOPTICS=True run
    """
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
    """
    Analyze absorption spectra generated from vaspkit
    """    
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
            d = read_json(fjson)
            return {float(E) : d[E] for E in d}
        
class VASPChargeAnalysis(object):
    """
    Analyze Bader or DDEC6 output
    """    
    def __init__(self, calc_dir):
        """
        Args:
            calc_dir (str) - path to VASP calculation
            
        Returns:
            calc_dir
        """        
        self.calc_dir = calc_dir
        
    def bader(self, fjson=False, remake=False):
        """
        Args:
            fjson (str or False) - where to write dictionary; if False, bader_charge.json in calc_dir
            remake (bool) - rewrite json (True) or not (False)
            
        Returns:
            dictionary of {el (str) : {idx (int) : {charge (float)}}
        """           
        fbader = os.path.join(self.calc_dir, 'ACF.dat')
        if not os.path.exists(fbader):
            print('No Bader charge file...')
            return        
        if not fjson:
            fjson = os.path.join(self.calc_dir, 'bader_charge.json')
        if remake or not os.path.exists(fjson):
            obj = VASPBasicAnalysis(self.calc_dir)
            idx_to_els = obj.idxs_to_els
            nsites = np.max(list(idx_to_els.keys()))
            pseudos = obj.pseudopotentials            
            data = {}
            with open(fbader) as f:
                count = 0
                for line in f:
                    count += 1
                    if count < 3:
                        continue
                    if count > 3+nsites:
                        break
                    line = [v for v in line[:-1].split(' ') if v != '']
                    idx, charge = int(line[0])-1, float(line[4])
                    el = idx_to_els[idx]
                    nval = pseudos[el]['nval']
                    delta_charge = nval - charge
                    data[idx] = {'el' : el,
                                 'charge' : delta_charge}
            new = {el : {} for el in list(set(idx_to_els.values()))}
            for idx in data:
                el = data[idx]['el']
                new[el][idx] = data[idx]['charge']
            return write_json(new, fjson)
        else:
            return read_json(fjson)
        
    def ddec(self, fjson=False, remake=False):
        """
        Args:
            fjson (str or False) - where to write dictionary; if False, ddec_charge.json in calc_dir
            remake (bool) - rewrite json (True) or not (False)
            
        Returns:
            dictionary of {el (str) : {idx (int) : {charge (float)}}
        """        
        fddec = os.path.join(self.calc_dir, 'DDEC6_even_tempered_net_atomic_charges.xyz')
        if not os.path.exists(fddec):
            print('No DDEC6 charge file...')
            return        
        if not fjson:
            fjson = os.path.join(self.calc_dir, 'ddec_charge.json')
        if remake or not os.path.exists(fjson):
            obj = VASPBasicAnalysis(self.calc_dir)  
            idx_to_els = obj.idxs_to_els
            nsites = np.max(list(idx_to_els.keys()))
            data = {}            
            with open(fddec) as f:
                count = 0
                charge_count = 1e6
                for line in f:
                    count += 1
                    if 'net_charge' in line:
                        charge_count = count
                    if count >= charge_count + 1:
                        line = [v for v in line[:-1].split(' ') if v != '']
                        idx, delta_charge = int(line[0])-1, float(line[5])
                        el = idx_to_els[idx]
                        data[idx] = {'el' : el,
                                     'charge' : delta_charge}
                    if count > charge_count + nsites:
                        break
            new = {el : {} for el in list(set(idx_to_els.values()))}
            for idx in data:
                el = data[idx]['el']
                new[el][idx] = data[idx]['charge']
            return write_json(new, fjson)
        else:
            return read_json(fjson)
        
    def bonds(self, fjson=False, remake=False):
        """
        Args:
            fjson (str or False) - where to write dictionary; if False, ddec_bonds.json in calc_dir
            remake (bool) - rewrite json (True) or not (False)
            
        Returns:
            dictionary of {el (str) : {idx (int) : {'ebos' : {partner_el-partner_idx : ebo (float)},
                                                    'sbo' : summed bond order (float)}}}
        """
        fbonds = os.path.join(self.calc_dir, 'DDEC6_even_tempered_bond_orders.xyz')
        if not os.path.exists(fbonds):
            print('No DDEC6 bonds file...')
            return        
        if not fjson:
            fjson = os.path.join(self.calc_dir, 'ddec_bonds.json')
        if remake or not os.path.exists(fjson):
            obj = VASPBasicAnalysis(self.calc_dir)
            idx_to_els = obj.idxs_to_els
            nsites = np.max(list(idx_to_els.keys()))
            data = {}            
            idx = -1
            with open(fbonds) as f:
                count = 0
                for line in f:
                    count += 1
                    if ('Printing EBOs' in line) or ('Printing BOs' in line):
                        idx += 1
                        el = idx_to_els[idx]
                        tmp = {'el' : el}
                    if 'sum of bond orders for this atom' in line:
                        sbo = float([v for v in line[:-1].split('=')[1].split(' ') if v != ''][0])
                        tmp['sbo'] = sbo
                    if 'Bonded to the' in line:
                        ebo = float([v for v in line[:-1].split('=')[1].split(' ') if v != ''][0])
                        ebo_partner_idx = int([v for v in line.split('translated image of atom number')[1].split('with')[0].split(' ') if v not in ['(', ')', '']][0]) - 1
                        ebo_partner_el = idx_to_els[ebo_partner_idx]
                        tag = '-'.join([ebo_partner_el, str(ebo_partner_idx)])
                        if 'ebos' not in tmp:
                            tmp['ebos'] = {}
                        if tag not in tmp['ebos']:
                            tmp['ebos'][tag] = ebo
                        else:
                            tmp['ebos'][tag] += ebo
                    if idx >= 0:
                        data[idx] = tmp
            new = {el : {} for el in list(set(idx_to_els.values()))}
            for idx in data:
                el = data[idx]['el']
                new[el][idx] = {'ebos' : data[idx]['ebos'],
                                'sbo' : data[idx]['sbo']}
            return write_json(new, fjson)
        else:
            return read_json(fjson)

class MadelungAnalysis(object):
    """
    Analyze electrostatic energy
    """    
    def __init__(self, calc_dir):
        """
        Args:
            calc_dir (str) - path to VASP calculation
            
        Returns:
            calc_dir
        """        
        self.calc_dir = calc_dir
        
    def site_charges(self, src, oxidation_states=False, charge_json=False):
        """
        Args:
            src (str) - 'bader', 'ddec', or 'ox'
            ox_states (dict or False) - if src == 'ox', must be specified as {el (str): oxidation state (int or float)}
            charge_json (str or False) - fjson arg in VASPChargeAnalysis.bader or .ddec
        Returns:
            list of charges (float or int) ordered by sites in POSCAR
                to-be-used as input to pyamtgen.analysis.ewald.EwaldSummation
        """
        calc_dir = self.calc_dir
        if src == 'ox':
            if not oxidation_states:
                print('you specified oxidation states as a source for charges, but didnt specify the oxidation states')
                return np.nan
        if src == 'bader':
            charge_dict = VASPChargeAnalysis(calc_dir).bader(fjson=charge_json)
        if src == 'ddec':
            charge_dict = VASPChargeAnalysis(calc_dir).ddec(fjson=charge_json)
        if (src != 'ox') and not isinstance(charge_dict, dict):
            print('%s failed' % src)
            return np.nan
        idxs_to_els = VASPBasicAnalysis(calc_dir).idxs_to_els
        charge_list = []
        if src == 'ox':
            for i in idxs_to_els:
                charge_list.append(oxidation_states[idxs_to_els[i]])
        else:
            for i in idxs_to_els:
                charge_list.append(charge_dict[idxs_to_els[i]][str(i)])
        
        return charge_list      
    
    def Ewald(self, src, oxidation_states=False, charge_json=False):
        """
        Args:
            src (str) - 'bader', 'ddec', or 'ox'
            ox_states (dict or False) - if src == 'ox', must be specified as {el (str): oxidation state (int or float)}
            charge_json (str or False) - fjson arg in VASPChargeAnalysis.bader or .ddec
        Returns:
            electrostatic energy (eV/atom)
        """
        
        from pymatgen.analysis.ewald import EwaldSummation
        from pymatgen import Structure
        from pymatgen.io.vasp.inputs import Poscar        
        charge_list = self.site_charges(src, oxidation_states, charge_json)
        if not isinstance(charge_list, list):
            return np.nan
        poscar = os.path.join(self.calc_dir, 'CONTCAR')
        if not os.path.exists(poscar):
            return np.nan
        p = Poscar.from_file(poscar, check_for_POTCAR=True)
        original_s = p.structure
        s = original_s.copy()
        s.add_oxidation_state_by_site(charge_list)
        ham = EwaldSummation(s, compute_forces=True)    
        return ham.total_energy/len(s)        
        
def main():
    return

if __name__ == '__main__':
    main()
