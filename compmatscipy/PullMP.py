import os, json
import pandas as pd
import requests
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import  Poscar
import numpy as np
from compmatscipy.handy_functions import read_json, write_json, list_of_dicts_to_dict
from compmatscipy.CompAnalyzer import CompAnalyzer
from itertools import combinations

class PullMP(object):
    """
    Helper to grab Materials Proejct data
    """
    
    def __init__(self, api_key):
        """
        Args:
            api_key (str) - API key to access Materials Project
        NOTE: my key = 'N3KdATtMmcsUL94g'
        """        
        self.api_key = 'N3KdATtMmcsUL94g'
        
    @staticmethod
    def queryable_properties():
        """
        Args:
            
        Returns:
            tuple of queryable properties for Materials Project
        """
        return  ("energy", "energy_per_atom", "volume",
                 "formation_energy_per_atom", "nsites",
                 "unit_cell_formula", "pretty_formula",
                 "is_hubbard", "elements", "nelements",
                 "e_above_hull", "hubbards", "is_compatible",
                 "spacegroup", "task_ids", "band_gap", "density",
                 "icsd_id", "icsd_ids", "cif", "total_magnetization",
                 "material_id", "oxide_type", "tags", "elasticity")
        
    @property
    def rester(self):
        """
        Args:
            
        Returns:
            MPRester object
        """
        return MPRester(self.api_key)
    
    @staticmethod
    def get_ground_states_from_MP(MP_data):
        """
        Args:
            MP_data (dict) - dictionary of Materials Project data that includes polymorphs
        Returns:
            dictionary of Materials Project data with only ground-state structures (at each composition)
        """
        id_key_dict = MP_data
        compounds = list(set([CompAnalyzer(id_key_dict[k]['pretty_formula']).std_formula() for k in id_key_dict]))
        min_IDs = []
        for c in compounds:
            IDs = [k for k in id_key_dict if CompAnalyzer(id_key_dict[k]['pretty_formula']).std_formula() == c]
            energies = [id_key_dict[ID]['formation_energy_per_atom'] for ID in IDs]
            min_IDs.append(IDs[np.argsort(energies)[0]])   
        return {k : id_key_dict[k] for k in min_IDs}    
    
    def specific_query(self, fjson, tag, props, remove_polymorphs=True, remake=False, write_it=True):
        """
        Args:
            fjson (str) - where to write data (if write_it=True)
            tag (str) - chemical system (el1-el2-...), formula (Al2O3), or ID (mp-1234) on which to query
            props (list) - list of queryable properties (str)
            remove_polymorphs - if True: filter data to only include ground-state structures
            remake - if True: rewrite json; else: read json
            write_it - if True: write json; else: return dictionary
        Returns:
            dictionary of MP data corresponding with query based on tag and props
        """
        if (remake == True) or not os.path.exists(fjson):
            list_of_dicts = self.rester.get_data(tag, 'vasp', '')
            id_key_dict = list_of_dicts_to_dict(list_of_dicts, 'material_id', props)
            if remove_polymorphs == True:
                id_key_dict = self.get_ground_states_from_MP(id_key_dict)            
            if write_it == True:
                return write_json(id_key_dict, fjson)
            else:
                return id_key_dict
        else:
            return read_json(fjson)
        
    def specific_hull_query(self, fjson, elements, props, remove_polymorphs=True, remake=False, write_it=True, include_els=False):
        """
        Args:
            fjson (str) - where to write data (if write_it=True)
            elements (list) - list of elements (str) that comprise desired chemical space
            props (list) - list of queryable properties (str)
            remove_polymorphs - if True: filter data to only include ground-state structures
            remake - if True: rewrite json; else: read json
            write_it - if True: write json; else: return dictionary
            include_els (bool) - if True, also retrieve the elemental phases; else: don't
        Returns:
            dictionary of MP data corresponding with stability-related query for chemical space defined by elements
        """        
        if (remake == True) or not os.path.exists(fjson):
            spaces = [list(combinations(elements, i)) for i in range(2, len(elements)+1)]
            spaces = [j for i in spaces for j in i]
            all_data = {}
            for space in spaces:
                space = '-'.join(sorted(list(space)))
                space_data = self.specific_query('blah.json', space, props, remove_polymorphs, remake=True, write_it=False)
                all_data[space] = space_data
            if include_els:
                for el in elements:
                    all_data[el] = self.specific_query('blah.json', el, props, remove_polymorphs, True, False)
            id_key_dict = {}
            for k in all_data:
                for ID in all_data[k]:
                    id_key_dict[ID] = all_data[k][ID]
#            if remove_polymorphs == True:
#                id_key_dict = self.get_ground_states_from_MP(id_key_dict)                      
            if write_it == True:
                return write_json(id_key_dict, fjson)
            else:
                return id_key_dict
        else:
            return read_json(fjson)
                
    def big_query(self, fjson, criteria, props, remove_polymorphs=True, remake=False, write_it=True):
        """
        Args:
            fjson (str) - where to write data (if write_it=True)
            criteria (dict) - MongoDB-style query input (https://github.com/materialsproject/mapidoc)
            props (list) - list of queryable properties (str)
            remove_polymorphs - if True: filter data to only include ground-state structures
            remake - if True: rewrite json; else: read json
            write_it - if True: write json; else: return dictionary
        Returns:
            dictionary of MP data corresponding with query based on criteria and props
        """        
        if (remake == True) or not os.path.exists(fjson):  
            if 'material_id' not in props:
                props += ['material_id']
            data = {'criteria': criteria,
                    'properties': props}
            r = requests.post('https://materialsproject.org/rest/v2/query',
                             headers={'X-API-KEY': self.api_key},
                             data={k: json.dumps(v) for k,v in data.items()})
            list_of_dicts = r.json()['response']
            id_key_dict = list_of_dicts_to_dict(list_of_dicts, 'material_id', props)
            if remove_polymorphs == True:
                id_key_dict = self.get_ground_states_from_MP(id_key_dict) 
            if write_it == True:
                return write_json(id_key_dict, fjson)
            else:
                return id_key_dict
        else:
            return read_json(fjson)
        
    def get_poscar(self, material_id, fpos):
        """
        Args:
            material_id (str) - MP ID
            fpos (str) - where to write POSCAR
        Returns:
            Poscar object (also writes POSCAR)
        """
        structure = self.rester.get_structure_by_material_id(material_id)
        poscar = Poscar(structure)
        poscar.write_file(fpos)
        return poscar
        
def main():
    return

if __name__ == '__main__':
    main()
