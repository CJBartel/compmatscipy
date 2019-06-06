import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import fmin_slsqp, minimize
from compmatscipy.CompAnalyzer import CompAnalyzer
from compmatscipy.handy_functions import read_json, write_json

class GetHullInputData(object):
    """
    Generates hull-relevant data
    Designed to be executed once all compounds and ground-state formation energies are known
    """
    
    def __init__(self, compound_to_energy, formation_energy_key):
        """
        Args:
            compound_to_energy (dict) - {formula (str) : {formation_energy_key (str) : formation energy (float)}}
            formation_energy_key (str) - key within compound_to_energy to use for formation energy
        
        Returns:
            dictionary of {formula (str) : formation energy (float)}
        """
        self.compound_to_energy = {k : compound_to_energy[k][formation_energy_key] for k in compound_to_energy}
        
    @property
    def compounds(self):
        """
        Args:
            
        Returns:
            list of compounds (str)
        """
        return list(self.compound_to_energy.keys())
    
    @property
    def chemical_spaces_and_subspaces(self):
        """
        Args:
            
        Returns:
            list of unique chemical spaces (tuple)
        """
        compounds = self.compounds
        return list(set([tuple(CompAnalyzer(c).els) for c in compounds]))
    
    @property
    def chemical_subspaces(self):
        """
        Args:
            
        Returns:
            list of unique chemical spaces (tuple) that do not define convex hull spaces
                (Ca, O, Ti) is the space of CaTiO3 and Ca2TiO4
                if CaTiO3 and CaO are found, (Ca, O) is a subspace
        """        
        all_spaces = self.chemical_spaces_and_subspaces
        subspaces = [all_spaces[i] for i in range(len(all_spaces)) 
                                   for j in range(len(all_spaces)) 
                                   if set(all_spaces[i]).issubset(all_spaces[j]) 
                                   if all_spaces[i] != all_spaces[j]]
        return list(set(subspaces))
    
    @property
    def hull_spaces(self):
        """
        Args:
            
        Returns:
            list of unique chemical spaces (set) that do define convex hull spaces
        """ 
        chemical_spaces_and_subspaces = self.chemical_spaces_and_subspaces
        chemical_subspaces = self.chemical_subspaces
        return [s for s in chemical_spaces_and_subspaces if s not in chemical_subspaces]
    
    def hull_data(self, fjson=False, remake=False):
        """
        Args:
            fjson (str) - file name to write hull data to
            remake (bool) - if True, write json; if False, read json
            
        Returns:
            dict of {chemical space (str) : {formula (str) : {'E' : formation energy (float),
                                                              'amts' : {el (str) : fractional amt of el in formula (float) for el in space}} 
                                            for all relevant formulas including elements}
                elements are automatically given formation energy = 0
                chemical space is now in 'el1_el2_...' format to be jsonable
        """
        if not fjson:
            fjson = 'hull_input_data.json'
        if (remake == True) or not os.path.exists(fjson):
            hull_data = {}
            hull_spaces = self.hull_spaces
            compounds = self.compounds
            compound_to_energy = self.compound_to_energy
            for space in hull_spaces:
                for el in space:
                    compound_to_energy[el] = 0
                relevant_compounds = [c for c in compounds if set(CompAnalyzer(c).els).issubset(set(space))] + list(space)
                hull_data['_'.join(list(space))] = {c : {'E' : compound_to_energy[c],
                                                         'amts' : {el : CompAnalyzer(c).fractional_amt_of_el(el=el) for el in space}}
                                                        for c in relevant_compounds}
            return write_json(hull_data, fjson)
        else:
            return read_json(fjson)
        
class AnalyzeHull(object):
    """
    Determines stability for one chemical space (hull)
    Designed to be parallelized over chemical spaces
    Ultimate output is a dictionary with hull results for one chemical space
    """
    
    def __init__(self, hull_data, chemical_space):
        """
        Args:
            hull_data (dict) - dictionary generated in GetHullInputData().hull_data
            chemical_space (str) - chemical space to analyze in 'el1_el2_...' (alphabetized) format
        
        Returns:
            grabs only the relevant sub-dict from hull_data
            changes chemical space to tuple (el1, el2, ...)
        """
        self.hull_data = hull_data[chemical_space]
        self.chemical_space = tuple(chemical_space.split('_'))
        
    @property 
    def sorted_compounds(self):
        """
        Args:
            
        Returns:
            alphabetized list of compounds (str) in specified chemical space
        """
        return sorted(list(self.hull_data.keys()))
    
    def amts_matrix(self, compounds='all', chemical_space='all'):
        """
        Args:
            compounds (str or list) - if 'all', use all compounds; else use specified list
            chemical_space - if 'all', use entire space; else use specified tuple
        
        Returns:
            matrix (2D array) with the fractional composition of each element in each compound (float)
                each row is a different compound (ordered going down alphabetically)
                each column is a different element (ordered across alphabetically)
        """
        if chemical_space == 'all':
            chemical_space = self.chemical_space
        hull_data = self.hull_data
        if compounds == 'all':
            compounds = self.sorted_compounds
        A = np.zeros((len(compounds), len(chemical_space)))
        for row in range(len(compounds)):
            compound = compounds[row]
            for col in range(len(chemical_space)):
                el = chemical_space[col]
                A[row, col] = hull_data[compound]['amts'][el]
        return A
    
    def formation_energy_array(self, compounds='all'):
        """
        Args:
            compounds (str or list) - if 'all', use all compounds; else use specified list
        
        Returns:
            array of formation energies (float) for each compound ordered alphabetically
        """        
        hull_data = self.hull_data
        if compounds == 'all':
            compounds = self.sorted_compounds
        return np.array([hull_data[c]['E'] for c in compounds])
    
    def hull_input_matrix(self, compounds='all', chemical_space='all'):
        """
        Args:
            compounds (str or list) - if 'all', use all compounds; else use specified list
            chemical_space - if 'all', use entire space; else use specified tuple
        
        Returns:
            amts_matrix, but replacing the last column with the formation energy
        """        
        A = self.amts_matrix(compounds, chemical_space)
        b = self.formation_energy_array(compounds)
        X = np.zeros(np.shape(A))
        for row in range(np.shape(X)[0]):
            for col in range(np.shape(X)[1]-1):
                X[row, col] = A[row, col]
            X[row, np.shape(X)[1]-1] = b[row]
        return X
    
    @property
    def hull(self):
        """
        Args:
            
        Returns:
            scipy.spatial.ConvexHull object
        """
        return ConvexHull(self.hull_input_matrix(compounds='all', chemical_space='all'))
    
    @property
    def hull_points(self):
        """
        Args:
            
        Returns:
            array of points (tuple) fed to ConvexHull
        """        
        return self.hull.points
    
    @property
    def hull_vertices(self):
        """
        Args:
            
        Returns:
            array of indices (int) corresponding with the points that are on the hull
        """         
        return self.hull.vertices
    
    @property
    def hull_simplices(self):
        
        return self.hull.simplices
    
    @property
    def stable_compounds(self):
        """
        Args:
            
        Returns:
            list of compounds that correspond with vertices (str)
        """          
        hull_data = self.hull_data
        hull_vertices = self.hull_vertices
        compounds = self.sorted_compounds
        return [compounds[i] for i in hull_vertices if hull_data[compounds[i]]['E'] <= 0]
    
    @property
    def unstable_compounds(self):
        """
        Args:
            
        Returns:
            list of compounds that do not correspond with vertices (str)
        """          
        compounds = self.sorted_compounds
        stable_compounds = self.stable_compounds
        return [c for c in compounds if c not in stable_compounds]
    
    def competing_compounds(self, compound):
        """
        Args:
            compound (str) - the compound (str) to analyze
        
        Returns:
            list of compounds (str) that may participate in the decomposition reaction for the input compound
        """
        compounds = self.sorted_compounds
        if compound in self.unstable_compounds:
            compounds = self.stable_compounds
        competing_compounds = [c for c in compounds if c != compound if set(CompAnalyzer(c).els).issubset(CompAnalyzer(compound).els)]
        return competing_compounds
    
    def A_for_decomp_solver(self, compound, competing_compounds):
        """
        Args:
            compound (str) - the compound (str) to analyze
            competing_compounds (list) - list of compounds (str) that may participate in the decomposition reaction for the input compound
        
        Returns:
            matrix (2D array) of elemental amounts (float) used for implementing molar conservation during decomposition solution
        """
        chemical_space = tuple(CompAnalyzer(compound).els)
        atoms_per_fu = [CompAnalyzer(c).num_atoms_in_formula() for c in competing_compounds]
        A = self.amts_matrix(competing_compounds, chemical_space)
        for row in range(len(competing_compounds)):
            for col in range(len(chemical_space)):
                A[row, col] *= atoms_per_fu[row]
        return A.T
    
    def b_for_decomp_solver(self, compound, competing_compounds):
        """
        Args:
            compound (str) - the compound (str) to analyze
            competing_compounds (list) - list of compounds (str) that may participate in the decomposition reaction for the input compound
       
        Returns:
            array of elemental amounts (float) used for implementing molar conservation during decomposition solution
        """        
        chemical_space = tuple(CompAnalyzer(compound).els)        
        return np.array([CompAnalyzer(compound).amt_of_el(el) for el in chemical_space])
    
    def Es_for_decomp_solver(self, compound, competing_compounds):
        """
        Args:
            compound (str) - the compound (str) to analyze
            competing_compounds (list) - list of compounds (str) that may participate in the decomposition reaction for the input compound
        
        Returns:
            array of formation energies per formula unit (float) used for minimization problem during decomposition solution
        """     
        atoms_per_fu = [CompAnalyzer(c).num_atoms_in_formula() for c in competing_compounds]        
        Es_per_atom = self.formation_energy_array(competing_compounds)    
        return [Es_per_atom[i]*atoms_per_fu[i] for i in range(len(competing_compounds))] 
        
    def decomp_solution(self, compound):
        """
        Args:
            compound (str) - the compound (str) to analyze
        
        Returns:
            scipy.optimize.minimize result 
                for finding the linear combination of competing compounds that minimizes the competing formation energy
        """        
        competing_compounds = self.competing_compounds(compound)
        A = self.A_for_decomp_solver(compound, competing_compounds)
        b = self.b_for_decomp_solver(compound, competing_compounds)
        Es = self.Es_for_decomp_solver(compound, competing_compounds)
        n0 = [0.01 for c in competing_compounds]
        bounds = [(0,100001) for c in competing_compounds]
        def competing_formation_energy(nj):
            nj = np.array(nj)
            return np.dot(nj, Es)
        constraints = [{'type' : 'eq',
                        'fun' : lambda x: np.dot(A, x)-b}]
        tol, maxiter, disp = 1e-4, 1000, False
        return minimize(competing_formation_energy,n0,
                     method='SLSQP',bounds=bounds,
                     constraints=constraints,tol=tol,
                     options={'maxiter' : maxiter,
                              'disp' : disp})
    
    def decomp_products(self, compound):
        """
        Args:
            compound (str) - the compound (str) to analyze
        
        Returns:
            dictionary of {competing compound (str) : {'amt' : stoich weight in decomp rxn (float),
                                                       'E' : formation energy (float)}
                                                        for all compounds in the competing reaction}
                np.nan if decomposition analysis fails
        """            
        hull_data = self.hull_data
        competing_compounds = self.competing_compounds(compound)
        
        if (len(competing_compounds) == 0) or (np.max([CompAnalyzer(c).num_els_in_formula for c in competing_compounds]) == 1):
            return {el : {'amt' : CompAnalyzer(compound).amt_of_el(el),
                          'E' : 0} for el in CompAnalyzer(compound).els}
        solution = self.decomp_solution(compound)
        if solution.success == True:
            resulting_amts = solution.x
        elif hull_data[compound]['E'] > 0:
            return {el : {'amt' : CompAnalyzer(compound).amt_of_el(el),
                          'E' : 0} for el in CompAnalyzer(compound).els}
        else:
            print(compound)
            print('\n\n\nFAILURE!!!!\n\n\n')
            print(compound)
            return np.nan
        min_amt_to_show = 1e-4
        decomp_products = dict(zip(competing_compounds, resulting_amts))
        relevant_decomp_products = [k for k in decomp_products if decomp_products[k] > min_amt_to_show]
        decomp_products = {k : {'amt' : decomp_products[k],
                                'E' : hull_data[k]['E']} if k in hull_data else 0 for k in relevant_decomp_products}
        return decomp_products
    
    def decomp_energy(self, compound):
        """
        Args:
            compound (str) - the compound (str) to analyze
        
        Returns:
            decomposition energy (float)
        """
        hull_data = self.hull_data
        decomp_products = self.decomp_products(compound)
        decomp_enthalpy = 0
        for k in decomp_products:
            decomp_enthalpy += decomp_products[k]['amt']*decomp_products[k]['E']*CompAnalyzer(k).num_atoms_in_formula()
        return (hull_data[compound]['E']*CompAnalyzer(compound).num_atoms_in_formula() - decomp_enthalpy) / CompAnalyzer(compound).num_atoms_in_formula()
    
    @property
    def hull_output_data(self):
        """
        Args:
            
        Returns:
            stability data (dict) for all compounds in the specified chemical space
                {compound (str) : {'Ef' : formation energy (float),
                                   'Ed' : decomposition energy (float),
                                   'rxn' : decomposition reaction (str),
                                   'stability' : stable (True) or unstable (False)}}
        """
        data = {}
        hull_data = self.hull_data
        compounds, stable_compounds = self.sorted_compounds, self.stable_compounds
        for c in compounds:
            if c in stable_compounds:
                stability = True
            else:
                stability = False
            Ef = hull_data[c]['E']
            Ed = self.decomp_energy(c)
            decomp_products = self.decomp_products(c)
            decomp_rxn = ['_'.join([str(np.round(decomp_products[k]['amt'], 4)), k]) for k in decomp_products]
            decomp_rxn = ' + '.join(decomp_rxn)
            data[c] = {'Ef' : Ef,
                       'Ed' : Ed,
                       'rxn' : decomp_rxn,
                       'stability' : stability}
        return data
                
def main():
    d = read_json(os.path.join('/Users/chrisbartel/Dropbox/postdoc/projects/paper-db/data/MP/MP_query_gs.json'))
    d = {k : d[k] for k in list(d.keys())[::5]}
    print(len(d))
    obj = GetHullInputData(d, 'H')
    from time import time
    start = time()
    old_spaces = obj.chemical_subspaces
    #print(spaces)
    print(len(old_spaces))
    end = time()
    print(end - start) 
    
    start = time()
    new_spaces = obj.new_chemical_subspaces
    #print(spaces)
    print(len(new_spaces))
    end = time()
    print(end - start)
    
    
    
    return d
    return

if __name__ == '__main__':
    d = main()