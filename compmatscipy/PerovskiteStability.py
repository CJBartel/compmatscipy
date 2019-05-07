import numpy as np
from itertools import combinations, product
import math
from compmatscipy.CompAnalyzer import CompAnalyzer
from compmatscipy.data import atomic_electronegativities_data, shannon_revised_effective_ionic_radii_data

def fixed_cation_oxidation_states():
    """
    Args:
        
    Returns:
        dictionary of {element (str) : oxidation state (int) for cations with likely fixed oxidation states}
    """
    plus_one = ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Ag']
    plus_two = ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra']
    plus_three = ['Sc', 'Y', 'La', 'Al', 'Ga', 'In',
                'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
    fixed_cations = plus_one + plus_two + plus_three
    data = {}
    for c in fixed_cations:
        if c in plus_one:
            data[c] = 1
        elif c in plus_two:
            data[c] = 2
        elif c in plus_three:
            data[c] = 3
    return data

def fixed_anion_oxidation_states():
    """
    Args:
        
    Returns:
        dictionary of {element (str) : oxidation state (int) for anions with likely fixed oxidation states}
    """
    minus_one = ['F', 'Cl', 'Br', 'I']
    minus_two = ['O', 'S', 'Se', 'Te']
    minus_three = ['N', 'P', 'As', 'Sb']
    fixed_anions = minus_one + minus_two + minus_three
    data = {}
    for c in fixed_anions:
        if c in minus_one:
            data[c] = -1
        elif c in minus_two:
            data[c] = -2
        elif c in minus_three:
            data[c] = -3
    return data 

def allowed_anions():
    """
    Args:
        
    Returns:
        list of elements (str) that tau should be able to classify
            NOTE: only trained on ['O', 'F', 'Cl', 'Br', 'I']
    """
    return ['O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I', 'N', 'P', 'As', 'Sb']   

def t(rA, rB, rX):
    """
    Args:
        rA (float) - ionic radius of A-site
        rB (float) - ionic radius of B-site
        rX (float) - ionic radius of X-site
    
    Returns:
        Goldschmidt's tolerance factor (float)
    """
    return (rA+rX)/(np.sqrt(2)*(rB+rX))

def tau(nA, rA, rB, rX):
    """
    Args:
        nA (int) - oxidation state of A-site
        rA (float) - ionic radius of A-site
        rB (float) - ionic radius of B-site
        rX (float) - ionic radius of X-site
   
    Returns:
        Bartel's tolerance factor (float)
    """    
    return rX/rB - nA*(nA-(rA/rB)/np.log(rA/rB))
        
class SinglePerovskiteStability(object):
    """
    Predict the formability of a given ABX3 compound
    """
    def __init__(self, user_input):
        """
        Args:
            user_input (str or dict) - if str, input as ABX3; else input as {'A' : A-site, 'B' : B-site, 'X' : X-site}
                NOTE: this script will switch A and B to enforce rA > rB
        
        Returns:
            user_input if valid; else np.nan
        """
        if isinstance(user_input, str):
            self.user_input = user_input
        elif isinstance(user_input, dict):
            if not sorted(list(user_input.keys())) == ['A', 'B', 'X']:
                print('please revise dictionary to have "A", "B", and "X" as keys')
                self.user_input = np.nan
            else:
                self.user_input = user_input
        else:
            print('user input must be ABX3 formula or {"A" : A-site, "B" : B-site, "X" : X-site}')
            self.user_input = np.nan

    @property
    def is_input_formula(self):
        """
        Args:
            
        Returns:
            True if user_input is str; else False
        """
        if isinstance(self.user_input, str):
            return True
        else:
            return False
        
    @property
    def els_to_amts(self):
        """
        Args:
            
        Returns:
            dictionary of {element (str) : number of that element in formula (int)}
        """
        user_input = self.user_input
        if self.is_input_formula == True:
            els = CompAnalyzer(user_input).els
            amts = CompAnalyzer(user_input).amts()
            return dict(zip(els, amts))
        else:
            return {user_input['A'] : 1, user_input['B'] : 1, user_input['X'] : 3}

    @property
    def X(self):
        """
        Args:
            
        Returns:
            specified X-site element (str)
        """
        els_to_amts = self.els_to_amts
        X = [el for el in els_to_amts if els_to_amts[el] == 3]
        if len(X) == 1:
            anion = (X[0])
        else:
            print('you have a single perovskite... but not one element with 3 molar equivalents')
            return np.nan
        if anion not in allowed_anions():
            print('sorry tau is not designed to classify that anion')
            return np.nan
        return anion
    
    @property
    def nX(self):
        """
        Args:
            
        Returns:
            oxidation state of X-site (int)
        """
        return fixed_anion_oxidation_states()[self.X]
    
    @property
    def cations(self):
        """
        Args:
            
        Returns:
            list of cations (str)
        """
        els_to_amts = self.els_to_amts
        X = self.X
        return sorted([el for el in els_to_amts if el != X])
    
    @property
    def allowed_oxidation_states(self):
        """
        Args:
            
        Returns:
            dictionary of {element (str) : [allowed oxidation states (int)]}
        """
        X = self.X
        nX = self.nX
        allowed_ni = {X : nX}
        cations = self.cations
        fixed_ni = fixed_cation_oxidation_states()
        shannon = shannon_revised_effective_ionic_radii_data()
        for c in cations:
            if (c in fixed_ni) and (fixed_ni[c] != 3):
                allowed_ni[c] = [fixed_ni[c]]
            else:
                allowed_ni[c] = [int(n) for n in sorted(list(shannon[c].keys()))]
        return allowed_ni
    
    @property
    def charge_balanced_cation_pairs(self):
        """
        Args:
            
        Returns:
            list of dictionaries of {element (str) : oxidation state (int) for cation in cations} for charge-balanced oxidations state combination
        """
        X_charge = self.nX * 3
        allowed_ni = self.allowed_oxidation_states
        cations = self.cations
        bal_combos = []
        for n1 in allowed_ni[cations[0]]:
            for n2 in allowed_ni[cations[1]]:
                if n1+n2 == -X_charge:
                    bal_combos.append({cations[0] : n1, cations[1] : n2})
        return bal_combos

    @property    
    def cation_oxidation_states(self):
        """
        Args:
            
        Returns:
            dictionary of {element (str) : oxidation state (int)} for cations
        """
        cations = self.cations
        combos = self.charge_balanced_cation_pairs
        if len(combos) == 0:
            print('cations cannot charge-balance anion')
            return np.nan
        if len(combos) == 1:
            return combos[0]
        fixed_ni = fixed_cation_oxidation_states()
        electronegativities = atomic_electronegativities_data()
        chis = {cation : electronegativities[cation] for cation in cations}
        more_electronegative_cation = max(chis, key=chis.get)
        less_electronegative_cation = min(chis, key=chis.get)
        if (len(combos) == 2) and (combos[0][cations[0]] == combos[1][cations[1]]):
            return [combo for combo in combos if combo[less_electronegative_cation] > combo[more_electronegative_cation]][0]
        if ((cations[0] in fixed_ni) and (fixed_ni[cations[0]] == 3)) or ((cations[1] in fixed_ni) and (fixed_ni[cations[1]] == 3)):
            test_combo = {cations[0] : 3, cations[1] : 3}
            if test_combo in combos:
                return test_combo
        oxidation_state_differences = [abs(combo[cations[0]] - combo[cations[1]]) for combo in combos]
        if chis[less_electronegative_cation] > 0.9*chis[more_electronegative_cation]:
            ideal_combos = [combos[i] for i in range(len(combos)) if oxidation_state_differences[i] == np.min(oxidation_state_differences) if combos[i][more_electronegative_cation] <= combos[i][less_electronegative_cation]]
            if len(ideal_combos) > 0:
                return ideal_combos[0]
            else:
                return [combos[i] for i in range(len(combos)) if oxidation_state_differences[i] == np.min(oxidation_state_differences)][0]
        else:
            ideal_combos = [combos[i] for i in range(len(combos)) if oxidation_state_differences[i] == np.max(oxidation_state_differences) if combos[i][more_electronegative_cation] <= combos[i][less_electronegative_cation]]
            if len(ideal_combos) > 0:
                return ideal_combos[0]
            else:
                return [combos[i] for i in range(len(combos)) if oxidation_state_differences[i] == np.max(oxidation_state_differences)][0]            
            
        
    @property
    def cation_radii_at_each_site(self):
        """
        Args:
            
        Returns:
            dictionary of {element (str) : {'n' : oxidation state (int),
                                            'rA' : radius (float) if element is A-site,
                                            'rB' : radius (float) if element is B-site} for element in cations}
        """
        cations = self.cations
        n_cations = self.cation_oxidation_states
        if isinstance(n_cations, float):
            return np.nan
        shannon = shannon_revised_effective_ionic_radii_data()
        data = {c : {} for c in cations}
        for c in cations:
            n = n_cations[c]
            cns = [int(cn) for cn in shannon[c][str(n)]]
            data[c]['n'] = n
            data[c]['rB'] = shannon[c][str(n)][str(min(cns, key=lambda x:abs(x-6)))]
            data[c]['rA'] = shannon[c][str(n)][str(min(cns, key=lambda x:abs(x-12)))]
        return data
    
    @property
    def A(self):
        """
        Args:
            
        Returns:
            A-site cation (str)
        """
        cation_radii_info = self.cation_radii_at_each_site
        if isinstance(cation_radii_info, float):
            return np.nan
        cations = self.cations
        if (cation_radii_info[cations[0]]['rA'] > cation_radii_info[cations[1]]['rA']) and (cation_radii_info[cations[0]]['rB'] > cation_radii_info[cations[1]]['rB']):
            return cations[0]
        if (cation_radii_info[cations[1]]['rA'] > cation_radii_info[cations[0]]['rA']) and (cation_radii_info[cations[1]]['rB'] > cation_radii_info[cations[0]]['rB']):
            return cations[1]
        if (cation_radii_info[cations[0]]['rB'] > cation_radii_info[cations[1]]['rB']):
            return cations[0]
        if (cation_radii_info[cations[1]]['rB'] > cation_radii_info[cations[0]]['rB']):
            return cations[1]           
        if (cation_radii_info[cations[0]]['rA'] > cation_radii_info[cations[1]]['rA']):
            return cations[0]
        if (cation_radii_info[cations[1]]['rA'] > cation_radii_info[cations[0]]['rA']):
            return cations[1]
        if cation_radii_info[cations[0]]['n'] < cation_radii_info[cations[1]]['n']:
            return cations[0]
        return cations[0]
    
    @property
    def B(self):
        """
        Args:
            
        Returns:
            B-site cation (str)
        """
        if isinstance(self.cation_radii_at_each_site, float):
            return np.nan        
        cations = self.cations
        A = self.A
        return [el for el in cations if el != A][0]
    
    @property
    def assigned_oxidation_states_and_radii(self):
        """
        Args:
        
        Returns:
            dictionary of {site : {'el' : element (str),
                                   'n' : oxidation state (int),
                                   'r' : radius (float)} for site in ['A', 'B', 'X']}
        """     
        A = self.A
        B = self.B
        X = self.X
        cation_radii_at_each_site = self.cation_radii_at_each_site
        if isinstance(cation_radii_at_each_site, float):
            return np.nan           
        nA = cation_radii_at_each_site[A]['n']
        nB = cation_radii_at_each_site[B]['n']
        nX = self.nX
        rA = cation_radii_at_each_site[A]['rA']
        rB = cation_radii_at_each_site[B]['rB']
        shannon = shannon_revised_effective_ionic_radii_data()
        rX = shannon[X][str(nX)]['6']
        return {'A' : {'el' : A,
                       'n' : nA,
                       'r' : rA},
                'B' : {'el' : B,
                       'n' : nB,
                       'r' : rB},
                'X' : {'el' : X,
                       'n' : nX,
                       'r' : rX}}
    
    @property
    def t(self):
        """
        Args:
            
        Returns:
            Goldschmidt's tolerance factor (float)
                0.825 < t < 1.059 --> likely perovskite
        """
        data = self.assigned_oxidation_states_and_radii
        if isinstance(data, float):
            return np.nan
        rA, rB, rX = data['A']['r'], data['B']['r'], data['X']['r']
        return t(rA, rB, rX)
    
    @property
    def tau(self):
        """
        Args:
            
        Returns:
            Bartel's tolerance factor (float)
                tau < 4.18 --> likely perovskite
        """        
        data = self.assigned_oxidation_states_and_radii
        if isinstance(data, float):
            return np.nan        
        rA, rB, rX = data['A']['r'], data['B']['r'], data['X']['r']
        nA = data['A']['n']
        return tau(nA, rA, rB, rX)
    
    def tau_prob(self, clf):
        """
        Args:
            clf (sklearn object) - calibrated classifier based on tau
                obtainable from compmatscipy.data.calibrated_tau_prob_clf
        
        Returns:
            probability of perovskite based on tau (float)
        """
        tau = self.tau
        if math.isnan(tau) or math.isinf(tau):
            return np.nan
        X = [[tau]]
        return clf.predict_proba(X)[0][1] 
    
class DoublePerovskiteStability(object):
    """
    Predict the stability of a given AA'BB'(XX')3 compound where A can = A', B can = B', and X can = X'
    """    
    def __init__(self, user_input):
        """
        Args:
            user_input (dict) - dictionary of A, B, and X sites {'A' : A_element (str), ...}; 
                if two ions are present on a site, specify as e.g., A1, A2
        
        Returns:
            {A1, A2, B1, B2, X1, X2 : the element on that site (str)}
        """
        if not isinstance(user_input, dict):
            print('a dictionary specifying the site-occupancies is required')
            self.user_input = np.nan
        keys = sorted(list(user_input.keys()))
        new_input = {}
        for site in ['A', 'B', 'X']:
            if site in keys:
                new_input[site+'1'] = user_input[site]
                new_input[site+'2'] = user_input[site]
        required_keys = ['A1', 'A2', 'B1', 'B2', 'X1', 'X2']
        for k in required_keys:
            if k not in new_input:
                if k in user_input:
                    new_input[k] = user_input[k]
                else:
                    print('either one or two elements must be specified for A, B, and X')
                    self.user_input = np.nan
        if (new_input['A1'] == new_input['A2']) and (new_input['B1'] == new_input['B2']) and (new_input['X1'] == new_input['X2']):
            print('you should use SinglePerovskiteStability unless you are sure that A and B are correctly specified')
        self.user_input = new_input      
        
    @property
    def As(self):
        """
        Args:
            
        Returns:
            tuple of specified A-sites (str)
        """
        user_input = self.user_input
        return (user_input['A1'], user_input['A2'])
    
    @property
    def Bs(self):
        """
        Args:
            
        Returns:
            tuple of specified B-sites (str)
        """        
        user_input = self.user_input
        return (user_input['B1'], user_input['B2'])

    @property
    def Xs(self):
        """
        Args:
            
        Returns:
            tuple of specified X-sites (str)
        """        
        user_input = self.user_input
        Xs = (user_input['X1'], user_input['X2'])
        for X in Xs:
            if X not in allowed_anions():
                print('sorry %s is not allowed as an anion in this code' % X)
                return np.nan
        return Xs 
        
    @property
    def cations(self):
        """
        Args:
            
        Returns:
            list of cations (str)
        """        
        return self.As + self.Bs
    
    @property
    def allowed_oxidation_states(self):
        """
        Args:
            
        Returns:
            dictionary of {element (str) : [allowed oxidation states (int)]}
        """
        Xs = self.Xs
        allowed_ni = {X : fixed_anion_oxidation_states()[X] for X in Xs}
        cations = self.cations
        fixed_ni = fixed_cation_oxidation_states()
        shannon = shannon_revised_effective_ionic_radii_data()
        for c in cations:
            if c in fixed_ni:
                allowed_ni[c] = [fixed_ni[c]]
            else:
                allowed_ni[c] = [int(n) for n in sorted(list(shannon[c].keys()))]
        return allowed_ni
    
    @property
    def charge_balanced_cation_pairs(self):
        """
        Args:
            
        Returns:
            list of dictionaries of {element (str) : oxidation state (int) for cation in cations} for charge-balanced oxidations state combination
        """
        allowed_ni = self.allowed_oxidation_states
        Xs = self.Xs
        X_charge = np.sum([allowed_ni[X]*3 for X in Xs])
        A1, A2, B1, B2 = self.cations
        bal_combos = []
        for n1 in allowed_ni[A1]:
            for n2 in allowed_ni[A2]:
                if (A1 == A2) and (n1 != n2):
                    continue
                for n3 in allowed_ni[B1]:
                    for n4 in allowed_ni[B2]:
                        if (B1 == B2) and (n3 != n4):
                            continue                        
                        if n1+n2+n3+n4 == -X_charge:
                            bal_combos.append({A1 : n1, A2 : n2, B1 : n3, B2 : n4})
        return bal_combos
    
    @property    
    def oxidation_states(self):
        """
        Args:
            
        Returns:
            dictionary of {element (str) : oxidation state (int)} for all ions
        """
        cations = self.cations
        combos = self.charge_balanced_cation_pairs
        if len(combos) == 0:
            print('cations cannot charge-balance anion')
            return np.nan
        if len(combos) == 1:
            combo = combos[0]
        else:
            options = {c : list(set([combo[c] for combo in combos])) for c in cations}
            variable_cations = [c for c in options if len(options[c]) > 1]
            electronegativities = atomic_electronegativities_data()
            fixed_cations = {c : options[c] for c in cations if c not in variable_cations}
            chis = {cation : electronegativities[cation] for cation in variable_cations}
            most_electronegative_cation = max(chis, key=chis.get)
            least_electronegative_cation = min(chis, key=chis.get)       
            if (len(combos) == 2) and (combos[0][variable_cations[0]] == combos[1][variable_cations[1]]):
                combo = [combo for combo in combos if combo[least_electronegative_cation] >= combo[most_electronegative_cation]][0]
            else:
                ideal_combos = [combo for combo in combos if combo[least_electronegative_cation] == np.max(list(combo.values())) if combo[most_electronegative_cation] == np.min(list(combo.values()))]
                if len(ideal_combos) == 0:
                    ideal_combos = [combo for combo in combos if combo[least_electronegative_cation] == np.max(list(combo.values()))]
                if len(ideal_combos) == 0:
                    ideal_combos = [combo for combo in combos if combo[most_electronegative_cation] == np.min(list(combo.values()))]
                if len(ideal_combos) == 0:
                    ideal_combos = combos
                combos = ideal_combos
                if len(combos) == 1:
                    combo = combos[0]
                else:
                    oxidation_state_stdevs = [np.std(list(combo.values())) for combo in combos]
                    if chis[least_electronegative_cation] > 0.9*chis[most_electronegative_cation]:
                        combo = [combos[i] for i in range(len(combos)) if oxidation_state_stdevs[i] == np.min(oxidation_state_stdevs)][0]
                    else:
                        combo = [combos[i] for i in range(len(combos)) if oxidation_state_stdevs[i] == np.max(oxidation_state_stdevs)][0]
            for c in fixed_cations:
                combo[c] = options[c][0]
        allowed_oxidation_states = self.allowed_oxidation_states
        X1, X2 = self.Xs
        combo[X1] = allowed_oxidation_states[X1]
        combo[X2] = allowed_oxidation_states[X2]
        return combo        
    
    @property
    def element_specific_oxidation_states_and_radii(self):
        """
        Args:
        
        Returns:
            dictionary of {site : {'el' : element (str),
                                   'n' : oxidation state (int),
                                   'r' : radius (float)} for site in ['A1', 'A2', 'B1', 'B2', 'X1', 'X2']}
        """
        shannon = shannon_revised_effective_ionic_radii_data()
        A1, A2, B1, B2 = self.cations
        X1, X2 = self.Xs
        oxidation_states = self.oxidation_states
        if isinstance(oxidation_states, float):
            return np.nan
        info = {}
        for ion in [A1, A2, B1, B2, X1, X2]:
            n = oxidation_states[ion]
            cns = [int(cn) for cn in shannon[ion][str(n)]]
            if ion in [A1, A2]:
                cn = 12
            else:
                cn = 6
            r = shannon[ion][str(n)][str(min(cns, key=lambda x:abs(x-cn)))]
            info[ion] = {'n' : n,
                         'r' : r}
        new_info = {}
        new_info['A1'] = info[A1]
        new_info['A1']['el'] = A1
        new_info['A2'] = info[A2]
        new_info['A2']['el'] = A2
        new_info['B1'] = info[B1]
        new_info['B1']['el'] = B1
        new_info['B2'] = info[B2]
        new_info['B2']['el'] = B2
        new_info['X1'] = info[X1]
        new_info['X1']['el'] = X1
        new_info['X2'] = info[X2]
        new_info['X2']['el'] = X2
        return new_info
    
    @property
    def site_specific_oxidation_states_and_radii(self):
        """
        Args:
            
        Returns:
            dictionary of {site : {'els' : tuple of elements (str),
                                   'n' : average oxidation state (int),
                                   'r' : average radius (float)} for site in ['A', 'B', 'X']}
        """
        el_specific = self.element_specific_oxidation_states_and_radii
        if isinstance(el_specific, float):
            return np.nan
        return {site : {'els' : tuple(set([el_specific[site+'1']['el'], el_specific[site+'2']['el']])),
                        'n' : np.mean([el_specific[site+'1']['n'], el_specific[site+'2']['n']]),
                        'r' : np.mean([el_specific[site+'1']['r'], el_specific[site+'2']['r']])} for site in ['A', 'B', 'X']}        
        
    @property
    def t(self):
        """
        Args:
            
        Returns:
            Goldschmidt's tolerance factor (float)
                0.825 < t < 1.059 --> likely perovskite
        """        
        site_specific = self.site_specific_oxidation_states_and_radii
        if isinstance(site_specific, float):
            return np.nan        
        rA, rB, rX = site_specific['A']['r'], site_specific['B']['r'], site_specific['X']['r']
        return t(rA, rB, rX)
    
    @property
    def tau(self):
        """
        Args:
            
        Returns:
            Bartel's tolerance factor (float)
                tau < 4.18 --> likely perovskite
        """
        site_specific = self.site_specific_oxidation_states_and_radii
        if isinstance(site_specific, float):
            return np.nan
        nA = site_specific['A']['n']
        rA, rB, rX = site_specific['A']['r'], site_specific['B']['r'], site_specific['X']['r']
        return tau(nA, rA, rB, rX)
    
    def tau_prob(self, clf):
        """
        Args:
            clf (sklearn object) - calibrated classifier based on tau
                obtainable from compmatscipy.data.calibrated_tau_prob_clf
        
        Returns:
            probability of perovskite based on tau (float)
        """
        tau = self.tau
        if math.isnan(tau) or math.isinf(tau):
            return np.nan
        X = [[tau]]
        return clf.predict_proba(X)[0][1] 

def main():
    return

if __name__ == '__main__':
    main()