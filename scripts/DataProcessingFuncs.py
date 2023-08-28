import re
from libchebipy._chebi_entity import ChebiEntity
import pandas as pd
import numpy as np

elemental_order = 'CHNOSP'
element_re = re.compile("([A-Z][a-z]?)([0-9.]+[0-9.]?|(?=[A-Z])?)")

elements_df = pd.read_csv(f"../data/Elements.csv", index_col="Symbol")
# Generate dicitonary with elemental symbol as keys, MW as values
mw_dict = elements_df.loc[:,"AtomicMass"].to_dict()

class metabolite():
    def __init__(self, chebi_id=None, name=None, formula=None):
        self.name = name
        self.formula = formula
        self.chebi_id = chebi_id
        
        if chebi_id:
            self.name = ChebiEntity(f"CHEBI:{chebi_id}").get_name()
            self.formula = ChebiEntity(f"CHEBI:{chebi_id}").get_formula()
        else:
            assert formula
            
        if formula:
            self.formula = formula
    
    def elements(self, return_format="dict", elemental_order='CHNOSP', c_norm=False):
        """ 
        Takes a formula as a string as input and returns a dictionary with elements as keys and 
        values of their count as keys. 
        """
        
        formula = self.formula
        tmp_formula = formula
        if tmp_formula is None:
            return {}

        tmp_formula = str(formula)

        composition = {}
        parsed = element_re.findall(tmp_formula)
        for (element, count) in parsed:
            if count == '':
                count = float(1)
            else:
                try:
                    count = float(count)
                    int_count = int(count)
                    if count == int_count:
                        count = int_count
#                     else:
#                         warn("%s is not an integer (in formula %s)" %
#                              (count, formula))
                except ValueError:
                    warn("failed to parse %s (in formula %s)" %
                         (count, formula))
                    return None
            if element in composition:
                composition[element] += float(count)
            else:
                composition[element] = float(count)
        if return_format == "dict":
            return composition
        elif return_format == "array":
            pass
    
    def c_formula(self):
        
        formula_dict = self.elements()
        c_val = formula_dict.get('C')
        assert c_val
        formula_array = np.array(list(formula_dict.values()), 
                                       dtype='float')
        formula_array = np.nan_to_num(formula_array,0)
        c_formula_array = formula_array/c_val
        
        c_formula_dict = dict(zip(list(formula_dict.keys()),
                                  list(c_formula_array)))
        return c_formula_dict
        
    @property
    def mass(self):
        formula_dict = self.elements()
        mx_dict = {}
        for key, value in formula_dict.items():
            weight = mw_dict.get(key)
            if weight:
                mx_dict[key] = mw_dict.get(key) * value
            elif key == 'M':pass # allows for the "metals Pseudometabolite for mass balancing"
            else:
                raise ValueError(f"Specified key: {key} isn't an element symbol")
        element_mass = np.array(list(mx_dict.values()))
        total_mass = np.sum(element_mass)
        return total_mass
    
    def elemental_mass_fraction(self):
       
        formula_dict = self.elements()
        mx_dict = {}
        for key, value in formula_dict.items():
            weight = mw_dict.get(key)
            if weight:
                mx_dict[key] = mw_dict.get(key) * value
            elif key == 'M':pass # allows for the "metals Pseudometabolite for mass balancing"
            else:
                raise ValueError(f"Specified key: {key} isn't an element symbol") # Put this in earlier to prevent?
        element_mass = np.array(list(mx_dict.values()))
        mx_array = element_mass/np.sum(element_mass)
        mx_dict = dict(zip(list(mx_dict.keys()),
                           list(mx_array)))

        return mx_dict

def weighted_average_formula(formulas_dict, mol_fractions, return_type="formula"):
#     formulas = df.loc[:, formula_col].to_dict()
    mol_fractions = mol_fractions.reshape((-1,1))
    
    form_list = [metabolite(name=k, formula=v) for k,v in formulas_dict.items()]

    element_arrays = [elements_dict_to_array(x.elements()).reshape((-1,1)) for x in form_list]
    element_matrix = np.concatenate(element_arrays, axis=1)

#     mol_frac = mol_fractions
    weighted_formula_array = element_matrix.dot(mol_fractions)
    
    if return_type == "formula":
        weighted_formula = array_to_formula(weighted_formula_array.round(2), elemental_order)

        return weighted_formula
    
    elif return_type == "array":
        
        return weighted_formula_array
    
def fractions_calculator(df, value_col, mass_col, family, value_type="molar_fraction", inplace=True):
    if inplace != True:
        df = df.copy()
        
    if value_type == "molar_fraction":
#         mol_frac = df.loc[:, value_col].values
        
        df[f"g/mol {family}"] = df[value_col] * df[mass_col]
        df[f"g/g {family}"] = df[f"g/mol {family}"]/df[f"g/mol {family}"].sum()
        df[f"mmol/g {family}"] = 1000 * (df[value_col] / df[f"g/mol {family}"])
    
    if value_type == "mass_fraction":
        df[f"g/g {family}"] = df[value_col]/df[value_col].sum()
        df[f"mmol/g {family}"] = 1000 * (df[f"g/g {family}"]/df[mass_col])
        df[f"mmol/mmol {family}"] = df[f"mmol/g {family}"]/df[f"mmol/g {family}"].sum()
        
    if value_type == "mmol_gcdw_fraction":
        df[f"g/g CDW"] = df[value_col] * (df[mass_col] / 1000) # convert to mmol/g
        df[f"g/g {family}"] = df[f"g/g CDW"] / df[f"g/g CDW"].sum()
        df[f"mmol/g {family}"] = 1000 * (df[f"g/g {family}"] / df[mass_col])
        df[f"mmol/mmol {family}"] = df[f"mmol/g {family}"]/df[f"mmol/g {family}"].sum()
        
    if inplace != True:
        return df
    
def array_to_formula(array, elemental_order='CHNOSP'):
    assert isinstance(array, np.ndarray)
    array = array.flatten()
    assert len(array) == len(elemental_order)
    elem_list = [x for x in elemental_order]
    formula_dict = dict(zip(elem_list, list(array)))
    form_list = []
    for elem, val in formula_dict.items():
        if val == 0:
            continue
        elif val == 1:
            form_list.append(elem)
        else:
            if int(val) == val:
                 val = int(val)
#             form_list.append("".join([elem, f"{val:.2f}"]))
            form_list.append("".join([elem, str(val)]))
    
    formula = "".join(form_list)
    return formula

def elements_dict_to_array(dict, elemental_order='CHNOSP'):
    """
    Takes a dictionary as input and formats it as an array in the order selected by elemental_order. 
    The keys of the dictionary are the elements, the values can be either the values of the elements or the
    mass fractions of the elements. Returns a value of 0 for elements not in the formula
    """
    element_dict = {element:0 for element in elemental_order}
    # ensure formula dict values are floats
    formula_dict = {k:float(v) for k,v in dict.items()}
    element_dict.update(formula_dict)
    
    array = np.nan_to_num(np.array(list(element_dict.values()), 
                                       dtype='float'),0)

    return array
