import pandas as pd
import numpy as np
import cobra
import urllib
from cobra import Metabolite, Reaction
from cobra.flux_analysis import pfba, find_blocked_reactions, fastcc, flux_variability_analysis as fva
from cobra.flux_analysis.loopless import loopless_solution
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
import matplotlib.patches as mpatches
import matplotlib
import math
from scipy.optimize import curve_fit
from cobra.sampling import OptGPSampler
from scipy.stats import mannwhitneyu, kruskal, rankdata, norm, hypergeom
from scipy.optimize import minimize
import statsmodels.sandbox.stats.multicomp as mc


import seaborn as sns
from multiprocessing import Pool
from itertools import compress, repeat
import arviz as az
from bisect import bisect_left
import time
import warnings


atp_metabolism_stoich = {"atp_c":-1,
                         "h2o_c":-1,
                         "pi_c":1,
                         "adp_c":1,
                         "h_c":1}

def under_to_brackets(str):
    split_str = str.rsplit(r"_c", 1)
    if len(split_str) == 1:
        new = str
    else:# len(str.split("_")) == 2:
        new = split_str[0] + "[c]"
    return new

def ATPM_predict(model, qs_0):
    with model as m:
        m.reactions.Ex_glc_D.bounds = (-qs_0/(180/1000), -qs_0/(180/1000))
        m.objective = m.reactions.ATPM
        m.reactions.ATPM.bounds = (0,1000)
        result = m.optimize()
        fitted_atpm = result.fluxes.ATPM
        return fitted_atpm

def constrain_model(model,qs, qssd, qp, qpsd, qco2, qcos2d, qo2, qo2sd, qsg, o2=True, co2=True):

    model.reactions.Ex_glc_D.bounds = (-qs-qssd, -qs+qssd)
    model.reactions.Ex_vHHV.bounds = (qp-qpsd, qp+qpsd)
    
    if o2:
        model.reactions.Ex_o2.bounds = (-qo2-qo2sd, -qo2+qo2sd)
    if co2:
        model.reactions.Ex_co2.bounds = (qco2-qcos2d, qco2+qcos2d)
    if qsg > 0:
        model.reactions.SK_glycogen_c.bounds = (-1.2*qsg, -0.8*qsg)

def set_objective(model, rxn):    
    model.reactions.get_by_id(rxn).bounds = (0,1000)
    model.objective = model.reactions.get_by_id(rxn)

atp_metabolism_stoich = {"atp_c":-1,
                     "h2o_c":-1,
                     "pi_c":1,
                     "adp_c":1,
                     "h_c":1}

def additional_atp(model, biomass_rxn, extra_atp_stoich, combine=True):
    reaction = model.reactions.get_by_id(biomass_rxn)
    extra_atp_stoich = {k:v*extra_atp_stoich for k,v in atp_metabolism_stoich.items()}
    reaction.add_metabolites(extra_atp_stoich, combine=combine)



def add_biomass(model, stoich_data, met_model_dict, equation_name="Dynamic"):
    
    # create three dictionaries for renaming etc. one converts to the macromolecule class, one to mass and the last is macro to model met id
    met_to_macro = met_model_dict.set_index("Model ID").Macrocomponent.to_dict()
    met_to_mass = met_model_dict.set_index("Model ID")["Average Mass"].to_dict()
    macro_to_model_dict = {"Carbohydrates":"CARBOHYDRATES_c", "Protein":"PROTEIN_c", "Lipids":"LIPIDS_c", "RNA":"RNA_c", "DNA":"DNA_c"}
    
    # take the stoich_data calculated and with ATP included, drop the extras added, we will add them back later, then make positive
    sliced = stoich_data.loc[~stoich_data.index.isin(['atp_c', 'h2o_c', 'h_c', 'pi_c', 'adp_c', 'biomass_c'])]
    extras = stoich_data.loc[stoich_data.index.isin(['atp_c', 'h2o_c', 'h_c', 'pi_c', 'adp_c', 'biomass_c'])]
    sliced = -sliced
    
    # convert to g component per gcdw
    g_gcdw = sliced.apply(lambda row: row* (met_to_mass[row.name]/1000), axis=1)
    # change macro clas snames to model names and then calculate calculate the g_gcdw of macro components
    sum_classes_gcdw = g_gcdw.rename(index=met_to_macro)
    sum_classes_gcdw = sum_classes_gcdw.groupby(sum_classes_gcdw.index).sum()
    sum_classes_wmodel_ids = sum_classes_gcdw.rename(index=macro_to_model_dict)
    # convert to mmol_gclass so that we can create sub macro classes represnted in mmol_g comp
    mmol_gclass = sliced.apply(lambda row: row/sum_classes_gcdw.loc[met_to_macro[row.name]],axis=1)
    # rmeove DNA and RNA as there are already static representations for these in the model
    mmol_gclass = mmol_gclass.drop(["DNA_c", "RNA_c"])
    # find out macro names for each metabolite
    mmol_gclass["macro"] = mmol_gclass.apply(lambda row:met_to_macro[row.name], axis=1)
   
    # reset index then make mutliple index
    mmol_gclass = mmol_gclass.reset_index().set_index(["macro", "index"])

    
    for name, stoich_col in mmol_gclass.iteritems():
    
    
        reaction = Reaction(f'{equation_name}Protein{name}')
        model.add_reactions([reaction])
        reaction.name = f'{equation_name} Protein composition time={name} mmol/g protein'
        reaction.subsystem = 'Biomass Composition'
        prot_stoich = (-stoich_col).Protein.to_dict()
        prot_stoich["PROTEIN_c"] = 1
        reaction.add_metabolites(prot_stoich)
        reaction.bounds = (0,0)

        reaction = Reaction(f'{equation_name}Carbohydrate{name}')
        model.add_reactions([reaction])
        reaction.name = f'{equation_name} Carbohydrate composition time={name} mmol/g carbohydrate'
        reaction.subsystem = 'Biomass Composition'
        carb_stoich = (-stoich_col).Carbohydrates.to_dict()
        carb_stoich["CARBOHYDRATES_c"] = 1
        reaction.add_metabolites(carb_stoich)
        reaction.bounds = (0,0)

        reaction = Reaction(f'{equation_name}Lipid{name}')
        model.add_reactions([reaction])
        reaction.name = f'{equation_name} Lipid composition time={name} mmol/g lipid'
        reaction.subsystem = 'Biomass Composition'
        lip_stoich = (-stoich_col).Lipids.to_dict()
        lip_stoich["LIPIDS_c"] = 1
        reaction.add_metabolites(lip_stoich)
        reaction.bounds = (0,0)

        reaction = Reaction(f'{equation_name}Biomass{name}')
        model.add_reactions([reaction])
        reaction.name = f'{equation_name} Biomass composition time={name}. Macros g_gcdw, others mmol_gcdw'
        reaction.subsystem = 'Biomass Composition'
        biomass_stoich = (-sum_classes_wmodel_ids[name]).to_dict()
        biomass_stoich.update(extras[name].to_dict())
        biomass_stoich["cof_c"] = -1
        reaction.add_metabolites(biomass_stoich)
        reaction.bounds = (0,0)

def remove_blocked(model,flux_span,solution):
    
    rxns = solution.fluxes[solution.fluxes.abs() < 1e-6].index.to_list()
    
    to_del = flux_span[(flux_span["minimum"] == 0)&(flux_span["maximum"] == 0)].index.to_list()
    forward = flux_span[(flux_span["minimum"] == 0)&(flux_span["maximum"] != 0)].index.to_list()
    reverse = flux_span[(flux_span["minimum"] != 0)&(flux_span["maximum"] == 0)].index.to_list()
    
    to_del = list(set(to_del).intersection(set(rxns)))
    
    print(f"{len(model.reactions)} reactions before making consistent, {len(model.reactions) - len(to_del)} after")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model.remove_reactions(to_del, remove_orphans=True)   

def constrain_predict(model, cultivation_parameters, o2=True, co2=True, slim=False, parsimonious=False, 
                      sampling=False, fluxva=False, fopt=1, verbose=False, chains=1, thinning=10000, 
                      loopless=False, processes=24, flux_span=None, objective_sense="maximize",n_samples=5000):
                     
    

    mu, musd, qs, qssd, qp, qpsd, qco2, qcos2d, qo2, qo2sd, qsg = cultivation_parameters 
    
    with model as temp_model:
        constrain_model(temp_model, qs, qssd, qp, qpsd, qco2, qcos2d, qo2, qo2sd, qsg, o2=o2, co2=co2)
        
        if slim:
            if loopless:
                try:
                    sol = temp_model.optimize()
                    print(sol.objective_value)
                    result = loopless_solution(temp_model, sol.fluxes.to_dict())
                    
                except: 
                    result = None
                   
            else:
                
                result = temp_model.slim_optimize()
                if math.isnan(result):
                    result = 1e8
                    if verbose:
                        test = temp_model.optimize()
                        print(test.shadow_prices.sort_values().tail(50))
                        print(test.shadow_prices.sort_values().head(50))
            
        elif parsimonious:
            try:
                result = pfba(temp_model, fraction_of_optimum=fopt)
            except: 
                result = None
                print(f"Failed - {temp_model.slim_optimize()}")
        elif sampling:
            # Thinning = Less correlated smaples but longer computation time
            # default thinning is 100, i.e. 1 in 100. 

            gp = OptGPSampler(temp_model, processes=processes, thinning=thinning, seed=42)
    
            ## can get a lot of warnings when smapling but no real issues
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                if chains==1:
                    samples = gp.sample(n_samples)
                    result = samples[gp.validate(samples) == 'v']

                    if loopless:

                        pool = Pool(processes=processes)
                        _, sols = zip(*result.iterrows())
                        corrected = pool.starmap(sampling_loopless_convert, zip(sols, repeat(temp_model)))
                        pool.close()

                        result = pd.concat(corrected, axis=1).T.reset_index(drop=True)                    

                elif chains >1:
                    samples = [gp.sample(n_samples) for x in np.arange(chains)]

                    result = [sample[gp.validate(sample) == 'v'] for sample in samples]
                    if loopless:
                        all_result = []
                        for sample in result:
                            pool = Pool(processes=processes)
                            _, sols = zip(*sample.iterrows())
                            corrected = pool.starmap(sampling_loopless_convert, zip(sols, repeat(temp_model)))
                            pool.close()

                            all_result.append(pd.concat(corrected, axis=1).T.reset_index(drop=True))
                        result = all_result    
        elif fluxva:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                result = fva(temp_model, loopless=loopless, processes=processes, fraction_of_optimum=fopt)
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    result = temp_model.optimize(objective_sense) 
            except: 
                result = None
                print("Failed")
        return result

def sampling_loopless_convert(sol, model):
    result = loopless_solution(model, sol.to_dict())
    return result.fluxes

def eWRSS(y, y_hat, y_sd):
#     Calcualte the relative distances. 
# COmpute distance of y and y_hat, divide by y_hat to get relative ratio then square and sum these ratios for all points
# Allows for all points to be equally weighted and prevent larger values dominating
    result = y - y_hat
    result = np.sum((result/y_sd)**2)
    return result

def RSS(y, y_hat):
#     Calcualte the relative distances. 
# COmpute distance of y and y_hat, divide by y_hat to get relative ratio then square and sum these ratios for all points
# Allows for all points to be equally weighted and prevent larger values dominating
    result = y - y_hat
    result = np.sum(result**2)
    return result


def DetermineGAME(model, cultivation_parameters, o2=True, co2=True, biomass_type="Consensus", method="RSS", individual=False, combine=True, loopless=False):

    stoich_to_check = np.linspace(0,200,101)
    
    with model as temp_model:
        result =[ATPStoichFit(x, temp_model, cultivation_parameters, biomass_type, o2, co2,  method, individual, combine, loopless) for x in stoich_to_check]
        optSS = result[np.argmin(result)]
        optGAM = float(stoich_to_check[np.argmin(result)])

        return optGAM, optSS

    

def ATPStoichFit(x_stoich, model, cultivation_parameters, biomass_type="Consensus", o2=True, co2=True, method="RSS", individual=False, combine=True, loopless=False):

    exp_y = cultivation_parameters.iloc[:,0].values.flatten()
    exp_y_sd = cultivation_parameters.iloc[:,1].values.flatten()
    
    pred_y = np.array([])
    
    if biomass_type in ["Consensus", "ScaledConsensus"]:
        cultivation_parameters["qStorGlyc mmol/gh"] = 0
    
    for name, values in cultivation_parameters.iterrows():
        with model as temp_model:

            if biomass_type in ["Consensus", "ScaledConsensus"]:
                 
                for rxn in ['STEROLS', 
                            'CARBOHYDRATES',
                            'LIPIDS',      
                            'PROTEINS',   
                            "PHOSPHOLIPIDS"]:
                    temp_model.reactions.get_by_id(rxn).bounds = (0,1000)

                additional_atp(temp_model, biomass_type, float(x_stoich), combine=combine)
                set_objective(temp_model, biomass_type)

            else:                    
                for rxn in [f'{biomass_type}Protein{name}',f'{biomass_type}Carbohydrate{name}',
                            f'{biomass_type}Lipid{name}', f'{biomass_type}Biomass{name}']:
                    temp_model.reactions.get_by_id(rxn).bounds = (0,1000)

                additional_atp(temp_model, f"{biomass_type}Biomass{name}", float(x_stoich), combine=combine)
                set_objective(temp_model, f"{biomass_type}Biomass{name}")


            pred_mu = constrain_predict(temp_model, values, slim=True, 
                                        o2=o2, co2=co2, loopless=loopless)
                                            

        pred_y = np.append(pred_y,pred_mu)
        
            
    if method == "RSS":
        dist = RSS(exp_y, pred_y)
    elif method == "eWRSS":
        dist = eWRSS(exp_y, pred_y, exp_y_sd)
    
    
    return dist


def calculateRhat(all_chains, plot=False):
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        rhat = az.rhat(all_chains)
    
    if plot:
        plt.figure(figsize=(15,5))
        for chain in all_chains:
            x = np.arange(0, chain.shape[0])
            plt.plot(x,chain)
        
    return rhat

def extractchains(results):
    all_samples = [x.values for x in results]
    shape = all_samples[0].shape[1]
    rxn_names = results[0].columns
    
    rxn_split = [[x[:,i] for x in all_samples] for i in np.arange(0,shape)]
    stacked_chains = [np.vstack(x) for x in rxn_split]
    grouped_chains = list(compress(stacked_chains,[~np.all(x==0) for x in stacked_chains]))
    rxns_to_evaluate = list(compress(rxn_names,[~np.all(x==0) for x in stacked_chains]))
    
    return grouped_chains, rxns_to_evaluate

def calculateDiagnostics(group, rxn):
        
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)

        if group.ndim > 1:
            ESS = az.ess(group)
            ESS_tail = az.ess(group,method="tail")

            rhat = calculateRhat(group)
            gws = []
            cESSs = []
            for chain in group:
                if np.count_nonzero(chain) < len(chain)*0.01:
                    gws.append(True)
                else:
                    gw = az.geweke(chain,0.01,0.5,intervals=100)
                    gws.append(np.any(np.abs(gw[:,1]) > 1.28))
                cESS = az.ess(group)
                cESS_tail = az.ess(chain, method="tail")
                cESSs.append(cESS_tail < 50)##(cESS < 50)^(cESS_tail < 100))

            result = pd.Series({"Rxn":rxn,
                                "Geweke Fails":sum(gws), 
                                "Rhat":rhat, 
                               # "ESS Split Fails":sum(cESSs), 
                                "ESS-Bulk":ESS,
                                "ESS-Tail":ESS_tail})
        else:
            ESS = az.ess(group)
            ESS_tail = az.ess(group,method="tail")
            if np.count_nonzero(group) < len(group)*0.01:
                gw = True
            else:
                gw = az.geweke(group,0.01,0.5,intervals=100)
                gw =  np.any(np.abs(gw[:,1]) > 1.28)


            result = pd.Series({"Rxn":rxn,
                                "Geweke Fail":gw, 
                                #"ESS Split Fail":ESS_tail<100,#(ESS<100)^(ESS_tail<100), 
                                "ESS-Bulk":ESS,
                                "ESS-Tail":ESS_tail})
        
    return result


def autocorrelation(chain, ax, title_extras):
   
    ax.plot(np.arange(0,chain.shape[0]),az.autocorr(chain))
    _ = ax.set(xlabel='Lag', ylabel='Autocorrelation', ylim=(-.1, 1))
    ax.set_title(f"Autocorrelation Plot {title_extras}")
    changes = np.sum((chain[1:]!=chain[:-1]))
    print("Acceptance Rate is: ", changes/(chain.shape[0]-1))


def plottrace(data, ax, colors):
    for i, chain in enumerate(data):
        ax.plot(np.arange(chain.shape[0]), chain,alpha=0.4, color=colors[i])

def plotrank(data,ax,colors):
    az.plot_rank(data,ax=ax,kind="vlines",vlines_kwargs={'lw':0}, marker_vlines_kwargs={'lw':3},colors=colors)

def MWUwEffects(x, y,calcES=False):
    m = x.shape[0]
    n = y.shape[0]
    
    assert m == n
    
    ranks = rankdata(np.concatenate([x,y]))
    R1 = ranks[0:m].sum()
    
    U1 = R1 - m*(m+1)/2                # method 2, step 3
    U2 = m * n - U1                    # as U1 + U2 = n1 * n2

    U, f = np.maximum(U1, U2), 2  # multiply SF by two for two-sided test

    # Tie correction according to     .. [2] Mann-Whitney U Test, Wikipedia,
#            http://en.wikipedia.org/wiki/Mann-Whitney_U_test
    
    _, t = np.unique(ranks, return_counts=True, axis=-1)
    tie_term = (t**3 - t).sum(axis=-1)

    s = np.sqrt(m*n/12 * ((m+n + 1) - tie_term/((m+n)*(m+n-1))))


    numerator = U - (m*n)/2

    # Continuity correction.
    # Because SF is always used to calculate the p-value, we can always
    # _subtract_ 0.5 for the continuity correction. This always increases the
    # p-value to account for the rest of the probability mass _at_ q = U.
    numerator -= 0.5

    # no problem evaluating the norm SF at an infinity
    with np.errstate(divide='ignore', invalid='ignore'):
        z = numerator / s

    
    mwu_p = norm.sf(z)
    mwu_p *= f

    # Ensure that test statistic is not greater than 1
    # This could happen for exact test when U = m*n/2
    mwu_p = np.clip(mwu_p, 0, 1)
    
    if calcES:

        # Compute the measure
        # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
        A = (2 * R1 - m * (m + 1)) / (2 * m * n)  # equivalent formula to avoid accuracy errors
        
         #     Cliff's delta according to https://en.wikipedia.org/wiki/Effect_size#Effect_size_for_ordinal_data
    # see also https://stats.stackexchange.com/questions/179971/cliffs-delta-or-vargha-delaney-a

        cliffd = A*2 - 1
        ## the following returns absolute values 
    #     cliffd = (U*2)/(m*n) -1

        levels = [0.11, 0.28, 0.43]  # effect sizes from Vargha and Delaney, 2000
        magnitude = ["negligible", "small", "medium", "large"]
        # scaled_A = 2*A - 1

        # magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
        magnitude = magnitude[bisect_left(levels, abs(cliffd))]


        # Effect size 1: Common Language Effect Size
        # CLES is tail-specific and calculated according to the formula given in
        # Vargha and Delaney 2000 which works with ordinal data.
        diff = x[:, None] - y
        # cles = max((diff < 0).sum(), (diff > 0).sum()) / diff.size
        # Tail = 'greater', with ties set to 0.5
        # Note that tail = 'two-sided' gives same output as tail = 'greater'
        cles = np.where(diff == 0, 0.5, diff > 0).mean()
        # cles = 1 - cles if alternative == "less" else cles

        # Effect size 2: rank biserial correlation (Wendt 1972)
        rbc = 1 - (2 * U) / diff.size  # diff.size = x.size * y.size

   

        return U, mwu_p, rbc, cles, A, magnitude, cliffd
    
    else:
        return U, mwu_p

def calcstats(x,y,rxn_name):
    U, mwu_p, rbc, cles, A, magnitude, cliffd = MWUwEffects(x, y, calcES=True)

    rhat = calculateRhat(np.vstack([x, y]))
    # Fill output DataFrame
    stats = pd.Series({"Reaction":rxn_name,
                          "U-val": U, 
                          "M P-value": mwu_p, ## Otherwise pandas rounds for visual
                          "RBC": rbc, 
                          "CLES": cles,
                          "Rhat":rhat,
                          "VD-A":A, 
                          "VD Magnitude":magnitude,
                          "Cliff's D":cliffd})
    return stats

def StoichToComp(stoich, cultivation_info, metabolite_dict, average=True):
    stoich = stoich.copy()
#     Ensure columns and index lined up
    stoich.reindex(cultivation_info.index, axis=1)
    cultivation_info = cultivation_info.copy()
    
    stoich.drop("Biomass", inplace=True)
    if average:
        # If value is negative in RSmmolg, then will generate sink reaction and provide the value in retent_info as a specific storage consumption
        cultivation_info[("qStorGlyc mmol/gh", "mean")] = np.abs(stoich[stoich < 0].loc["glycogen"] * cultivation_info[("Growth Rate model 1/h", "mean")]).fillna(0)
        # Convert the biomass consumption values to a 0
    else:
        # If value is negative in RSmmolg, then will generate sink reaction and provide the value in retent_info as a specific storage consumption
        cultivation_info["qStorGlyc mmol/gh"] = np.abs(stoich[stoich < 0].loc["glycogen"] * cultivation_info["Growth Rate model 1/h"]).fillna(0)
        # Convert the biomass consumption values to a 0
    stoich = stoich[stoich > 0].fillna(0)



    # atp_rsmmolg = RSmmolg.apply(lambda x: np.where(x < 0, 0, x))
    atp_requirements = []
    for macro in ["Protein","DNA", "RNA"]: # "Carbohydrates", 
        df = metabolite_dict[metabolite_dict.Macrocomponent == macro]
        mask = df.index[df.index.isin(stoich.index)]
        mass_dict = df["Average Mass"].to_dict()
        comp_gg = stoich.loc[mask,:].apply(lambda row: (mass_dict[row.name]/1000) * row, axis=1).sum()
        if macro == "Protein":
            ATP_requirement = stoich.loc[mask,:].sum() * 4.3
        elif macro == "RNA":
            ATP_requirement = stoich.loc[mask,:].sum() * 2.4
        elif macro == "DNA":
            ATP_requirement = stoich.loc[mask,:].sum() * 3.4
        atp_requirements.append(ATP_requirement)
        
    ATP_requirement = pd.concat(atp_requirements,axis=1).sum(axis=1)

    stoich = -stoich
    energetics = pd.DataFrame(columns=stoich.columns)
    stoich.loc["ATP"] = -ATP_requirement
    stoich.loc["water"] = -ATP_requirement
    stoich.loc["hydron"] = ATP_requirement
    stoich.loc["phosphate(3-)"] =  ATP_requirement # RSmmolg.loc["Pi"] +
    stoich.loc["ADP"] = ATP_requirement
    stoich.loc["biomass_c"] = 1
    met_name_dict = metabolite_dict["Model ID"].dropna().to_dict()
    stoich.rename(index=met_name_dict, inplace=True)
    
    return stoich, cultivation_info


def PropagateError(df, value_col, error_col, biomass_col, output_name, average=True):
                                                       
    q_rate = ((df[value_col]*1000)/1.4)/df[biomass_col]
    q_rate_error = ((df[error_col]*1000)/1.4)/df[biomass_col]
    
    mask = q_rate_error.notna()
    n= mask.sum()

    delta_systematic = np.sqrt((q_rate_error**2).sum())/n
    delta_random = q_rate.std()
    delta_net = np.sqrt(np.sum([delta_random**2,delta_random**2]))
    if average:
        index = pd.MultiIndex.from_tuples([(f"{output_name} mmol/gh", "mean"), (f"{output_name} mmol/gh", "std")])
    else:
        index = pd.MultiIndex.from_tuples([f"{output_name} mmol/gh", f"{output_name} mmol/gh"])
    return pd.Series([q_rate.mean(), delta_net], index=index)

def assume_error(df, rows, col, rel_error, limit=True):
        
        if limit:
            mask = (df.loc[:,(col, "std")]/df.loc[:,(col, "mean")]) <= rel_error/100
            mask = np.logical_and(mask, mask.index.isin(rows))
            df.loc[mask,(col, "std")] = df.loc[mask,(col, "mean")] * rel_error/100
        else:
            df.loc[:,(col, "std")] = df.loc[:,(col, "mean")] * rel_error/100


def set_biomass_objective(model,biomass_type, name):
    if biomass_type in ["Consensus", "ScaledConsensus"]:
        set_objective(model, biomass_type)

        for rxn in ['STEROLS', 
                    'CARBOHYDRATES',
                    'LIPIDS',      
                    'PROTEINS',   
                    "PHOSPHOLIPIDS", 
                    "COF", "RNA", "DNA"]:
            model.reactions.get_by_id(rxn).bounds = (0,1000)
    else:

        for rxn in ["COF", "RNA", "DNA", 
                    f'{biomass_type}Protein{name}',f'{biomass_type}Carbohydrate{name}',
                    f'{biomass_type}Lipid{name}', f'{biomass_type}Biomass{name}']:
            model.reactions.get_by_id(rxn).bounds = (0,1000)

        set_objective(model, f"{biomass_type}Biomass{name}")

def calculate_cofactors(samples_df, interested_cofactors, stoich_matrix, source=False):
    
    total_supply_df = pd.DataFrame(index=samples_df.index,columns=interested_cofactors)

    if source: 
        uniq_index = samples_df.index.droplevel(2).unique()
        repeat_index = uniq_index.repeat(len(interested_cofactors)).to_frame()
        repeat_index["Cofactor"] = np.tile(interested_cofactors,len(uniq_index))
        index = pd.MultiIndex.from_frame(repeat_index).reorder_levels([-1,0,1,2])
        supply_source_df = pd.DataFrame(index=index,columns=samples_df.columns) 

    for cof in interested_cofactors:
        rates = samples_df.values*stoich_matrix.loc[cof,samples_df.columns].values
        cols = samples_df.columns[~np.isnan(rates).any(axis=0)]

        rates = rates[:,~np.isnan(rates).any(axis=0)]
        rates = np.where(rates<0,0,rates)

        df = pd.DataFrame(data=rates, index=samples_df.index, columns=cols)

        total_supply = df.sum(axis=1)
        total_supply_df.loc[:,cof] = total_supply

        if source:
            median = df.groupby(["Equation", "Growth Rate", "Time"]).median().reset_index()
            median["Cofactor"] = cof
            median.set_index(["Cofactor", "Equation", "Time","Growth Rate"], inplace=True)
            supply_source_df.loc[median.index, median.columns] = median
    if source:
        return total_supply_df, supply_source_df
    else:
        return total_supply_df

def ax_settings(ax, var_name, x_min, x_max, label=False):
    ax.set_xlim(x_min,x_max)
    ax.set_yticks([])
    
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.spines['bottom'].set_edgecolor('#444444')
    ax.patch.set_alpha(0)
    
    if label:
        ax.text(-0.37, 0.1, var_name, fontsize=12, transform = ax.transAxes) 
    ax.tick_params(axis='x', labelsize= 15) 
    ax.xaxis.set_major_locator(MaxNLocator(4))
    return None 

from typing import Union
from operator import attrgetter

def build_reaction_string(reaction, use_metabolite_names: bool = False) -> str:
    """Generate a human readable reaction str.

    Parameters
    ----------
    use_metabolite_names: bool
        Whether to use metabolite names (when True) or metabolite ids (when False,
        default).

    Returns
    -------
    str
        A human readable str.
    """

    def format(number: Union[int, float]) -> str:
        return "" if number == 1 else str(number).rstrip(".") + " "

    id_type = "id"
    if use_metabolite_names:
        id_type = "name"
    reactant_bits = []
    product_bits = []
    for met in sorted(reaction._metabolites, key=attrgetter("id")):
        coefficient = reaction._metabolites[met]
        name = str(getattr(met, id_type))
        compartment = str(getattr(met, "compartment"))
        name += f" ({compartment})"
        if coefficient >= 0:
            product_bits.append(format(coefficient) + name)
        else:
            reactant_bits.append(format(abs(coefficient)) + name)

    reaction_string = " + ".join(reactant_bits)
    if not reaction.reversibility:
        if reaction.lower_bound < 0 and reaction.upper_bound <= 0:
            reaction_string += " <-- "
        else:
            reaction_string += " --> "
    else:
        reaction_string += " <=> "
    reaction_string += " + ".join(product_bits)
    return reaction_string


def plot_6_ridge(samples_df, pfba_df, interesting_reactions,out_name,biomass_equations, model):
    samples_df = samples_df.reset_index()
    number_sps = len(samples_df.Time.unique())


    fig = plt.figure(figsize=(16,12),dpi=300)
    outer_grid = fig.add_gridspec(2, 3, wspace=0.07, hspace=0.25)

    for i, (rxn,ub,lb) in enumerate(interesting_reactions):
        a = 0
        if i > 2:
            # gridspec inside gridspec
            inner_grid = outer_grid[1, i-3].subgridspec(nrows=number_sps,ncols=1, wspace=0, hspace=0)
        else:
            # gridspec inside gridspec
            inner_grid = outer_grid[0, i].subgridspec(nrows=number_sps,ncols=1, wspace=0, hspace=0)


        axs = inner_grid.subplots()  # Create all subplots for the inner grid.

        t_gr_dict = samples_df.set_index("Time")["Growth Rate"].to_dict()
        features = samples_df["Time"].unique()
        # features = df["Growth Rate"].unique()
        cmap = plt.cm.get_cmap('tab10', 10)
        color_list = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)][:len(biomass_equations)]

        for j in range(number_sps):

            if (i==0)^(i==3):
                ax_settings(axs[j], f'$\mu$ = {str(t_gr_dict[features[j]])} $h ^{{{-1}}}$' , ub, lb, True)  
            else:
                ax_settings(axs[j], f'$\mu$ = {str(t_gr_dict[features[j]])} $h ^{{{-1}}}$' , ub, lb, False)
            
            data = samples_df.sort_index().copy()
            data = data[data["Time"] == features[j]]
            
            min_v, max_v = np.percentile(data[rxn].values, [2.5,97.5])
            data.loc[(data[rxn] < min_v)^(data[rxn] > max_v), rxn] = np.nan        

            sns.kdeplot(data=data, x=rxn,
                    ax=axs[j], hue="Equation", hue_order=biomass_equations,
                        shade=True, common_grid=False, common_norm=False,bw_adjust=0.5, 
                        palette=color_list, legend=False)

            for k, eqn in enumerate(biomass_equations):
                axs[j].axvline(pfba_df.loc[(eqn, features[j]),rxn].values, 0,1,color=color_list[k],ls="--")

            if j < (number_sps - 1): 
                axs[j].set_xticks([])
                axs[j].set_xlabel("")

        # string = model.reactions.get_by_id(rxn).build_reaction_string(True)
        string = build_reaction_string(model.reactions.get_by_id(rxn),True)
        if string.find(">") > -1:
            string = string.split(">")
            string = (">\n").join(string)
        elif string.rfind("-") > -1:
            string = string.rsplit("-",1)
            string = ("-\n").join(string)


        axs[-1].set_xlabel("") 
        axs[0].text(0.5, 2.2, rxn, fontsize=15, transform = axs[0].transAxes,horizontalalignment='center',verticalalignment='bottom') 
        axs[0].text(0.5, 2, string, fontsize=9, transform = axs[0].transAxes,horizontalalignment='center',verticalalignment='top')  

    labels = [eqn if not eqn == "ScaledConsensus" else "Consensus - Scaled" for eqn in biomass_equations]
              
    legend_elements = [mpatches.Patch(edgecolor=cmap(k), label=eqn, 
                                      facecolor=cmap(k)[:-1] + (0.3,)) for k, eqn in enumerate(labels)]
    

    legend = fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.9,0.46), fontsize=12, frameon=False)

    fig.text(0.5, 0.06, r'Normalized Flux ($\frac{q_{Reaction}}{q_{Glucose}}$)', ha='center', va='center', fontsize=18)


    fig.savefig(f"../results/plots/Ridge_6x6_{out_name}.png",
            bbox_inches='tight',transparent=True)


def enrich(input_rxn_list, background_rxn_list, subsystem_df):
    data = subsystem_df[subsystem_df["Rxn"].isin(input_rxn_list)]
    background = subsystem_df[subsystem_df["Rxn"].isin(background_rxn_list)]

    output = pd.DataFrame(columns=["Subsystem annotation",
                                   "No. Reactions in input w/ annotation",
                                   "No. Reactions in input",
                                   "No. All reactions w/ annotation",
                                   "No. All reactions",
                                   "p-value"])
    for subsystem in data.Subsystem.unique():
        k = len(data) #input list (list of reactions with large effect size
        x = len(data[data.Subsystem == subsystem]) #num of reactions in the input list with the annotation
        N = len(set(background_rxn_list)) # number of all reactions 
        m = len(background[background.Subsystem == subsystem])
        if m > 5:
            p = hypergeom.sf(x, N, k, m)
        else:
            p = np.nan
        output = pd.concat([output, pd.Series({"Subsystem annotation":subsystem,
                                               "No. Reactions in input w/ annotation":x,
                                               "No. Reactions in input":k,
                                               "No. All reactions w/ annotation":m,
                                               "No. All reactions":N,
                                               "p-value":p}).to_frame().T], 
                           ignore_index=True)
    reject, adj_pvalues, corrected_a_sidak, corrected_a_bonf =  mc.multipletests(output["p-value"], method='fdr_bh')
    output["adj_pval"] = adj_pvalues
    return output

