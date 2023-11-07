import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.special import erf
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as Poly
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def round_to_half(values):
    """Round an array of numbers to the nearest half."""
    return np.around(values * 2) / 2

def gompertz_function(x,a,b,c):   
    return np.array([a * np.exp(-np.exp(b-c*x))]).flatten()

def calculate_production(production_func, time, inlet_flow_rate_a, conc_protein, volume_vessel_b):
    """Calculate the production rate of the protein."""
    time = time.reshape(-1, 1)
    derivative_conc_protein = (production_func.deriv()(time) / 1000).reshape(-1, 1)
    washout = conc_protein * (inlet_flow_rate_a / volume_vessel_b)
    production = derivative_conc_protein + washout
    return production

def calculate_ss(observed, estimated, weighting):
    """Calculate the weighted sum of squares (SS) and coefficient of determination (R^2)."""
    observed = observed.flatten()
    estimated = estimated.flatten()
    weighting = weighting.flatten()

    ss_residuals = np.nansum((weighting * (estimated - observed)) ** 2)
    ss_total = np.nansum((weighting * (observed - np.mean(observed))) ** 2)

    r_squared = 1 - (ss_residuals / ss_total)

    return ss_residuals, r_squared

def dcdt(time, conc, inlet_flow_rate_a, volume_vessel_b, volume_vessel_a, maintenance_substrate, max_yield_x_substrate, max_yield_p_substrate, production_func, death_func=False, death_rate=False, gompertz_parameters=False):
    """
    Define the differential equations for the fermentation process.

    Args:
        time (float): Current time point.
        conc (array): Concentrations of biomass, substrate, product, dead biomass, and total biomass.
        inlet_flow_rate_a (float): Inlet flow rate for vessel A.
        volume_vessel_b (float): Volume of vessel B.
        volume_vessel_a (float): Volume of vessel A.
        maintenance_substrate (float): Maintenance coefficient for substrate.
        max_yield_x_substrate (float): Maximum yield of biomass on substrate.
        max_yield_p_substrate (float): Maximum yield of product on substrate.
        production_func (callable): Production rate function.
        death_func (callable, optional): Death rate function.
        death_rate (float, optional): Death rate of biomass.
        gompertz_parameters (array, optional): Parameters for the Gompertz function.

    Returns:
        dc (array): The derivatives of concentrations.
    """
    # Unpack concentrations
    conc_biomass_vessel_b, conc_substrate_vessel_a, conc_protein_vessel_b, conc_dead_biomass_vessel_b, total_dry_cell_weight = conc
    conc_substrate_vessel_b = 0  # Initialize substrate concentration in vessel B

    # Model Parameters
    # Vessel A - Mixing Vessel
    conc_substrate_refill = 3.5  # Refill concentration of the substrate

    # Vessel B - Fermentor
    outlet_flow_rate_vessel_b = inlet_flow_rate_vessel_b = outlet_flow_rate_vessel_a = inlet_flow_rate_a
    specific_substrate_consumption_rate = ((inlet_flow_rate_vessel_b / volume_vessel_b) * (conc_substrate_vessel_a - conc_substrate_vessel_b)) / conc_biomass_vessel_b

    
    # Calculate specific growth rate (mu) and specific production rate (qp)
    if isinstance(gompertz_parameters, np.ndarray):
        # If Gompertz parameters are provided, use the Gompertz function to calculate production rate
        gompertz_function = production_func
        specific_production_rate_residual = gompertz_function(0, *gompertz_parameters)
        specific_growth_rate = (specific_substrate_consumption_rate - specific_production_rate_residual / max_yield_p_substrate - maintenance_substrate) * max_yield_x_substrate
        specific_production_rate = gompertz_function(specific_growth_rate, *gompertz_parameters)
    else:
        # If no Gompertz parameters, calculate production rate from derivative of production function
        derivative_conc_protein = production_func.deriv()(np.array([[time]])) / 1000
        outflow = conc_protein_vessel_b * (inlet_flow_rate_a / volume_vessel_b)
        production = derivative_conc_protein + outflow
        specific_production_rate = production / conc_biomass_vessel_b
        specific_growth_rate = (specific_substrate_consumption_rate - specific_production_rate / max_yield_p_substrate - maintenance_substrate) * max_yield_x_substrate

    # Calculate death rate (kd)
    if death_func:
        derivative_viability = death_func.deriv()(time)
        viability = conc_biomass_vessel_b / total_dry_cell_weight
        death_rate = specific_growth_rate * (1 - viability) - derivative_viability / viability
        death_rate = max(death_rate, 0)  # Ensure non-negative death rate
    else:
        death_rate = death_rate

    # System of differential equations
    d_conc_biomass_vessel_b = (specific_growth_rate * conc_biomass_vessel_b - death_rate * conc_biomass_vessel_b).flatten()
    d_conc_substrate_vessel_a = ((inlet_flow_rate_a / volume_vessel_a) * conc_substrate_refill - (outlet_flow_rate_vessel_a / volume_vessel_a) * conc_substrate_vessel_a).flatten()
    d_conc_protein_vessel_b = (specific_production_rate * conc_biomass_vessel_b - (outlet_flow_rate_vessel_b / volume_vessel_b) * conc_protein_vessel_b).flatten()
    d_conc_dead_biomass_vessel_b = (death_rate * conc_biomass_vessel_b).flatten()
    d_total_dry_cell_weight = (specific_growth_rate * conc_biomass_vessel_b).flatten()

    # Combine derivatives into a single array
    dc = np.concatenate([d_conc_biomass_vessel_b, d_conc_substrate_vessel_a, d_conc_protein_vessel_b, d_conc_dead_biomass_vessel_b, d_total_dry_cell_weight])
    return dc

# Note: The `funobj` and `retentostat_regression` functions are lengthy and are recommended to be split into smaller functions for better readability and maintainability.

# The `plot_regression_model` function is responsible for plotting the regression model results and saving the figures. It is recommended to separate the plotting and saving functionalities for reusability and clarity.

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def funobj(params, inlet_flow_rate_a, time_lab_h, substrate_conc_lab, viable_cell_conc, dry_weight, dry_weight_std, cell_conc_std_v, volume_vessel_b, volume_vessel_a, protein_conc, max_yield_x_substrate, max_yield_p_substrate, production_func, death_func, gompertz_parameters=False, optimize=True):
    """
    Objective function for the optimization of fermentation process parameters.
    
    Args:
        params (list): Parameters to optimize; maintenance coefficient [ms] and death rate [kd].
        inlet_flow_rate_a (float): Inlet flow rate for vessel A.
        time_lab_h (np.array): Lab measurement time points in hours.
        substrate_conc_lab (np.array): Lab measurements of substrate concentration.
        viable_cell_conc (np.array): Lab measurements of viable cell concentration.
        dry_weight (np.array): Lab measurements of dry cell weight.
        dry_weight_std (np.array): Standard deviation of dry cell weight measurements.
        cell_conc_std_v (np.array): Standard deviation of viable cell concentration measurements.
        volume_vessel_b (float): Volume of vessel B.
        volume_vessel_a (float): Volume of vessel A.
        protein_conc (np.array): Lab measurements of protein concentration.
        max_yield_x_substrate (float): Maximum yield coefficient for biomass on substrate.
        max_yield_p_substrate (float): Maximum yield coefficient for product on substrate.
        production_func (callable): Function representing the production rate.
        death_func (callable, optional): Function representing the death rate.
        gompertz_parameters (np.ndarray, optional): Parameters for the Gompertz function if applicable.
        optimize (bool): Flag indicating whether to return just the sum of squared errors for optimization or all simulation results.

    Returns:
        If optimize is True, returns a float representing the sum of squared errors for the optimization.
        If optimize is False, returns a tuple with simulation results and calculated metrics.
    """
    
    # Set the outflow of vessel A and inflow of vessel B equal to the inflow of vessel A
    inflow_vessel_b = outflow_vessel_a = inlet_flow_rate_a
    
    dilution_rate = (inflow_vessel_b/volume_vessel_b)[0]

    # Define the time span for the ODE solver in hours
    days_of_operation = 28  # Duration for which the retentostat will run
    time_start = 0
    time_end = 24 * days_of_operation
    time_step = 0.5  # Time steps for the ODE solver
    time_span = np.arange(time_start, time_end, time_step) #.reshape(-1, 1)

    # Initialize concentrations based on steady state from the chemostat phase
    substrate_conc_initial = substrate_conc_lab[0]  # Substrate concentration in inflow (g/L)
    protein_conc_initial = protein_conc[0] / 1000 if isinstance(gompertz_parameters, np.ndarray) else production_func(np.array([0])) / 1000  # Protein concentration converted to g/L
    
    # Extract the parameters for maintenance coefficient and optionally for death rate
    maintenance_coefficient = params[0]
    death_rate = False if death_func else params[1]

    # Calculate initial biomass concentrations based on whether a death function is provided
    if death_func:
        viable_biomass_initial = dry_weight[0] * death_func(0)
        dead_biomass_initial = dry_weight[0] * (1 - death_func(0))
    else:
        viable_biomass_initial = viable_cell_conc[0]
        dead_biomass_initial = dry_weight[0] - viable_cell_conc[0]


    # Combine initial conditions into a single array
    initial_conditions = np.concatenate([viable_biomass_initial, substrate_conc_initial, protein_conc_initial, dead_biomass_initial, dry_weight[0]]).flatten()
   
    # Determine if dense output is needed based on optimization flag
    dense_output = not optimize


    # Solve the system of ODEs using the RK45 method
    solution = solve_ivp(
        dcdt, 
        [time_start, time_end], 
        initial_conditions, 
        args=(inlet_flow_rate_a, volume_vessel_b, volume_vessel_a, maintenance_coefficient, max_yield_x_substrate, max_yield_p_substrate, production_func, death_func, death_rate, gompertz_parameters), 
        method='RK45', 
        t_eval=time_span, 
        dense_output=dense_output
    )

    # Handle unsuccessful ODE solution
    if solution.status != 0:
        return np.inf  # Use infinity to indicate an unsolvable ODE system
    

    # Unpack solution
    biomass_viable, substrate_conc_vessel_a, protein_conc, biomass_dead, biomass_total = solution.y
    
    
    # Calculate the sum of squares errors and R-squared values for the biomass and dry weight
    t_indices = np.isin(time_span, time_lab_h)
    SSE_viable_biomass, R2_viable_biomass = calculate_ss(viable_cell_conc, biomass_viable[t_indices], 1 / cell_conc_std_v)
    SSE_dead_biomass, R2_dead_biomass = calculate_ss(dry_weight - viable_cell_conc, biomass_dead[t_indices], 1 / dry_weight_std)
    SSE_total_biomass, R2_total_biomass = calculate_ss(dry_weight, biomass_total[t_indices], 1 / dry_weight_std)
    SSE_total = SSE_viable_biomass + SSE_total_biomass
    
    if optimize:
        return SSE_total # Return the sum of squared errors if optimizing
    
    else:
        substrate_concentration_vessel_b = 0
        specific_substrate_uptake = (dilution_rate*(substrate_conc_vessel_a-substrate_concentration_vessel_b))/biomass_viable

        # Calculate specific growth rate (mu) and specific production rate (qp) from the solution
        # The specific growth rate is calculated from the change in total biomass concentration
        # The specific production rate is calculated from the change in protein concentration
        delta_t = np.diff(time_span, axis=0)
        delta_t = np.append(delta_t, delta_t[[-1]], axis=0)

        delta_protein_conc = np.diff(protein_conc, axis=0)
        delta_protein_conc = np.append(delta_protein_conc, delta_protein_conc[[-1]], axis=0)

        specific_production_rate = (delta_protein_conc / delta_t + dilution_rate * protein_conc) / biomass_viable

        delta_biomass_total = np.diff(biomass_total, axis=0)
        delta_biomass_total = np.append(delta_biomass_total, delta_biomass_total[[-1]], axis=0)

        specific_growth_rate = delta_biomass_total / delta_t / biomass_total
        pirt_growth_rate = (specific_substrate_uptake - specific_production_rate/max_yield_p_substrate - maintenance_coefficient)*max_yield_x_substrate

        delta_biomass_dead = np.diff(biomass_dead, axis=0)
        delta_biomass_dead = np.append(delta_biomass_dead, delta_biomass_dead[[-1]], axis=0)

        death_rate = (delta_biomass_dead/delta_t) * 1000
        
        # Otherwise, return all calculated values
        
        return time_span, biomass_viable, substrate_conc_vessel_a, protein_conc, biomass_dead, biomass_total, specific_substrate_uptake, specific_growth_rate, pirt_growth_rate, specific_production_rate, death_rate, SSE_total, R2_viable_biomass, R2_dead_biomass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def retentostat_regression(time_csin, substrate_conc_lab, inlet_flow_rate_a, volume_vessel_a, volume_vessel_b, viable_cell_conc_lab, time_lab_h, viable_cell_conc_std_v, dry_weight, dry_weight_std, protein_conc, max_yield_x_substrate, max_yield_p_substrate, production_func, death_func=False, gompertz_parameters=False):
    """
    Perform regression analysis on retentostat fermentation data to optimize parameters.
    
    Args:
        time_csin (np.array): Time points for substrate concentration input.
        substrate_conc_lab (np.array): Lab measurements of substrate concentration.
        inlet_flow_rate_a (float): Inlet flow rate for vessel A.
        volume_vessel_a (float): Volume of vessel A.
        volume_vessel_b (float): Volume of vessel B.
        viable_cell_conc_lab (np.array): Lab measurements of viable cell concentration.
        time_lab_h (np.array): Lab measurement time points in hours.
        viable_cell_conc_std_v (np.array): Standard deviation of viable cell concentration measurements.
        dry_weight (np.array): Lab measurements of dry cell weight.
        dry_weight_std (np.array): Standard deviation of dry weight measurements.
        protein_conc (np.array): Lab measurements of protein concentration.
        max_yield_x_substrate (float): Maximum yield of biomass on substrate.
        max_yield_p_substrate (float): Maximum yield of product on substrate.
        production_func (callable): Function modeling the production rate.
        death_func (callable, optional): Function modeling the death rate.
        gompertz_parameters (np.ndarray, optional): Gompertz model parameters.

    Returns:
        tuple: A DataFrame with regression results, a Series with statistical results, and a solution object from the ODE solver.
    """

    # Set the outflow of vessel A to match the inflow
    outflow_vessel_a = inlet_flow_rate_a
    inflow_vessel_b = outflow_vessel_a

    # Define initial guesses for optimization parameters
    initial_guess = [0.001] if death_func else [0.001, 0.0003]
    parameter_bounds = [(0, 0.01)] if death_func else [(0, 0.01), (0, 0.01)]

    # Optimization tolerance
    tolerance = 1e-8

    # Perform optimization using the L-BFGS-B algorithm
    optimization_result = minimize(
        fun=funobj,
        x0=initial_guess,
        method="L-BFGS-B",
        args=(inlet_flow_rate_a, time_lab_h, substrate_conc_lab, viable_cell_conc_lab, dry_weight, dry_weight_std, viable_cell_conc_std_v, volume_vessel_b, volume_vessel_a, protein_conc, max_yield_x_substrate, max_yield_p_substrate, production_func, death_func, gompertz_parameters, True),
        bounds=parameter_bounds,
        options={'ftol': tolerance, 'gtol': tolerance, 'maxls': 100, 'disp': False, 'maxiter': 10000, 'maxfun': 10000}
    )

    # Extract optimized parameters
    optimized_params = optimization_result.x
    # Calculate parameter uncertainty using the inverse Hessian matrix
    uncertainty = np.sqrt(np.diag(optimization_result.hess_inv.todense()) * max(1, abs(optimization_result.fun)) * tolerance)

    # Print the optimized parameters with their uncertainties
    for i, (param, error) in enumerate(zip(optimized_params, uncertainty)):
        print(f'Parameter {i}: {param:.4e} ± {error:.1e}')

    # Run the model with optimized parameters to get detailed results
    detailed_results = funobj(
        optimized_params,
        inlet_flow_rate_a, time_lab_h, substrate_conc_lab, viable_cell_conc_lab,
        dry_weight, dry_weight_std, viable_cell_conc_std_v, volume_vessel_b,
        volume_vessel_a, protein_conc, max_yield_x_substrate, max_yield_p_substrate, production_func,
        death_func, gompertz_parameters, optimize=False
    )

    # Parse detailed results
    time_span, biomass_viable, substrate_conc, protein_conc, biomass_dead, biomass_total, specific_substrate_uptake, specific_growth_rate, pirt_growth_rate, specific_production_rate, death_rate, SSE_total, R2_viable_biomass, R2_dead_biomass = detailed_results
    # t, Cx, Cs, Cp, Cxd, Cdw, qs, mu, hp_mu, qp, Yps, qs_mu, qs_ms, SSE_total, R2, R2_d, sol 

    # Calculate additional metrics
    maintenance_coefficient_opt = optimized_params[0] * 1000  # Convert to mg/gx/h
    maintenance_coefficient_sd = uncertainty[0] * 1000  # Convert uncertainty to mg/gx/h
    k_death_opt, k_death_sd = (None, None)
    if not death_func:
        k_death_opt = optimized_params[1] * 1000  # Convert to 1^-3 h^-1 if death rate was optimized
        k_death_sd = uncertainty[1] * 1000  # Convert uncertainty to 1^-3  h^-1

    # Calculate accumulated product concentration and doubling time
    doubling_time = np.log(2) / specific_growth_rate[-2]

    regression_results = pd.DataFrame(
        data=np.stack([time_span, time_span / 24, specific_growth_rate, pirt_growth_rate, substrate_conc, biomass_viable, biomass_total, protein_conc * 1000, specific_substrate_uptake, specific_production_rate * 1000, death_rate, (biomass_viable / biomass_total * 100)], axis=1),
        columns=[
            'Time h', 'Time d', 'Growth Rate model 1/h', 'Growth Rate Pirt 1/h', 'Cs g/L',
            'Biomass viable model g/L', 'Biomass model g/L', 'Cp mg/L', 'qS model g/gh', 'qP model mg/gh',
            'Death rate (10^-3 h^-1)', 'Viability model %'
        ]
    )

    # Prepare statistical results Series
    stats_series = pd.Series(
        [maintenance_coefficient_opt, maintenance_coefficient_sd, k_death_opt, k_death_sd, SSE_total, R2_viable_biomass, R2_dead_biomass, specific_growth_rate[-2], doubling_time / 24, specific_production_rate[-2] * 1000],
        index=[
            'mS opt (mg/g.h)', 'mS SD (mg/g.h)', 'kd opt (10^-3 1/h)', 'kd opt SD (10^-3 1/h)', 'SSE',
            'R2', 'R2_d', 'Mu end (h^-1)', 'Doubling time (d)', 'qP at 0 (mg/gh)'
        ]
    )

    return regression_results, stats_series
