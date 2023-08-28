from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.special import erf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P


def GenerateDeathFunction(time, viab, deg):

    p = P.fit(time.flatten(), viab.flatten(), deg)
    return p

def CalculateProduction(prod_func, time, Fin_A, Cp, V_B):
    time = time.reshape(-1,1)
    dCp = (prod_func.deriv()(time)/1000).reshape(-1,1)
    washout = Cp*(Fin_A/V_B)
    production = dCp + washout
    return production


def ProteinProductionFunc(time, measured_Cp, deg):
    y_train = measured_Cp.flatten()
    X_train = time.flatten()
    p = P.fit(X_train, y_train, deg)
    return p



def dcdt(t,c,Fin_A, V_B, V_A,  ms, Yxs_max, Yps_max, prod_func, death_func):
    

    Cx_B, Cs_A, Cp_B, Cxd_B, DCW = c
    Cs_B = 0
    
    # %% Model Parameters

    # % Vessel A - Mixing Vessel
    # the concentration in the second medium vessel; eventually the concentration in vessel A.
    Cs_R = 3.5

    # % Vessel B - Fermentor
    Fout_B = Fin_B = Fout_A = Fin_A
    qs = ((Fin_B/V_B)*(Cs_A-Cs_B))/Cx_B

    
    # trained on mg/L so convert to g/L
    dCp = prod_func.deriv()(np.array([[t]]))/1000
    outflow = Cp_B*(Fin_A/V_B)
    production = dCp + outflow
    qp = production/Cx_B
    

    mu = (qs - qp/Yps_max - ms) * Yxs_max

    dViab = death_func.deriv()(t)


    Viab = Cx_B/DCW
    kd = mu*(1-Viab) - dViab/Viab
    
    if kd < 0:
        kd = 0
    
   
    # equations 
    dc1 = (mu*Cx_B - kd*Cx_B).flatten() # balance of Cx_B
    dc2 = ((Fin_A/V_A)*Cs_R - (Fout_A/V_A)*Cs_A).flatten() # Balance of Cs_A
    dc3 = (qp*Cx_B - (Fout_B/V_B) * Cp_B).flatten()  # Balance of Cp_B: Product formation
    dc4 = (kd * Cx_B).flatten() # balance of dead biomass
    dc5 = (mu*Cx_B).flatten()
   
    dc = np.concatenate([dc1, dc2, dc3, dc4, dc5])
 
    return dc


def funobj(p, Fin_A, TIME_LAB_V,  CS_in_LAB, CX_VIAB, DW, DW_std,CX_LAB_std_V, V_B, V_A, Yps_max, prod_func, death_func, optimize=True):
    
           
           
    Fin_B = Fout_A = Fin_A # % L/h
 
    # %% Time span of the ode-solver in h

    days = 28 # days the retentostat will run
    tinit = 0
    tend = 24*days
    dt = 0.5 # the time steps that are used for the ode solver.
    
    # tspan must be monotonic for the ODE solver. Append the range, the sample times and then sort and only unique
    tspan = np.arange(tinit,tend, dt).reshape(-1,1)
    tspan = np.concatenate([tspan, (TIME_LAB_V*24).reshape(-1,1)])
    tspan = np.unique(tspan)

    # %% Initial concentrations for the ode-solver
    # % In this model, these are the values that are obtained from the
    # % chemostat phase, where we know there is a steady state. The time of 0 hours is when the chemostat is
    # % switched to a retentostat.

    C_S0 = CS_in_LAB[0] #  g/L Substrate in the inflow of vessel A
    C_P0 = prod_func(0)/1000
    
    ms = p[0] # maintenance coefficient % gs/gx/h
    # Yxs_max = p[1]
    Yxs_max = 0.54
    

    C_X0 = DW[0] * death_func(0) #CX_VIAB[0]
    C_Xd0 = DW[0] * (1-death_func(0)) # DW[0] - CX_VIAB[0]

    c0 = [C_X0, C_S0, np.array([C_P0]), C_Xd0, DW[0]]
    c0 = [x.flatten() for x in c0]
    c0 = np.concatenate(c0)

    Cs_R = CS_in_LAB[-1]
    
    # ODE solver
    # RK45
    
    sol = solve_ivp(dcdt, [tinit, tend], c0, method='RK45', t_eval=tspan.flatten(),args=(Fin_A, V_B, V_A,  ms, Yxs_max, Yps_max, prod_func, death_func))
    if sol.status != 0:
        return 1e6
    c = sol.y
    
    Cx = c[0,:].reshape(-1,1)
    Cs = c[1,:].reshape(-1,1)
    Cp = c[2,:].reshape(-1,1)
    Cxd = c[3,:].reshape(-1,1)
    Cdw = c[4,:].reshape(-1,1) # Cx + Cxd
 
    
    # % Calculate mu
    # % mu is calculated from the Herbert-pirt equation.
    # % Based on this mu, the qs, qp and Yps values are calculated       

    Cs_B = 0
    
    qs = (Fin_B/V_B*(Cs-Cs_B))/Cx

    production = CalculateProduction(prod_func, tspan, Fin_A, Cp, V_B).reshape(-1,1)
    qp = production/Cx

    mu = (qs - qp/Yps_max - ms) * Yxs_max
    
    Yps = qp/qs
    qs_mu = (mu/Yxs_max)/qs
    qs_ms = ms/qs
    
    
    # % Select t values where we have measurements and then calculate sum of squares
    t_mask = np.isin(tspan/24, TIME_LAB_V)
    

    SSE_X = (((Cx[t_mask,:] - CX_VIAB) / CX_LAB_std_V)**2).sum(axis=0)
    SSE_tot = (((Cx[t_mask,:] - np.mean(CX_VIAB,axis=1)) / CX_LAB_std_V)**2).sum(axis=0)
    # SSE_tot = (((CX_VIAB - np.mean(CX_VIAB,axis=1)) / CX_LAB_std_V)**2).sum(axis=0)

    SSE_Xd = (((Cxd[t_mask,:] - (DW -CX_VIAB)) / DW_std)**2).sum(axis=0)
    SSE_totd = (((Cxd[t_mask,:] - np.mean(DW-CX_VIAB)) / DW_std)**2).sum(axis=0)
    # SSE_totd = ((((DW-CX_VIAB) - np.mean(DW-CX_VIAB)) / DW_std)**2).sum(axis=0)
    
    SSE_DCW = (((Cdw[t_mask,:] - DW) / DW_std)**2).sum(axis=0)
    SSE_totdcw = (((Cdw[t_mask,:] - np.mean(DW)) / DW_std)**2).sum(axis=0)

    Rsq = 1-(SSE_X/SSE_tot)[0]
    Rsq_d = 1-(SSE_Xd/SSE_totd)[0]

    # % This value should be optimized
  
    SSE_total = (SSE_X + SSE_DCW)[0]


    if optimize:
        return SSE_total
    else: 
        return tspan, Cx, Cs, Cp, Cxd, Cdw, qs, mu, qp, Yps, qs_mu, qs_ms, SSE_total, Rsq, Rsq_d
    


def retentostat_regression(TIME_CSin, CS_in_LAB,Fin_A,V_A, V_B,CX_VIAB, TIME_LAB_V, CX_LAB_std_V, DW, DW_std, Yps_max, prod_func, death_func):
  
    # %% Process Parameters
    # % Vessel A - Mixing Vessel
    Fout_A = Fin_A # L/h
    
    # % Vessel B - Fermenter
    FinB = Fout_A
    
    # %% Optimisation function
    # initial guesses
    x0 = np.array([0.001])#

    ftol = 1e-8

    bounds = [(0, 0.01)]

    res = minimize(fun=funobj, x0=x0, 
                   options={'ftol':1e-8, 'gtol':1e-8,'maxls':100,'disp': False, 
                            'maxiter':10000, 'maxfun':10000}, method="L-BFGS-B",
                   args=(Fin_A, TIME_LAB_V,  CS_in_LAB, CX_VIAB, DW, DW_std,CX_LAB_std_V, V_B, V_A, Yps_max, prod_func, death_func, True),
                  bounds=bounds)
    xopt = res.x
    
    # Parameter uncertainty form inverse hessian. According to this SO: https://stackoverflow.com/questions/43593592
    tmp_i = np.zeros(len(res.x))
    uncertainty = []
    for i in range(len(res.x)):
        tmp_i[i] = 1.0
        hess_inv_i = res.hess_inv(tmp_i)[i]
        uncertainty_i = np.sqrt(max(1, abs(res.fun)) * ftol * hess_inv_i)
        uncertainty.append(uncertainty_i)
        tmp_i[i] = 0.0
        print('x^{0} = {1:12.4e} ± {2:.1e}'.format(i, res.x[i], uncertainty_i))
    
    # Function defined such that it returns a scalar at first, but can then switch to a 'optimise=False' method to export the desired info with xopt
    xopt = res.x
    tspan, Cx, Cs, Cp, Cxd, Cdw, qs, mu, qp, Yps, qs_mu, qs_ms, SSE_total, Rsq, Rsq_d = funobj(xopt, Fin_A, TIME_LAB_V,  CS_in_LAB, CX_VIAB, DW, DW_std,
                                                                                               CX_LAB_std_V, V_B, V_A, Yps_max, prod_func, death_func, optimize=False)
                                                                                               

    ms_opt = xopt[0]*1000 # % mg/gx/h
   
    ms_sd = uncertainty[0] *1000 # % mg/gx/h


    mu_end = mu[-1]
    t_dub = np.log(2)/mu_end
    
    AccCp = Cp[0]
    for i in range(Cp.shape[0] - 1):
        values = AccCp[-1] + (Cx* qp * 0.5)[i]
        AccCp = np.append(AccCp, values, axis=0)
    AccCp = AccCp.reshape(-1,1)
    Cp = Cp*1000
    qp = qp*1000
    AccCp = AccCp*1000
    qp_0 = qp[-1]
    

    viability = 100*(Cx/Cdw)
    difference = np.append(np.diff(Cxd,axis=0),np.diff(Cxd,axis=0)[[-1]],axis=0)
 
    
    death_rate = 1000*(difference/Cx)
    
    data = np.concatenate([x.reshape(-1,1) for x in [tspan, tspan/24, mu, Cs, Cx,Cdw, Cp, qs, qp, AccCp, death_rate, viability]], axis=1)

    regression_results = pd.DataFrame(data=data, columns=['Time h', 'Time d', 'Growth Rate model 1/h', 'Cs g/L', 'Biomass viable model g/L','Biomass model g/L', 
                                                          'Cp mg/L', 'qS model g/gh', "qP model mg/gh", "Total titre mg/L", "Death rate (10^-3 h^-1)", "Viability model %"])
   
    
    stats_series = pd.Series([ms_opt, ms_sd, SSE_total, Rsq, Rsq_d, mu_end[0], t_dub[0]/24, qp_0[0], ],
                             index=['mS opt (mg/g.h)', 'mS SD (mg/g.h)', 'SSE', 'Rsq', "Rsq_d",  
                                    'Mu end (h^-1)', 'Doubling time (d)', "qP at 0 (mg/gh)"])


    return regression_results, stats_series
     
    
    
    

def plot_regression_model(results, og_time, og_cs_in, stats, og_cx, og_viab_cx, productivity, name):
    
    fig = plt.figure(figsize=(16, 6), dpi=400)
    spec = fig.add_gridspec(1, 1, wspace=0.8, bottom=0.1)
    
    gssub = spec[0].subgridspec(1, 3)
    ax1 = fig.add_subplot(gssub[0, 0])

    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(gssub[0, 1])

    ax4 = ax3.twinx()
    ax5 = fig.add_subplot(gssub[0, 2])

    ax6 = ax5.twinx()
    
    fig.text(0.5, -0.04, 'Time [d]', ha='center', fontsize=15)
    
    
    ax1.plot(results["Time d"], results["Cs g/L"], c="#e41a1c", lw=3)
    ax1.scatter(og_time.flatten(),og_cs_in.flatten(), facecolors="None", edgecolors="#e41a1c")
    ax1.set_ylabel('$Glucose,\ Cs_{in}\ [g\ L^{-1}]$', fontsize=15,labelpad=3, c="#e41a1c")
    ax1.set_ylim(0, 11)

    ax2.plot(results["Time d"], results["qS model g/gh"], c="#377eb8" ,lw=3)
    ax2.hlines(stats[0]/1000, 0, 30, colors="black", linestyles="--")
    ax2.text(0.5, stats[0]/1000, '$m_S$', 
        horizontalalignment='left',
        verticalalignment='bottom', 
        fontsize=15)
    ax2.set_ylabel('$qS\ [g\ g^{-1} h^{-1}]$', fontsize=15,labelpad=3, c="#377eb8" )
    
    ax2.set_ylim(0, 0.05)

    ax3.plot(results["Time d"], results["Growth Rate model 1/h"], c="#de981f", linestyle="dashed", lw=3)
    ax3.set_ylabel('$\mu\ [h^{-1}]$', fontsize=15,labelpad=3, c="#de981f") #c="#4daf4a")
    ax3.set_ylim(0, 0.03)
    
    ax4.plot(results["Time d"], regression_results["Biomass model g/L"], c="#e41a1c", lw=1.5, label="Total")
    ax4.scatter(og_time.flatten(),og_cx.flatten(),  facecolors="None", edgecolors="#e41a1c")
    ax4.set_ylabel('$Biomass,\ Cx\ [g\ L^{-1}]$', fontsize=15,labelpad=3)#, c="#e41a1c")

    ax4.plot(results["Time d"], results["Biomass viable model g/L"], c="#377eb8" , lw=3, label="Viable")
    ax4.scatter(og_tim.flatten(),og_viab_cx.flatten(), facecolors="None", edgecolors="#377eb8")
    ax4.legend(loc=2,bbox_to_anchor=(-0.02,1.01), frameon=False)
    ax4.set_ylim(0, 22)

    ax5.plot(results["Time d"], results["qP model mg/gh"], c="#377eb8" , lw=3)
    ax5.set_ylabel('$qP\ [mg\ g^{-1} h^{-1}]$', fontsize=15,labelpad=3, c="#377eb8" )
    ax5.set_ylim(0, 0.4)
    
    ax6.plot(results["Time d"], results["Cp mg/L"], c="#e41a1c", lw=3)
    ax6.scatter(productivity["time (h)"].values/24, productivity["CP measured (mg/L)"].values,  facecolors="None", edgecolors="#e41a1c")
    ax6.set_ylabel('$C_{VHH}\ [mg\ L^{-1}]$', fontsize=15,labelpad=3, c="#e41a1c")
    ax6.set_ylim(0, 50)
    

    [x.set_xlim(0, 28) for x in [ax1,ax2,ax3,ax4,ax5, ax6]]
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=None)
       
    fig.savefig(f"../results/plots/{name}PresentationDynamicRetentostatFitting.png", 
            bbox_inches="tight",  transparent=True)
    plt.close()
    
    fig = plt.figure(figsize=(8, 6), constrained_layout=True, dpi=400)


    ax = fig.add_subplot()
    axt = ax.twinx()
    
    ax.plot(results["Time d"], results["Growth Rate model 1/h"], c="#de981f", linestyle="dashed", lw=3)
    ax.set_ylabel('$\mu\ [h^{-1}]$', fontsize=15,labelpad=3, c="#de981f")
    ax.set_ylim(0, 0.03)
    ax.set_xlim(0, 28)
    
    axt.plot(results["Time d"], results["Biomass model g/L"], c="#e41a1c", lw=1.5, alpha=0.9, label="Total")
    axt.scatter(og_time.flatten(),og_cx.flatten(),  facecolors="None", edgecolors="#e41a1c", alpha=0.9)
    

    axt.plot(results["Time d"], results["Biomass viable model g/L"], c="#377eb8" , lw=3, label="Viable")
    axt.scatter(og_time.flatten(),og_viab_cx.flatten(), facecolors="None", edgecolors="#377eb8")
    axt.legend(loc=2,bbox_to_anchor=(-0.02,1.01), frameon=False)
    axt.set_ylim(0, 22)

    
    axt.legend(loc=2,bbox_to_anchor=(0.02,1.01), frameon=False)
    axt.set_xlabel('Time [d]', ha='center', fontsize=15,labelpad=3)
    axt.set_ylabel('$Biomass,\ Cx\ [g\ L^{-1}]$', fontsize=15,labelpad=3)
    axt.set_ylim(0, 22)
    axt.set_xlim(0, 28)
    
    fig.savefig(f"../results/plots/{name}PresentationDynamicBiomass.png", bbox_inches="tight", transparent=True)
    plt.close()