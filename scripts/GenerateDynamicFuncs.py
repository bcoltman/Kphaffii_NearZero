from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy import special
from math import floor, log10

def fit_function(x, y, initial_search_parameters, function, bounds=(-np.inf, np.inf), sigma=None):
    if function == "asymptotic":
        function = asymptotic_function
    elif function == "quadratic":
        function = quadratic_function
    elif function == "cubic":
        function = cubic_function
    elif function == "quartic":
        function = quartic_function
    elif function == "linear":
        function = linear_function
    elif function == "gompertz":
        function = gompertz_function
    elif function == "error_func":
        function = error_func

        
    popt, pcov = curve_fit(function,
                           x, y,
                           p0=initial_search_parameters,
                           sigma=sigma,
                           absolute_sigma=True,
                           maxfev=50000, 
                           bounds=bounds, ftol=1.5e-9)
    y_pred = function(x, *popt)
    
    rmse = np.sqrt(mean_squared_error(y,y_pred))
    r2 = r2_score(y,y_pred)
    
    p = len(initial_search_parameters) # number of parameters
    n = len(y) # number of observations
    
    if (p+1)>=n:
        return rmse, r2, popt, pcov
    else:
        adj_r2 = 1 - ((1-r2)*(n-1))/(n-p-1)
        return rmse, adj_r2, popt, pcov
    
    

def function_pred_and_deriv(x, function, *args):
    if function == "asymptotic":
        function = asymptotic_function
        func_der = asymptotic_derivative
    elif function == "quadratic":
        function = quadratic_function
        func_der = quadratic_derivative
    elif function == "cubic":
        function = cubic_function
        func_der = cubic_derivative
    elif function == "quartic":
        function = quartic_function
        func_der = quartic_derivative
    elif function == "linear":
        function = linear_function
        func_der = linear_derivative
    elif function == "gompertz":
        function = gompertz_function
        func_der = gompertz_derivative
    elif function == "error_func":
        function = error_func
        func_der = error_func_derivative
        
    
    y_pred = function(x, *args)
    derivative = func_der(x, *args)
    
    return y_pred, derivative



def linear_function(x, *args):
    a, b = args
    return a*x + b

def linear_derivative(x, *args):
    a, b = args
    return np.full_like(x, a)

def quadratic_function(x,*args):
    a, b, c = args
    return a*x**2 + b*x + c
def quadratic_derivative(x, *args):
    a, b, c = args
    return 2*a*x + b

def cubic_function(x,*args):
    a, b, c, d = args
    return a*x**3 + b*x**2 + c*x + d
def cubic_derivative(x, *args):
    a, b, c, d = args
    return 3*a*x**2 + 2*b*x + c

def quartic_function(x,*args):
    a, b, c, d, e = args
    return a*x**4 + b*x**3 + c*x**2 + d*x + e
def quartic_derivative(x, *args):
    a, b, c, d, e = args
    return 4*a*x**3 + 3*b*x**2 + 2*c*x + d

def gompertz_function(x,a,b,c):
    # return a * np.exp(-np.exp(b-c*x))
    return a * np.exp(-b*np.exp(-c*x))

def gompertz_derivative(x,a,b,c):
    # return a*c* np.exp(b - np.exp(b - c*x) - c*x)
    return a*b*c*np.exp(b*(-np.exp(-c*x)) - c*x)

def asymptotic_function(x,*args):
    """
    Defines a limited growth where y approaches a horizontal asymptote as x tends to infinity. 
    Alternatively known as:
    - Monomolecular growth
    - Mitscherlich law
    - von Bertalanffy law
    
    a is the maximum attainable Y
    b is Y at x=0
    c is proportional to the relative rate of Y increase while X increases
    """
    a, b, c = args
    return a - (a - b) * np.exp(-c* x)

    
def asymptotic_derivative(x,*args):
    """
    Calculate the instantaneous rate change of the asymptotic function
    """
    a,b,c = args
    return (a-b) * (np.exp(-c*x) * c) 

def error_func(x, a, b, c, d):
    return d + 0.5*c*(1 + special.erf(a*(x-b)))

def error_func_derivative(x, a, b, c, d):
    return (a*c*np.exp((-a**2)*(-b + x)**2))/np.sqrt(np.pi)

def smarter_round(sig):
    def rounder(x):
        offset = sig - floor(log10(abs(x)))
        initial_result = round(x, offset)
        if str(initial_result)[-1] == '5' and initial_result == x:
            return round(x, offset - 2)
        else:
            return round(x, offset - 1)
    return rounder