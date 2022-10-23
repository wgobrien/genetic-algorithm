import numpy as np

# see https://en.wikipedia.org/wiki/Test_functions_for_optimization for more

def rastrigin(parameters):
    '''
    The function has one global minimum f(x*)=0 at x*=(0,...,0).
    '''
    n = len(parameters.values())
    val = 10 * n
    for x in parameters.values():
        val += x**2 - 10 * np.cos(x*2*np.pi)
    return val

def rosenbrock(parameters):
    '''
    The function has one global minimum f(x*)=0 at x*=(1,...,1).
    '''
    n = len(parameters.values())
    x = list(parameters.values())
    val = 0
    for i in range(n-1):
        val += 100*(x[i+1] - x[i]**2)**2 + (x[i]-1)**2
    return val

def sphere_function(parameters):
    '''
    The function has one global minimum f(x*)=0 at x*=(0,...,0).
    '''
    n = len(parameters.values())
    x = list(parameters.values())

    val = 0
    for i in range(n):
        val += x[i]**2
    return val

def ackley_function(parameters, amin=True):
    '''
    The function has one global minimum f(0,0)=0.
    '''
    amin = 1 if amin else -1 
    x = list(parameters.values())
    n = len(x)
    if n != 2:
        raise ValueError('Invalid dimensions. Ackley function takes values in |R^2')
    return amin*(-20*np.exp(-.2*np.sqrt(.5*(x[0]**2+x[1]**2))) - np.exp(.5*(np.cos(2*np.pi*x[0]+np.cos(2*np.pi*x[1])))) + np.exp(1) + 20)