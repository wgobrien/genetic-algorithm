import numpy as np

def rastrigin(parameter):
    n = 3
    val = 10 * n
    for x in parameter.values():
        val += x**2 - 10 * np.cos(x*2*np.pi)
    return val