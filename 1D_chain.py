#!/usr/bin/env python
import math
import sympy as sp
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def to_variables(xx):
    return reduce(lambda x,y : x+y, xx.tolist())

def to_tensor(xx):
    return xx.reshape(3,102)

def F_landau_(xx):
    """ Helper for receiving vector parameters """
    #variables = to_variables(xx)
    return F_landau_num(*tuple(xx))

def parse_text(xx):
    yy = ''
    for x in xx:
        if x != '$' and x != '{' and x != '}':
            yy += x
    return yy

sp.init_printing()  # LaTeX like pretty printing for IPython
title = '$F_{Landau}$ Part'
T = 300

a_1 = 4.9*(T - 1103*1E5)
a_11 = 5.42*1E8
a_12 = 1.5*1E8
G_11 = 0.6*0.98*1E10

Px = np.array([ x for x, _ in zip(sp.numbered_symbols('Px'), range(0,102))])
Px_initial = np.array([(2/np.pi)*math.atan((x - 50)/50.0) for x in range(0,102)])
Py = np.array([ x for x, _ in zip(sp.numbered_symbols('Py'), range(0,102))])
Py_initial = np.array(102*[-1])
Pz = np.array([ x for x, _ in zip(sp.numbered_symbols('Pz'), range(0,102))])
Pz_initial = np.array(102*[-1])

P = np.array([Px, Py, Pz])
P_initial = np.array([Px_initial, Py_initial ,Pz_initial])

variables = to_variables(P)

F_landau_sym = np.sum(a_1*(P[0][:-1]**2 + P[1][:-1]**2 + P[2][:-1]**2) + a_11*(P[0][:-1]**4 + P[1][:-1]**4 + P[2][:-1]**4) + a_12*(P[0][:-1]**2*P[1][:-1]**2 + P[1][:-1]**2*P[2][:-1]**2 + P[2][:-1]**2*P[0][:-1]**2)
                      + G_11*(np.diff(P[0])**2 + np.diff(P[1])**2 + np.diff(P[2])**2))
F_landau_num = sp.lambdify(variables, F_landau_sym, modules='numpy')

P_ = minimize(F_landau_, P_initial, tol=1e-2)

P = to_tensor(P_.x)
fig=plt.figure(figsize=(10, 6))
#plt.subplots_adjust(hspace=0.0,wspace=0.5,left=0.10,right=0.99,top=0.99,bottom=0.1)
plt.plot(range(0,102), P[0], 'r-', label='[100]')
plt.plot(range(0,102), np.sqrt(P[1]**2+P[2]**2), 'g-', label='[011]')
plt.title(title)
plt.grid()
plt.legend()
#plt.plot(range(0,102),P[2],'b-')
plt.savefig(parse_text(title)+'.pdf', format='pdf')
