#!/usr/bin/env python
import math
import sympy as sp
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def to_variables(xx):
    return reduce(lambda x,y : x+y, xx.tolist())

def to_tensor(xx):
    return xx.reshape(9,100)

def to_unilist(xx):
    return([x for x in [y for y in [xx]]])

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
title = '$F_{Landau-Ginzburg-strain}$ Part'
T = 300

a_1 = 4.9*(T - 1103)*1E5
a_11 = 5.42E8
a_12 = 1.5E8
G_11 = 0.6*0.98*1E10

c_1111 = 3.02E11
c_1122 = 1.62E11
c_1212 = 0.68E11
Q_1111 = 0.032
Q_1122 = -0.016
Q_1212 = 0.01

q_1111 = 2*c_1111*Q_1111
q_1122 = 2*c_1111*Q_1122
q_1212 = 2*c_1212*Q_1212

Px = np.array([ x for x, _ in zip(sp.numbered_symbols('Px'), range(0,102))])
Py = np.array([ x for x, _ in zip(sp.numbered_symbols('Py'), range(0,102))])
Pz = np.array([ x for x, _ in zip(sp.numbered_symbols('Pz'), range(0,102))])
Px[0] = -1; Px[101] = 1; Py[0] = -1; Py[101] = -1; Pz[0] = -1; Pz[101] = -1

P = np.array([Px, Py, Pz])

Px_initial = np.array([(2.0/np.pi)*math.atan((x - 50)/50.0) for x in range(0,102)])
Py_initial = np.array(102*[-1])
Pz_initial = np.array(102*[-1])
P_initial = np.array([Px_initial[1:101], Py_initial[1:101] ,Pz_initial[1:101]])

epsilon_11 = np.array([ x for x, _ in zip(sp.numbered_symbols('epsilon_11'), range(0,102))])
epsilon_22 = np.array([ x for x, _ in zip(sp.numbered_symbols('epsilon_22'), range(0,102))])
epsilon_33 = np.array([ x for x, _ in zip(sp.numbered_symbols('epsilon_33'), range(0,102))])
epsilon_12 = np.array([ x for x, _ in zip(sp.numbered_symbols('epsilon_12'), range(0,102))])
epsilon_23 = np.array([ x for x, _ in zip(sp.numbered_symbols('epsilon_23'), range(0,102))])
epsilon_31 = np.array([ x for x, _ in zip(sp.numbered_symbols('epsilon_31'), range(0,102))])
epsilon = np.array([epsilon_11, epsilon_22, epsilon_33, epsilon_12, epsilon_23, epsilon_31])

variables_initial = [Px_initial[1:101],
                     Py_initial[1:101],
                     Pz_initial[1:101],
                     np.random.rand(1,101)[0],
                     np.random.rand(1,101)[0],
                     np.random.rand(1,101)[0],
                     np.random.rand(1,101)[0],
                     np.random.rand(1,101)[0],
                     np.random.rand(1,101)[0]]

variables = to_variables(P[:,1:101]) + to_variables(epsilon)
print(variables_initial)
F_landau_sym = np.sum(a_1*(P[0][:-1]**2 + P[1][:-1]**2 + P[2][:-1]**2)
                    + a_11*(P[0][:-1]**4 + P[1][:-1]**4 + P[2][:-1]**4)
                    + a_12*(P[0][:-1]**2*P[1][:-1]**2 + P[1][:-1]**2*P[2][:-1]**2 + P[2][:-1]**2*P[0][:-1]**2)
                    + G_11*(np.diff(P[0])**2 + np.diff(P[1])**2 + np.diff(P[2])**2)
                    + 0.5*c_1111*(epsilon_11[:-1]**2 + epsilon_22[:-1]**2 + epsilon_33[:-1]**2)
                    + c_1122*(epsilon_11[:-1]*epsilon_22[:-1] + epsilon_22[:-1]*epsilon_33[:-1] + epsilon_33[:-1]*epsilon_11[:-1])
                    + 0.5*c_1212*(epsilon_12[:-1]**2 + epsilon_23[:-1]**2 + epsilon_31[:-1]**2)
                    + 0.5*q_1111*(epsilon_11[:-1]*P[0][:-1]**2 + epsilon_22[:-1]*P[1][:-1]**2 + epsilon_33[:-1]*P[2][:-1]**2)
                    + q_1122*(epsilon_11[:-1]*P[1][:-1]**2 + epsilon_22[:-1]*P[2][:-1]**2 + epsilon_33[:-1]*P[0][:-1]**2)
                    + 0.5*q_1212*(epsilon_12[:-1]*P[0][:-1]*P[1][:-1] + epsilon_23[:-1]*P[1][:-1]*P[2][:-1] + epsilon_31[:-1]*P[2][:-1]*P[0][:-1]))

F_landau_num = sp.lambdify(variables, F_landau_sym, modules='numpy')

Pepsilon_ = minimize(F_landau_, variables_initial, tol=1e-2)
Pepsilon = to_tensor(Pepsilon_.x)

# plot part
fig=plt.figure(figsize=(10, 6))
#plt.subplots_adjust(hspace=0.0,wspace=0.5,left=0.10,right=0.99,top=0.99,bottom=0.1)
plt.plot(range(1,101), Pepsilon[0], 'r-', label='[100]')
plt.plot(range(1,101), np.sqrt(Pepsilon[1]**2+Pepsilon[2]**2), 'g-', label='[011]')
plt.title(title)
plt.grid()
plt.legend()
#plt.plot(range(0,102),P[2],'b-')
plt.savefig(parse_text(title)+'.pdf', format='pdf')
