#!/usr/bin/env python
import math
import sympy as sp
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def to_variables(xx):
    return reduce(lambda x,y : x+y, xx.tolist())

def to_tensor(xx):
    return xx.reshape(9,DW_size)

def to_unilist(xx):
    return([x for x in [y for y in [xx]]])

def F_landau_(xx):
    """ Helper for receiving vector parameters """
    print('... ... ...')
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
DW_size = 102

a_1 = 4.9*(T - 1103)*1E5*1E-10
a_11 = 5.42E8*1E-10
a_12 = 1.5E8*1E-10
G_11 = 0.6*0.98*1E-10*1E-10

c_1111 = 3.02E11*1E-10
c_1122 = 1.62E11*1E-10
c_1212 = 0.68E11*1E-10
Q_1111 = 0.032
Q_1122 = -0.016
Q_1212 = 0.01

q_1111 = 2*c_1111*Q_1111
q_1122 = 2*c_1111*Q_1122
q_1212 = 2*c_1212*Q_1212

Px = np.array([ x for x, _ in zip(sp.numbered_symbols('Px_'), range(0,DW_size))])
Py = np.array([ x for x, _ in zip(sp.numbered_symbols('Py_'), range(0,DW_size))])
Pz = np.array([ x for x, _ in zip(sp.numbered_symbols('Pz_'), range(0,DW_size))])
#Px[0] = -1; Px[101] = 1; Py[0] = -1; Py[101] = -1; Pz[0] = -1; Pz[101] = -1

P = np.array([Px, Py, Pz])

Px_initial = np.array([(2.0/np.pi)*math.atan((x - 50)/50.0) for x in range(0,DW_size)])
Py_initial = np.array(DW_size*[-1])
Pz_initial = np.array(DW_size*[-1])
P_initial = np.array([Px_initial, Py_initial, Pz_initial])

epsilon_11 = np.array([ x for x, _ in zip(sp.numbered_symbols('epsilon_11_'), range(0,DW_size))])
epsilon_22 = np.array([ x for x, _ in zip(sp.numbered_symbols('epsilon_22_'), range(0,DW_size))])
epsilon_33 = np.array([ x for x, _ in zip(sp.numbered_symbols('epsilon_33_'), range(0,DW_size))])
epsilon_12 = np.array([ x for x, _ in zip(sp.numbered_symbols('epsilon_12_'), range(0,DW_size))])
epsilon_23 = np.array([ x for x, _ in zip(sp.numbered_symbols('epsilon_23_'), range(0,DW_size))])
epsilon_31 = np.array([ x for x, _ in zip(sp.numbered_symbols('epsilon_31_'), range(0,DW_size))])
epsilon = np.array([epsilon_11, epsilon_22, epsilon_33, epsilon_12, epsilon_23, epsilon_31])

variables_initial = [Px_initial,
                     Py_initial,
                     Pz_initial,
                     np.random.rand(1,DW_size)[0]/100.0,
                     np.random.rand(1,DW_size)[0]/100.0,
                     np.random.rand(1,DW_size)[0]/100.0,
                     np.random.rand(1,DW_size)[0]/100.0,
                     np.random.rand(1,DW_size)[0]/100.0,
                     np.random.rand(1,DW_size)[0]/100.0]

variables = to_variables(P) + to_variables(epsilon)

F_landau_sym = np.sum(a_1*(P[0]**2 + P[1]**2 + P[2]**2)
                    + a_11*(P[0]**4 + P[1]**4 + P[2]**4)
                    + a_12*(P[0]**2*P[1]**2 + P[1]**2*P[2]**2 + P[2]**2*P[0]**2)) \
                    + np.sum(G_11*(np.diff(P[0])**2 + np.diff(P[1])**2 + np.diff(P[2])**2)) \
                    + np.sum(0.5*c_1111*(epsilon_11**2 + epsilon_22**2 + epsilon_33**2)
                    + c_1122*(epsilon_11*epsilon_22 + epsilon_22*epsilon_33 + epsilon_33*epsilon_11)
                    + 0.5*c_1212*(epsilon_12**2 + epsilon_23**2 + epsilon_31**2)
                    + 0.5*q_1111*(epsilon_11*P[0]**2 + epsilon_22*P[1]**2 + epsilon_33*P[2]**2)
                    + q_1122*(epsilon_11*P[1]**2 + epsilon_22*P[2]**2 + epsilon_33*P[0]**2)
                    + 0.5*q_1212*(epsilon_12*P[0]*P[1] + epsilon_23*P[1]*P[2] + epsilon_31*P[2]*P[0]))

print(F_landau_sym)
F_landau_num = sp.lambdify(variables, F_landau_sym, modules='numpy')

cons = [{'type': 'eq', 'fun': lambda x:  (to_tensor(np.array(x))[0,0] + 1)**2},
        {'type': 'eq', 'fun': lambda x:  (to_tensor(np.array(x))[0,-1] - 1)**2},
        {'type': 'eq', 'fun': lambda x:  (to_tensor(np.array(x))[1,0] + 1)**2},
        {'type': 'eq', 'fun': lambda x:  (to_tensor(np.array(x))[1,-1] + 1)**2},
        {'type': 'eq', 'fun': lambda x:  (to_tensor(np.array(x))[2,0] + 1)**2},
        {'type': 'eq', 'fun': lambda x:  (to_tensor(np.array(x))[2,-1] + 1)**2}]

Pepsilon_ = minimize(F_landau_, variables_initial, method='SLSQP', constraints=cons, tol=1e-3)
Pepsilon = to_tensor(Pepsilon_.x)

print(Pepsilon_.x)
print(F_landau_(Pepsilon_.x))
# plot part
fig=plt.figure(figsize=(10, 6))
#plt.subplots_adjust(hspace=0.0,wspace=0.5,left=0.10,right=0.99,top=0.99,bottom=0.1)
plt.plot(range(0,DW_size), Pepsilon[0], 'r-', label='[100]')
plt.plot(range(0,DW_size), np.sqrt(Pepsilon[1]**2+Pepsilon[2]**2), 'g-', label='[011]')
plt.title(title)
plt.grid()
plt.legend()
#plt.plot(range(0,DW_size),P[2],'b-')
plt.savefig(parse_text(title)+'.pdf', format='pdf')
