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
    return F_landau_num(*tuple(xx))

def parse_text(xx):
    yy = ''
    for x in xx:
        if x != '$' and x != '{' and x != '}':
            yy += x
    return yy

sp.init_printing()  # LaTeX like pretty printing for IPython
title = '$F_{Landau-Ginzburg-elastic-electrostriction}$'
T = 300.0
P0 = 0.54566
DW_size = 51

a_1 = 4.9*(T - 1103)*1E5*1E-10
a_11 = 6.5E8*1E-10
a_12 = 1.0E8*1E-10
G_11 = 0.6*0.98*1E-10*1E-10

c_1111 = 3.02E11*1E-10
c_1122 = 1.62E11*1E-10
c_1212 = 0.68E11*1E-10
Q_1111 = 0.032
Q_1122 = -0.016
Q_1212 = 0.01

q_1111 = 2*c_1111*Q_1111
q_1122 = c_1111*Q_1122 + c_1122*Q_1111
q_1212 = 2*c_1212*Q_1212

Px = np.array([ x for x, _ in zip(sp.numbered_symbols('Px_'), range(0,DW_size))])
Py = np.array([ x for x, _ in zip(sp.numbered_symbols('Py_'), range(0,DW_size))])
Pz = np.array([ x for x, _ in zip(sp.numbered_symbols('Pz_'), range(0,DW_size))])
#Px[0] = -1; Px[101] = 1; Py[0] = -1; Py[101] = -1; Pz[0] = -1; Pz[101] = -1

P = np.array([Px, Py, Pz])
Px_initial = np.array([(P0*4.0/np.pi)*math.atan((x - DW_size/2)/(DW_size/2.)) for x in range(0,DW_size)])
#Px_initial = np.array(DW_size*[-1*P0])
Py_initial = np.array(DW_size*[1*P0])
Pz_initial = np.array(DW_size*[1*P0])
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
                     np.random.rand(1,DW_size)[0]*0.0,
                     np.random.rand(1,DW_size)[0]*0.0,
                     np.random.rand(1,DW_size)[0]*0.0,
                     np.random.rand(1,DW_size)[0]*0.0,
                     np.random.rand(1,DW_size)[0]*0.0,
                     np.random.rand(1,DW_size)[0]*0.0]

print(variables_initial)
variables = to_variables(P) + to_variables(epsilon)

F_landau_sym = np.sum(a_1*(P[0]**2 + P[1]**2 + P[2]**2)
                    + a_11*(P[0]**4 + P[1]**4 + P[2]**4)
                    + a_12*(P[0]**2*P[1]**2 + P[1]**2*P[2]**2 + P[2]**2*P[0]**2)) \
                    + np.sum(G_11*(np.diff(P[0])**2 + np.diff(P[1])**2 + np.diff(P[2])**2)) \
                    + np.sum(0.5*c_1111*(epsilon_11**2 + epsilon_22**2 + epsilon_33**2)
                    + c_1122*(epsilon_11*epsilon_22 + epsilon_22*epsilon_33 + epsilon_33*epsilon_11)
                    + 0.5*c_1212*(epsilon_12**2 + epsilon_23**2 + epsilon_31**2)
                    - 0.5*q_1111*(epsilon_11*P[0]**2 + epsilon_22*P[1]**2 + epsilon_33*P[2]**2)
                    - q_1122*(epsilon_11*(P[1]**2+P[2]**2) + epsilon_22*(P[2]**2+P[0]**2) + epsilon_33*(P[0]**2+P[1]**2))
                    - 0.5*q_1212*(epsilon_12*P[0]*P[1] + epsilon_23*P[1]*P[2] + epsilon_31*P[2]*P[0]))

F_landau_num = sp.lambdify(variables, F_landau_sym, modules='numpy')

cons = ({'type': 'eq', 'fun': lambda x:  (to_tensor(np.array(x))[0,0] + P0)**2},
        {'type': 'eq', 'fun': lambda x:  (to_tensor(np.array(x))[0,-1] - P0)**2},
        {'type': 'eq', 'fun': lambda x:  (to_tensor(np.array(x))[1,0] - P0)**2},
        {'type': 'eq', 'fun': lambda x:  (to_tensor(np.array(x))[1,-1] - P0)**2},
        {'type': 'eq', 'fun': lambda x:  (to_tensor(np.array(x))[2,0] - P0)**2},
        {'type': 'eq', 'fun': lambda x:  (to_tensor(np.array(x))[2,-1] - P0)**2})

bnds = tuple((-1,1) for x in variables)

Pepsilon_ = minimize(F_landau_, variables_initial, method='SLSQP', constraints=cons, bounds=bnds, tol=1e-10)
Pepsilon = to_tensor(Pepsilon_.x)

print(to_tensor(Pepsilon_.x))
print(F_landau_(Pepsilon_.x))

# plot part
fig=plt.figure(figsize=(10, 10))
plt.suptitle(title, fontsize=24)
axp = fig.add_subplot(2, 1, 1)
## Move left y-axis and bottim x-axis to centre, passing through (0,0)
#ax.spines['left'].set_position('center')
#ax.spines['bottom'].set_position('center')
## Eliminate upper and right axes
#ax.spines['right'].set_color('none')
#ax.spines['top'].set_color('none')
## Show ticks in the left and lower axes only
#ax.xaxis.set_ticks_position('bottom')
#ax.yaxis.set_ticks_position('left')

#plt.subplots_adjust(hspace=0.0,wspace=0.5,left=0.10,right=0.99,top=0.99,bottom=0.1)
axp.plot(np.arange(-DW_size/2,DW_size/2)+1, Pepsilon[0], 'r-o', label='$P_{100}$')
axp.plot(np.arange(-DW_size/2,DW_size/2)+1, Pepsilon[1], 'g-s', label='$P_{010}$')
axp.plot(np.arange(-DW_size/2,DW_size/2)+1, Pepsilon[2], 'b-p', label='$P_{001}$')
axp.plot(np.arange(-DW_size/2,DW_size/2)+1, np.sqrt(Pepsilon[1]**2+Pepsilon[2]**2+Pepsilon[0]**2), 'k-', label='$|P_{total}|$')
axp.axhline(color='k', ls='dashed')
axp.axvline(color='k', ls='dashed')
axp.set_ylim(-1,1)
#axp.set_xlabel('lattice grid',fontsize=12)
axp.set_ylabel('$P (C/m^2)$',fontsize=18)
axp.grid()
axp.legend(loc='lower right')
#plt.plot(range(0,DW_size),P[2],'b-')

axe = fig.add_subplot(2, 1, 2)
axe.plot(np.arange(-DW_size/2,DW_size/2)+1, Pepsilon[3], 'r-o', label='$\epsilon_{11}$')
axe.plot(np.arange(-DW_size/2,DW_size/2)+1, Pepsilon[4], 'g-s', label='$\epsilon_{22}$')
axe.plot(np.arange(-DW_size/2,DW_size/2)+1, Pepsilon[5], 'b-p', label='$\epsilon_{33}$')
axe.plot(np.arange(-DW_size/2,DW_size/2)+1, Pepsilon[6], 'k-^', label='$\epsilon_{12}$')
axe.plot(np.arange(-DW_size/2,DW_size/2)+1, Pepsilon[7], 'y-<', label='$\epsilon_{23}$')
axe.plot(np.arange(-DW_size/2,DW_size/2)+1, Pepsilon[8], 'm->', label='$\epsilon_{31}$')
axe.axhline(color='k', ls='dashed')
axe.axvline(color='k', ls='dashed')
axe.set_ylabel('strain',fontsize=18)
axe.grid()
axe.legend(loc='lower right')

plt.savefig(parse_text(title)+'.pdf', format='pdf')
