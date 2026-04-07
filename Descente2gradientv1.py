import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
from pyomo.environ import *

def trouver_argmin_ipopt (f):
    #on suppose que la fonction est ok avec le solveur
    fonction = ConcreteModel()
    fonction.x = Var(domain=Reals)
    fonction.obj = Objective(expr = f(fonction.x), sense = minimize)
    solver = SolverFactory('ipopt')
    solver.solve(fonction)
    print("la valeur qui minimise la fonction est x = " , fonction.x.value)
    return fonction.x.value

def N(x):
    return abs(x)

def gradient_dim1(f,fp,xz):
    # les arguments sont respectivement f, f' et x indice 0
    k=0 # nb d'itérations
    eps1=0.05   # habituellement , espilon ≃ 10 exp-8
    eps2=0.5    # habituellement , epsilon ≃ 10 exp -6
    x=[xz,xz] # couple x0 , x1 puis x1,x2 etc ....
    d=-fp(xz)
    def phi(y):
            return f(xz+y*d)
    x[1]=xz+trouver_argmin_ipopt(phi)*d
    cond_arret= (N(fp(x[1]))<=eps1) or (N(x[0]-x[1])<=eps2*N(x[1]))
    while cond_arret==False :
        k=k+1
        d=-fp(x[1])
        def phi(y):
            return f(x[1]+y*d)
        p=trouver_argmin_ipopt(phi)
        
        x[0]=x[1]
        x[1]=x[0]+p*d
        cond_arret= (N(fp(x[1]))<=eps1) or (N(x[0]-x[1])<=eps2*N(x[1]))
    return ( k , x[1])
