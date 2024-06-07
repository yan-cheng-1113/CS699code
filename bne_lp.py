import gurobipy as gp
from gurobipy import Model, GRB
import numpy as np
import scipy.sparse as sp
from ortools.linear_solver import pywraplp

def solve(p, mu, tmu, eps):
    model = Model("BNE")

    # vars
    f_11 = model.addVar(vtype=GRB.CONTINUOUS, name='f_11')
    f_12 = model.addVar(vtype=GRB.CONTINUOUS, name='f_12') 
    f_21 = model.addVar(vtype=GRB.CONTINUOUS, name='f_21')
    f_22 = model.addVar(vtype=GRB.CONTINUOUS, name='f_22')
    f_arr = [f_11, f_12, f_21, f_22]

    # auxiliary vars to represent distance between f_ij and mu[0]
    d_1 = model.addVar(vtype=GRB.CONTINUOUS, name='d_1')
    d_2 = model.addVar(vtype=GRB.CONTINUOUS, name='d_2')
    d_3 = model.addVar(vtype=GRB.CONTINUOUS, name='d_3')
    d_4 = model.addVar(vtype=GRB.CONTINUOUS, name='d_4')

    # auxiliary vars for |f_ij - (t_1+mu_1)/2|
    y_1 = model.addVar(vtype=GRB.CONTINUOUS, name='y_1')
    y_2 = model.addVar(vtype=GRB.CONTINUOUS, name='y_2') 
    y_3 = model.addVar(vtype=GRB.CONTINUOUS, name='y_3')
    y_4 = model.addVar(vtype=GRB.CONTINUOUS, name='y_4')
    # auxiliary vars for |f_ij - (t_1+mu_2)/2|
    y_5 = model.addVar(vtype=GRB.CONTINUOUS, name='y_5')
    y_6 = model.addVar(vtype=GRB.CONTINUOUS, name='y_6')
    y_7 = model.addVar(vtype=GRB.CONTINUOUS, name='y_7')
    y_8 = model.addVar(vtype=GRB.CONTINUOUS, name='y_8')
    # auxiliary vars for |f_ij - (t_2+mu_1)/2|
    y_9 = model.addVar(vtype=GRB.CONTINUOUS, name='y_9')
    y_10 = model.addVar(vtype=GRB.CONTINUOUS, name='y_10')
    y_11 = model.addVar(vtype=GRB.CONTINUOUS, name='y_11')
    y_12 = model.addVar(vtype=GRB.CONTINUOUS, name='y_12')
    # auxiliary vars for |f_ij - (t_2+mu_2)/2|
    y_13 = model.addVar(vtype=GRB.CONTINUOUS, name='y_13')
    y_14 = model.addVar(vtype=GRB.CONTINUOUS, name='y_14')
    y_15 = model.addVar(vtype=GRB.CONTINUOUS, name='y_15')
    y_16 = model.addVar(vtype=GRB.CONTINUOUS, name='y_16')

    # objective function
    model.setObjective(d_1 + d_2 + d_3 + d_4 +
                       y_1 + y_2 + 10*y_3 + y_4 + 
                       y_5 + y_6 + 10*y_7 + y_8 +
                       y_9 + y_10 + 10*y_11 + y_12 +
                       y_13 + y_14 + 10*y_15 + y_16, GRB.MINIMIZE)

    # constraints
    model.addConstr(p[0]*y_1 + p[1]*y_2 <= p[0]*y_3 + p[1]*y_4 - eps)
    model.addConstr(p[0]*y_5 + p[1]*y_6 <= p[0]*y_7 + p[1]*y_8 - eps) 
    model.addConstr(p[0]*y_9 + p[1]*y_10 <= p[0]*y_11 + p[1]*y_12 - eps)
    model.addConstr(p[0]*y_13 + p[1]*y_14 <= p[0]*y_15 + p[1]*y_16 - eps) 
     
    # constraints for d
    model.addConstr(d_1 >= f_11 - mu[0])
    model.addConstr(d_1 >= mu[0] - f_11)
    model.addConstr(d_2 >= f_12 - mu[0])
    model.addConstr(d_2 >= mu[0] - f_12)
    model.addConstr(d_3 >= f_21 - mu[0])
    model.addConstr(d_3 >= mu[0] - f_21)
    model.addConstr(d_4 >= f_22 - mu[0])
    model.addConstr(d_4 >= mu[0] - f_22)

    # constraints for |f_ij - (t_1+mu_1)/2| 
    model.addConstr(y_1 >= f_11 - tmu[0])
    model.addConstr(y_1 >= tmu[0] - f_11)
    model.addConstr(y_2 >= f_12 - tmu[0])
    model.addConstr(y_2 >= tmu[0] - f_12)
    model.addConstr(y_3 >= f_21 - tmu[0])
    model.addConstr(y_3 >= tmu[0] - f_21)
    model.addConstr(y_4 >= f_22 - tmu[0])
    model.addConstr(y_4 >= tmu[0] - f_22)
    # constraints for |f_ij - (t_1+mu_2)/2| 
    model.addConstr(y_5 >= f_21 - tmu[1])
    model.addConstr(y_5 >= tmu[1] -f_21)
    model.addConstr(y_6 >= f_22 - tmu[1])
    model.addConstr(y_6 >= tmu[1] -f_22) 
    model.addConstr(y_7 >= f_11 - tmu[1])
    model.addConstr(y_7 >= tmu[1] -f_11) 
    model.addConstr(y_8 >= f_12 - tmu[1])
    model.addConstr(y_8 >= tmu[1] -f_12)
    # constraints for |f_ij - (t_2+mu_1)/2|
    model.addConstr(y_9 >= f_11 - tmu[2])
    model.addConstr(y_9 >= tmu[2] -f_11)
    model.addConstr(y_10 >= f_21 - tmu[2])
    model.addConstr(y_10 >= tmu[2] -f_21) 
    model.addConstr(y_11 >= f_12 - tmu[2])
    model.addConstr(y_11 >= tmu[2] -f_12) 
    model.addConstr(y_12 >= f_22 - tmu[2])
    model.addConstr(y_12 >= tmu[2] -f_22)
    # constraints for |f_ij - (t_2+mu_2)/2|
    model.addConstr(y_13 >= f_12 - tmu[3])
    model.addConstr(y_13 >= tmu[3] -f_12)
    model.addConstr(y_14 >= f_22 - tmu[3])
    model.addConstr(y_14 >= tmu[3] -f_22) 
    model.addConstr(y_15 >= f_11 - tmu[3])
    model.addConstr(y_15 >= tmu[3] -f_11) 
    model.addConstr(y_16 >= f_21 - tmu[3])
    model.addConstr(y_16 >= tmu[3] -f_21)

    model.optimize()
    # for v in f_arr:
    #     print(f'{v.varName} {v.x}')
    for v in model.getVars():
        print(f'{v.varName} {v.x}')
    print(f'Obj: {model.objVal}')


def main():
    p = [0.6, 0.4]
    mu = [100, 110, 85]
    t = [120, 80]
    tmu = [(t[0]+mu[1])/2, (t[0]+mu[2])/2, (t[1]+mu[1])/2, (t[1]+mu[2])/2]
    solve(p, mu, tmu, 5)


if __name__ == "__main__":
    main()