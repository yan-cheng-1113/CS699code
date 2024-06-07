import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.optimize import fsolve 
import copy

cur_belief = np.array([0.0, 0.0])
big_target = [np.array([5.0,0.0]), np.array([20.0,0.0])]
small_target = [np.array([-5.0, 0.0]), np.array([-20.0, 0.0])]
prob_big = [0.7, 0.3]
prob_small = [0.7, 0.3]

def v_solver(vars):
    global big_target, small_target, prob_big, prob_small
    v1, v2, v3, v4 = vars
    eq1 = prob_small[0]*((big_target[0][0]-v1)**2-(big_target[0][0]-v3)**2) + prob_small[1]*((big_target[0][0]-v2)**2-(big_target[0][0]-v4)**2)
    eq2 = prob_small[0]*((big_target[1][0]-v1)**2-(big_target[1][0]-v3)**2) + prob_small[1]*((big_target[1][0]-v2)**2-(big_target[1][0]-v4)**2)
    eq3 = prob_big[0]*((small_target[0][0]-v1)**2-(small_target[0][0]-v2)**2) + prob_big[1]*((small_target[0][0]-v3)**2-(small_target[0][0]-v4)**2)
    eq4 = prob_big[0]*((small_target[1][0]-v1)**2-(small_target[1][0]-v2)**2) + prob_big[1]*((small_target[1][0]-v3)**2-(small_target[1][0]-v4)**2) 
    return [eq1, eq2, eq3, eq4]

def g_fct(x, sd):
    global cur_belief
    c = np.exp(-(x - cur_belief[0]) ** 2 / (2 * sd ** 2))
    return c

def new_belief_gaussian(x, y, sd1, sd2):
    # x = x - d + 2
    c_1 = g_fct(x, sd1)
    c_2 = g_fct(y, sd2) 
    ret = (cur_belief[0] + c_1 * x + c_2 * y) / (1 + c_1 + c_2)
    # print(f'new belief: {ret}')
    return ret

def new_belief_mean(x_i, x_arr, n):
    ret = x_i
    for x in x_arr:
        ret += x
    return (ret / n)

def distance_mean():
    pass

def distance(x, coefs):
    result = 0
    for i in range(len(coefs[4])):
        # coef[0]: other player's point, coef[1]：sd1, coef[2]: sd2, coef[3]: target point
        result += coefs[4][i] * ((new_belief_gaussian(x, coefs[0][i][0], coefs[1], coefs[2]) - coefs[3]) ** 2)
    return result

def best_points(x, y, target, sd1, sd2, prob):
    ret = []
    for i in range(len(target)):
        coefs = [y, sd1, sd2, target[i][0], prob]
        x0=x[i][0]
        result = opt.basinhopping(distance, x0, niter=1000, minimizer_kwargs={'args':coefs})
        # result = opt.minimize(distance, x0, args=coefs)
        # print(result.message)
        new_x = np.array([result.x[0], 0.0])
        ret.append(new_x)
    return ret


def big_endian_cont(x, y, sd1, sd2):
    global big_target, prob_small
    print(':::::::::::::::::::::::BIG:::::::::::::::::::::::')
    return best_points(x, y, big_target, sd1, sd2, prob_small)

def small_endian_cont(x, y, sd1, sd2):
    global small_target, prob_big
    print(':::::::::::::::::::::::SMALL:::::::::::::::::::::::')
    return best_points(x, y, small_target, sd1, sd2, prob_big)

def sim_cont(sd):
    cur_big = [copy.deepcopy(cur_belief), copy.deepcopy(cur_belief)]
    cur_small = [copy.deepcopy(cur_belief), copy.deepcopy(cur_belief)]
    # cur_big = np.array([60.0, 0.0]) 
    # cur_small = np.array([39.99999, 0.0])
    big1_arr=[cur_big[0][0]]
    big2_arr=[cur_big[1][0]]
    small1_arr=[cur_small[0][0]]
    small2_arr=[cur_small[1][0]]
    print('SIM STARTS')
    for i in range(100):
        last_big = cur_big
        last_small = cur_small
        print(f'------------------------ITERATION {i+1}---------------------------')
        cur_big = big_endian_cont(cur_big, last_small, sd[0], sd[1])
        # n_b = new_belief(cur_big[0], last_small[0], sd[0], sd[1])
        # print(f'belief: {n_b}, big: {cur_big[0]}, small: {last_small[0]}')
        print(f'big_type1: {cur_big[0][0]}, big_type2: {cur_big[1][0]}, small_type1: {last_small[0][0]}, small_type2: {last_small[1][0]}')
        cur_small = small_endian_cont(cur_small, last_big, sd[1], sd[0])
        # n_b = new_belief(last_big[0], cur_small[0], sd[0], sd[1])
        # print(f'belief: {n_b}, big: {last_big[0]}, small: {cur_small[0]}')
        print(f'big_type1: {last_big[0][0]}, big_type2: {last_big[1][0]}, small_type1: {cur_small[0][0]}, small_type2: {cur_small[1][0]}')
        print(':::::::::::::::::::::::SUMMARY:::::::::::::::::::::::')
        print(f'big_type1: {cur_big[0][0]}, big_type2: {cur_big[1][0]}, small_type1: {cur_small[0][0]}, small_type2: {cur_small[1][0]}')
        print('------------------------ITERATION ENDS---------------------------')
        # n_b = new_belief(cur_big[0], cur_small[0], sd[0], sd[1]) 
        # print(f'belief: {n_b}')
        flag = 0
        for j in range(len(cur_big)):
            if (np.linalg.norm(cur_big[j]-last_big[j]) > 0.001):
                flag = 1

        for j in range(len(cur_small)):
            if (np.linalg.norm(cur_small[j]-last_small[j]) > 0.001):
                flag = 1

        if (i>0 and flag == 0):
            print(f'sim converges at ITERATION {i}')
            # n_b = new_belief(cur_big[0], cur_small[0], sd[0], sd[1]) 
            # print(f'belief: {n_b}, big: {cur_big[0]}, small: {cur_small[0]}')
            print(f'big_type1: {cur_big[0][0]}, big_type2: {cur_big[1][0]}, small_type1: {cur_small[0][0]}, small_type2: {cur_small[1][0]}')
            break 
        big1_arr.append(cur_big[0][0])
        big2_arr.append(cur_big[1][0])
        small1_arr.append(cur_small[0][0])
        small2_arr.append(cur_small[1][0])
    print('SIM ENDS')

def main():
    sd=[100,100]
    sim_cont(sd)
    # v1,v2,v3,v4 = fsolve(v_solver, (0,0,0,0))
    # print(v1, v2, v3, v4)
    

if __name__ == "__main__":
    main()