import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import copy
import random

big_target = np.array([10.0,0.0])
small_target = np.array([-10.0, 0.0])

def distance(x,  coefs):
    nb = new_belief(x, coefs[0])
    return (nb - coefs[1]) ** 2

def new_belief(x, y):
    return (x + y) / 2

def best_response(x, y, target, threshold):
    x0 = x[0]
    coefs = [y[0], target[0]]
    cons = ({'type': 'ineq',
       'fun': lambda x0: threshold - abs(x0 - y)})
    result = opt.minimize(distance, x0, args=coefs, constraints=cons)
    print(result.message)
    x0 = np.array([result.x[0], 0.0])
    return x0

def big_br(x, y, threshold):
    global big_target
    print(':::::::::::::::::::::::BIG:::::::::::::::::::::::')
    return best_response(x, y, big_target, threshold)

def small_br(x, y, threshold):
    global small_target
    print(':::::::::::::::::::::::SMALL:::::::::::::::::::::::')
    return best_response(x, y, small_target, threshold)

def sim():
    global cur_big, cur_small
    threshold = abs(big_target[0] - small_target[0])
    # cur_big = copy.deepcopy(big_target)
    # cur_small = copy.deepcopy(small_target)
    cur_big = np.array([100.0, 0.0])
    cur_small = np.array([-100.0, 0.0])
    big_arr = [cur_big[0]]
    small_arr = [cur_small[0]]
    for i in range(10):
        last_big = cur_big
        last_small = cur_small
        print(f'------------------------ITERATION {i+1}---------------------------')
        cur_big = big_br(cur_big, last_small, threshold)
        n_b = new_belief(cur_big[0], last_small[0])
        print(f'belief: {n_b}, big: {cur_big[0]}, small: {last_small[0]}')
        cur_small = small_br(cur_small, last_big, threshold)
        n_b = new_belief(last_big[0], cur_small[0])
        print(f'belief: {n_b}, big: {last_big[0]}, small: {cur_small[0]}')
        print('------------------------ITERATION ENDS---------------------------')
        if (i>0 and (np.linalg.norm(cur_big-last_big) <= 0.001) and (np.linalg.norm(cur_small-last_small) <= 0.001)):
            print(f'sim converges at ITERATION {i}')
            n_b = new_belief(cur_big[0], cur_small[0]) 
            print(f'belief: {n_b}, big: {cur_big[0]}, small: {cur_small[0]}')
            break 
        big_arr.append(cur_big[0])
        small_arr.append(cur_small[0])
    print('SIM ENDS')
    plot_trend(big_arr, small_arr)

def plot_trend(x_arr, y_arr):
    iters = [i for i in range(len(x_arr))]
    plt.scatter(x_arr, iters, c = 'blue', label='big')
    plt.scatter(y_arr, iters, c = 'red', s=20, label='small')
    plt.yticks(iters, iters)
    plt.xlabel('points chosen')
    plt.ylabel('iteration')
    plt.legend()
    plt.grid()
    plt.show()
    

def main():
    sim()
    # plot()

if __name__ == "__main__":
    main()