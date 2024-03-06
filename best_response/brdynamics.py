import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from itertools import combinations
import copy

cur_belief = np.array([-70.0, 0.0])
big_target = np.array([50.0,0.0])
small_target = np.array([-50.0, 0.0])

# plot br sequentially; as a function of iters; end reuslts as a function of params; loss function at the equilibria;
# change cols in the table

def getdata(fileName):
    with open(fileName) as f:
        lines = f.readlines()
        x = np.empty([len(lines), 2])
        i = 0
        for line in lines:
            coordinates = line.split()
            x_1 = float(coordinates[0])
            x_2 = float(coordinates[1])
            x[i] = np.array([x_1, x_2])
            i+=1
    return x

def plot_trend(x, y):
   global cur_belief

   return

"""Given a distance, calculates its weight using Gaussian distribution"""
def g_fct(x, sd):
    global cur_belief
    c = np.exp(-(x - cur_belief[0]) ** 2 / (2 * sd ** 2))
    return c

def new_belief(x, y, sd1, sd2):
    c_1 = g_fct(x, sd1)
    c_2 = g_fct(y, sd2) 
    ret = (cur_belief[0] + c_1 * x + c_2 * y) / (1 + c_1 + c_2)
    # print(f'new belief: {ret}')
    return ret

#coef[0]: y, coef[1]: variance, coef[2]: target
def distance(x, coefs):
    return (new_belief(x, coefs[0], coefs[1], coefs[2]) - coefs[3]) ** 2

def best_points_cont(x, y, target, sd1, sd2):
    coefs = [y[0], sd1, sd2, target[0]]
    x0=x[0]
    result = opt.basinhopping(distance, x0, niter=1000, minimizer_kwargs={'args':coefs})
    # result = opt.minimize(distance, x0, args=coefs)
    print(result.message)
    x0 = np.array([result.x[0], 0.0])
    return x0


def big_endian_cont(x, y, sd1, sd2):
    global big_target
    print(':::::::::::::::::::::::BIG:::::::::::::::::::::::')
    return best_points_cont(x, y, big_target, sd1, sd2)
    # return best_combination(points, weighted_pos, small_target, num_points)

def small_endian_cont(x, y, sd1, sd2):
    global small_target
    print(':::::::::::::::::::::::SMALL:::::::::::::::::::::::')
    return best_points_cont(x, y, small_target, sd1, sd2)
    # return best_combination(points, weighted_pos, small_target, num_points)

def sim_cont(sd):
    cur_big = copy.deepcopy(cur_belief)
    cur_small = copy.deepcopy(cur_belief)
    # cur_big = np.array([60.0, 0.0]) 
    # cur_small = np.array([39.99999, 0.0])
    # cur_big = np.array([0.0, 0.0])
    # cur_small = np.array([0.0, 0.0])
    big_arr=[cur_big[0]]
    small_arr=[cur_small[0]]
    print('SIM STARTS')
    for i in range(100):
        last_big = cur_big
        last_small = cur_small
        print(f'------------------------ITERATION {i+1}---------------------------')
        cur_big = big_endian_cont(cur_big, last_small, sd[0], sd[1])
        n_b = new_belief(cur_big[0], last_small[0], sd[0], sd[1])
        print(f'belief: {n_b}, big: {cur_big[0]}, small: {last_small[0]}')
        cur_small = small_endian_cont(cur_small, last_big, sd[1], sd[0])
        n_b = new_belief(last_big[0], cur_small[0], sd[0], sd[1])
        print(f'belief: {n_b}, big: {last_big[0]}, small: {cur_small[0]}')
        print('------------------------ITERATION ENDS---------------------------')
        n_b = new_belief(cur_big[0], cur_small[0], sd[0], sd[1]) 
        print(f'belief: {n_b}')
        if (i>0 and (np.linalg.norm(cur_big-last_big) <= 0.001) and (np.linalg.norm(cur_small-last_small) <= 0.001)):
            print(f'sim converges at ITERATION {i}')
            n_b = new_belief(cur_big[0], cur_small[0], sd[0], sd[1]) 
            print(f'belief: {n_b}, big: {cur_big[0]}, small: {cur_small[0]}')
            break 
        big_arr.append(cur_big[0])
        small_arr.append(cur_small[0])
    print('SIM ENDS')

    iters = [i for i in range(len(big_arr))]
    y = [0 for _ in range(len(big_arr))]
    s_arr=np.arange(10, 10 + 10*len(big_arr),10)

    # plt.scatter(big_arr, y, c = 'blue', s = s_arr, label='big')
    # plt.scatter(small_arr, y, c = 'red', s = s_arr, label='small')

    plt.scatter(big_arr, iters, c = 'blue', label='big')
    plt.scatter(small_arr, iters, c = 'red', s = 20, label='small')
    plt.yticks(iters, iters)
    plt.xlabel('points chosen')
    plt.ylabel('iteration')
    plt.title('Points Chosen at Each Iteration')
    plt.grid()
    plt.legend()
    plt.show()

def main():
    # points = getdata('best_response/data3.txt')
    # plotdata(points)

    # sim_discrete(points, 10)
    sd=[10,10]
    
    sim_cont(sd)
    

    # weights =[g_fct(np.linalg.norm(point-cur_belief),0,20) for point in points]
    # for i in range(len(points)):
    #     print(f'point: {points[i][0]}, weight: {weights[i]}')
    # sim2(points)


    
if __name__ == "__main__":
    main()