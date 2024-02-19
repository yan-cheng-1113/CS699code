import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from itertools import combinations
import copy

cur_belief = np.array([0.0, 0.0])
big_target = np.array([30.0,0.0])
small_target = np.array([10.0, 0.0])

"""Load into data"""
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

def get_indices(list, points):
    indices = []
    for point in list:
        #print(f'current point {point}')
        #print(np.flatnonzero((point==points).all(1))[0])
        indices.append(np.flatnonzero((point==points).all(1))[0])
    
    for i in range(len(indices)):
        indices[i] += 1
    return indices

def plotdata(points):
   plt.scatter(points[:,0], points[:,1], label='sources')
   plt.scatter(big_target[0], big_target[1], c='purple', s=100, label='big_target')
   plt.scatter(small_target[0], small_target[1], c='orange', s=100, label="small_target")
   plt.scatter(cur_belief[0], cur_belief[1], c='green', s=100, label='current_belief')
   plt.title('Data Points Distribution')
   plt.legend()
   plt.show() 
   return

"""Given a distance, calculates its weight using Gaussian distribution"""
def g_fct(x, variance):
    global cur_belief
    c = np.exp(-(x - cur_belief[0]) ** 2 / (2 * variance ** 2))
    return c

def new_belief(x, y, var):
    c_1 = g_fct(x, var)
    c_2 = g_fct(y, var) 
    ret = (cur_belief[0] + c_1 * x + c_2 * y) / (1 + c_1 + c_2)
    # print(f'new belief: {ret}')
    return ret

#coef[0]: y, coef[1]: variance, coef[2]: target
def distance(x, coefs):
    return (new_belief(x, coefs[0], coefs[1]) - coefs[2]) ** 2

def best_combination(points, weighted_pos_arr, target, num_points):
    global cur_belief
    distance = 100
    index = -1
    best_v = np.array([-1.0, 0.0])
    combs = [list(comb) for comb in combinations(points, num_points)]

    for i in range(len(combs)):
        u_arr = [np.array([-1.0, 0.0]) for _ in range(num_points)]
        v = np.array([-1.0, 0.0])
        cur_ds = [np.linalg.norm(point - cur_belief) for point in combs[i]]
        fcts = [g_fct(cur_d, 0, 20) for cur_d in cur_ds]
        cur_ds = [cur_ds[i] * fcts[i] for i in range(len(cur_ds))]

        for j in range(len(combs[i])):
            if(combs[i][j][0] > cur_belief[0]):
                u_arr[j][0] = cur_belief[0] + cur_ds[j]
            else:
                u_arr[j][0] = cur_belief[0] - cur_ds[j]
        
        if len(weighted_pos_arr) == 0:
            v = (sum(u_arr) + cur_belief) / (len(u_arr) + 1)
        else:
            v = (sum(u_arr) + sum(weighted_pos_arr) + cur_belief) / (len(u_arr) * 2 + 1)
        
        if(np.linalg.norm(v - target) < distance):
            index = i
            distance = np.linalg.norm(v - target)
            wpts_arr =  u_arr
            best_v = v
    cur_belief = best_v
    best_comb = combs[index]
    return best_comb, wpts_arr

def best_points_dis(points, x, y, target, variance):
    coefs = [y[0], variance, target[0]]
    x0=x[0]
    result = opt.minimize(distance, x0, args=coefs)
    print(result.message)
    x0 = np.array([result.x[0], 0.0])
    ds = np.linalg.norm(points - x0, axis=1)
    index = np.argsort(ds)[0]
    return points[index], index

def best_points_cont(x, y, target, variance):
    coefs = [y[0], variance, target[0]]
    x0=x[0]
    result = opt.minimize(distance, x0, args=coefs)
    print(result.message)
    x0 = np.array([result.x[0], 0.0])
    return x0

def big_endian_dis(points, x, y, variance):
    global big_target
    print('::::::::::::::::::::::BIG::::::::::::::::::::::::')
    return best_points_dis(points, x, y, big_target, variance)
    # return best_combination(points, weighted_pos, big_target, num_points)

def small_endian_dis(points, x, y, variance):
    global small_target
    print(':::::::::::::::::::::::SMALL:::::::::::::::::::::::')
    return best_points_dis(points, x, y, small_target, variance)
    # return best_combination(points, weighted_pos, small_target, num_points)

def big_endian_cont(x, y, variance):
    global big_target
    print(':::::::::::::::::::::::BIG:::::::::::::::::::::::')
    return best_points_cont(x, y, big_target, variance)
    # return best_combination(points, weighted_pos, small_target, num_points)

def small_endian_cont(x, y, variance):
    global small_target
    print(':::::::::::::::::::::::SMALL:::::::::::::::::::::::')
    return best_points_cont(x, y, small_target, variance)
    # return best_combination(points, weighted_pos, small_target, num_points)

def big_endian2(points, weighted_pos, num_points):
    global big_target
    print('::::::::::::::::::::::BIG::::::::::::::::::::::::')
    return best_combination(points, weighted_pos, big_target, num_points)



def small_endian2(points, weighted_pos, num_points):
    global small_target
    print(':::::::::::::::::::::::SMALL:::::::::::::::::::::::')
    return best_combination(points, weighted_pos, small_target, num_points)

def sim_discrete(points, v):
    cur_big = copy.deepcopy(big_target)
    cur_small = copy.deepcopy(small_target)
    # cur_big = np.array([0.0, 0.0])
    # cur_small = np.array([0.0, 0.0])
    big_i = -1
    small_i = -1
    print('SIM STARTS')
    for i in range(10):
        last_big_i = big_i
        last_small_i = small_i
        print(f'------------------------ITERATION {i}---------------------------')
        cur_big, big_i = big_endian_dis(points, cur_big, cur_small, v)
        # cur_big, weighted_pos_b = big_endian(points, weighted_pos_s, num_pts)
        cur_small, small_i = small_endian_dis(points, cur_small, cur_big, v)
        n_b = new_belief(cur_big[0], cur_small[0], v)
        # cur_small, weighted_pos_s = small_endian(points, weighted_pos_b, num_pts) 
        print(f'belief: {n_b}, big: {cur_big[0]}, big_i: {big_i}, small: {cur_small[0]}, small_i: {small_i}')
        #print(f'small: {cur_small}')
        print('------------------------ITERATION ENDS---------------------------')
        if (i>0 and last_big_i == big_i and last_small_i == small_i):
            print(f'sim converges at ITERATION {i-1}')
            break 
    print('SIM ENDS')

def sim_cont(v):
    cur_big = copy.deepcopy(big_target)
    cur_small = copy.deepcopy(small_target)
    # cur_big = np.array([60.0, 0.0]) 
    # cur_small = np.array([39.99999, 0.0])
    # cur_big = np.array([0.0, 0.0])
    # cur_small = np.array([0.0, 0.0])
    print('SIM STARTS')
    for i in range(10):
        last_big = cur_big
        last_small = cur_small
        print(f'------------------------ITERATION {i}---------------------------')
        cur_big = big_endian_cont(cur_big, cur_small, v)
        cur_small = small_endian_cont(cur_small, cur_big, v)
        n_b = new_belief(cur_big[0], cur_small[0], v)
        print(f'belief: {n_b}, big: {cur_big[0]}, small: {cur_small[0]}')
        print('------------------------ITERATION ENDS---------------------------')
        if (i>0 and (np.linalg.norm(cur_big-last_big) <= 0.001) and (np.linalg.norm(cur_small-last_small) <= 0.001) ):
            print(f'sim converges at ITERATION {i-1}')
            break 
    print('SIM ENDS')

def sim2(points):
    print('SIM Starts')
    num_pts = 3
    big_indices = []
    small_indices = []
    weighted_pos_s = []
    for i in range(10):
        if i > 0:
            last_big = big_indices[-1]
            last_small = small_indices[-1]

        print(f'------------------------ITERATION {i}---------------------------')
        cur_big, weighted_pos_b = big_endian2(points, weighted_pos_s, num_pts)
        bi = get_indices(cur_big, points)
        
        print(f'big: {cur_big} weighted big: {weighted_pos_b} current: {cur_belief}')
        cur_small, weighted_pos_s = small_endian2(points, weighted_pos_b, num_pts) 
        si = get_indices(cur_small, points)  
        
        print(f'small: {cur_small} weighted small: {weighted_pos_s} current: {cur_belief}')
        #print(f'small: {cur_small}')
        print('------------------------ITERATION ENDS---------------------------')

        if i > 0:
            if  bi == last_big and si == last_small:
                print(f'CONVERGES at iteration {i-1}')
                print(f'final big: {big_indices}, final small: {small_indices}, final belief: {cur_belief}')
                break 
        
        big_indices.append(bi)
        small_indices.append(si)

def main():
    points = getdata('best_response/data3.txt')
    # plotdata(points)

    # sim_discrete(points, 10)
    sim_cont(10)
    

    # weights =[g_fct(np.linalg.norm(point-cur_belief),0,20) for point in points]
    # for i in range(len(points)):
    #     print(f'point: {points[i][0]}, weight: {weights[i]}')
    # sim2(points)


    
if __name__ == "__main__":
    main()