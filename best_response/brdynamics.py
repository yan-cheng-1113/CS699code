import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
from scipy.stats import norm
import copy
import random
import math
from itertools import permutations

"""
TIE BREAKING BY MOVING TO THE CLOSEST POINT TO THE PREV ITER

"""
# cur_belief = np.array([0.0, 0.0])
bound = [-100.0, 100.0]
pts = np.arange(-50, 50, 0.01)
# targets = np.arange(-40, 40, 20.0).tolist()
targets = [-30, -20, 25, 40]
sig = 1
# m = 0

# plot br sequentially; as a function of iters; end reuslts as a function of params; loss function at the equilibria;
# change cols in the table

# def getdata(fileName):
#     with open(fileName) as f:
#         lines = f.readlines()
#         x = np.empty([len(lines), 2])
#         i = 0
#         for line in lines:
#             coordinates = line.split()
#             x_1 = float(coordinates[0])
#             x_2 = float(coordinates[1])
#             x[i] = np.array([x_1, x_2])
#             i+=1
#     return x

def loss_plt(x_arr, i, n_b_func):
    global targets
    n = len(x_arr)
    pts = [(random.uniform(-100, 110), 0)[0] for _ in range(1000)]
    new_belief = []
    for pt in pts:
        x_arr[i] = pt
        new_belief.append(n_b_func(x_arr[i], x_arr, i))
    # new_belief = [(sum(x_arr)[0] - x_arr[i][0] + pt) / n for pt in pts]
    loss = [(targets[i] - nb) ** 2 for nb in new_belief]
    figure, (ax1, ax2) = plt.subplots(2) 
    ax1.plot(pts, loss, 'ro', markersize = 1)
    ax1.set(ylabel='loss')
    ax2.plot(pts, new_belief, 'o', markersize = 1)
    ax1.set(xticks=np.arange(-100, 110, 10.0))
    ax2.set(xticks=np.arange(-100, 110, 10.0))
    ax2.set(xlabel='point chosen', ylabel='new belief')
    ax1.title.set_text(f'loss of influencer {i}')
    plt.show()

def bf_opt(x_arr, i, target, n_b_func):
    global pts
    arr = copy.deepcopy(x_arr)
    cur_loss = (n_b_func(x_arr[i], x_arr, i) - target)**2
    res = x_arr[i]
    if(cur_loss == 0):
        return res
    min_loss = cur_loss
    distances = np.array([(pt - x_arr[i])**2 for pt in pts])
    d_inds = distances.argsort() # sort indices of points based on their distances to x_i
    # brute force to find the optimal point
    for ind in d_inds:
        arr[i] = pts[ind]
        new_belief = n_b_func(arr[i], arr, i)
        loss = (target - new_belief)**2
        if loss < min_loss - 0.1:
            min_loss = loss
            res = pts[ind]
    return res

"""Given a distance, calculates its weight using Gaussian distribution"""
# def g_fct(x, sd):
#     global cur_belief
#     c = np.exp(-(x - cur_belief[0]) ** 2 / (2 * sd ** 2))
#     return c

# def new_belief_gaussian(x, y, sd1, sd2, d):
#     # x = x - d + 2
#     c_1 = g_fct(x, sd1)
#     c_2 = g_fct(y, sd2) 
#     ret = (cur_belief[0] + c_1 * x + c_2 * y) / (1 + c_1 + c_2)
#     # print(f'new belief: {ret}')
#     return ret

def new_belief_mean(x_i, x_arr, i):
    ret = (sum(x_arr) + x_i - x_arr[i]) / len(x_arr)
    return ret

def new_belief_median(x_i, x_arr, i):
    # concatenated = np.concatenate(x_arr)
    return np.median(x_arr)

def g_fct(x_i, m_i):
    global sig
    temp_sig = 0.0001
    c = np.exp(-(x_i - m_i) ** 2 / (2 * (temp_sig ** 2)))
    return c

def new_belief_mean_gaussian(x_i, x_arr, i):
    global targets
    m_i = (sum(x_arr) - x_arr[i]) / (len(x_arr) - 1) # mean excluding i
    c = g_fct(x_i, m_i)
    if c < 0.7:
        return m_i
    else:
        return (sum(x_arr) + x_i - x_arr[i]) / len(x_arr)
    
def new_belief_mean_interval(x_i, x_arr, i):
    global targets, sig
    m_i = (sum(x_arr) - x_arr[i]) / (len(x_arr) - 1) # mean excluding i
    if m_i - sig < x_i < m_i + sig:
        return (sum(x_arr) + x_i - x_arr[i]) / len(x_arr) 
    else:
        return m_i

def new_belief_mean_w_penalty(x_i, x_arr, i):
    global targets
    tot = sum(x_arr) + x_i - x_arr[i]
    ret = (tot) / len(x_arr)

    cheating_distances = []
    # find cheaters
    for j in range(len(x_arr)):
        m_temp = (sum(x_arr) - x_arr[j]) / (len(x_arr) - 1) if j == i else (tot - x_arr[j]) / (len(x_arr) - 1) 
        if ((x_arr[j] - m_temp) * (targets[j] - m_temp)) > 0:
            cheating_distances.append((x_arr[j] - m_temp))

    # penalize the biggest cheater
    eps = 1.5 / len(x_arr) # make eps a list
    if len(cheating_distances) > 0:
       max_cheating = max(cheating_distances, key=abs)
       ret -= eps * math.copysign(1, max_cheating)*abs(max_cheating)
    return ret

def distance(x_i, coefs):
    # return (new_belief_median(x_i, coefs[0], coefs[1]) - coefs[2]) ** 2
    # return (new_belief_mean_gaussian(x_i, coefs[0], coefs[1]) - coefs[2]) ** 2
    # return (new_belief_mean_w_penalty(x_i, coefs[0], coefs[1]) - coefs[2]) ** 2
    # return (new_belief_mean(x_i, coefs[0], coefs[1]) - coefs[2]) ** 2
    
    pass

def best_points(func, x_i, coefs, nb_func):
    global bound
    # x0 = x_i
    # result = opt.minimize(func, x0, args=coefs)
    # x0 = result.x[0]
    # if x0 < bound[0]:
    #     x0 = bound[0]
    # if x0 > bound[1]:
    #     x0 = bound[1]
    # return x0

    return bf_opt(coefs[0], coefs[1], coefs[2], nb_func)
    
def sim(x_arr,n, func):
    global targets
    print('SIM STARTS')
    print('-----------------------\n')
    conv = False
    pts_arr = [[] for _ in range(n)]
    # cur_belief = new_belief_mean(x_arr[0], x_arr, 0)
    # cur_belief = new_belief_median(x_arr[0], x_arr, 0)
    # cur_belief = new_belief_mean_w_penalty(x_arr[0], x_arr, 0)

    for i in range(n):
        pts_arr[i].append(x_arr[i])
    print(x_arr)
    # print(cur_belief)
    print()

    for i in range(100):
        last_x = copy.deepcopy(x_arr)
        print(f'------------------------ITERATION {i+1}---------------------------\n') 
        for j in range(n):
            coefs = [x_arr, j, targets[j]] 
            x_arr[j] = best_points(distance, last_x[j], coefs, func)
            # if j == 2:
            #     loss_plt(last_x, targets, j)
            str_arr = ', '.join(format(pt, '.3f') for pt in x_arr)
            print(f'Points chosen: [{str_arr}]')
            # cur_belief = new_belief_mean(x_arr[0], x_arr, 0) 
            cur_belief = func(x_arr[j], x_arr, j) 
            # print(f'Belief: {cur_belief}\n')
            print(f'Belief: {cur_belief:.3f}\n')
        print('------------------------ITERATION ENDS---------------------------\n')
        if(i > 0):
            if all(abs(last_x[j] - x_arr[j]) <= 0.01 for j in range(n)):
                print(f'sim converges at ITERATION {i}\n')
                print(f'Belief: {cur_belief:.3f}\n')
                conv = True
                break
    
        for j in range(n):
            pts_arr[j].append(x_arr[j])
    
    print('---------------------------------\n')
    for i in range(n):
        str_lst = ', '.join(format(pt, '.3f') for pt in pts_arr[i])
        print(f'Influncer {i}\'s sequence: [{str_lst}]\n')

    iters = [j for j in range(len(pts_arr[0]))] 
    sizes = [5 + 20 * i for i in range(n)]
    sizes.reverse()
    for i in range(n):
        # iters = [j + 0.05*i for j in range(len(pts_arr[0]))]
        plt.scatter(iters, pts_arr[i], s=sizes[i], label = f'influencer{i}')
    plt.xticks(iters, iters)
    plt.ylabel('points chosen')
    plt.xlabel('iterations')
    plt.title('Points Chosen at Each Iteration')
    plt.grid()
    plt.legend()
    plt.show()   

    return cur_belief



#coef[0]: y, coef[1]: variance, coef[2]: target
# def distance_gaussian(x_i, coefs):
#     return (new_belief_gaussian(x_i, coefs[0], coefs[1], coefs[2], coefs[4]) - coefs[3]) ** 2

# def best_points_gaussian(x, y, target, sd1, sd2):
#     d = target[0] - sd1
#     if target[0] < 0:
#         d = target[0] + sd1
#     coefs = [y[0], sd1, sd2, target[0], d]
#     x0=x[0]
#     result = opt.basinhopping(distance_gaussian, x0, niter=1000, minimizer_kwargs={'args':coefs})
#     # result = opt.minimize(distance, x0, args=coefs)
#     print(result.message)
#     x0 = np.array([result.x[0], 0.0])
#     return x0


# def big_endian_cont(x, y, sd1, sd2):
#     global big_target
#     print(':::::::::::::::::::::::BIG:::::::::::::::::::::::')
#     return best_points_gaussian(x, y, big_target, sd1, sd2)
#     # return best_combination(points, weighted_pos, small_target, num_points)

# def small_endian_cont(x, y, sd1, sd2):
#     global small_target
#     print(':::::::::::::::::::::::SMALL:::::::::::::::::::::::')
#     return best_points_gaussian(x, y, small_target, sd1, sd2)
#     # return best_combination(points, weighted_pos, small_target, num_points)

# def sim_cont(sd):
#     cur_big = copy.deepcopy(cur_belief)
#     cur_small = copy.deepcopy(cur_belief)
#     # cur_big = np.array([60.0, 0.0]) 
#     # cur_small = np.array([39.99999, 0.0])
#     # cur_big = np.array([50.0, 0.0])
#     # cur_small = np.array([-50.0, 0.0])
#     big_arr=[cur_big[0]]
#     small_arr=[cur_small[0]]
#     print('SIM STARTS')
#     for i in range(100):
#         last_big = cur_big
#         last_small = cur_small
#         print(f'------------------------ITERATION {i+1}---------------------------')
#         cur_big = big_endian_cont(cur_big, last_small, sd[0], sd[1])
#         # n_b = new_belief(cur_big[0], last_small[0], sd[0], sd[1])
#         # print(f'belief: {n_b}, big: {cur_big[0]}, small: {last_small[0]}')
#         print(f'big: {cur_big[0]}, small: {cur_small[0]}')
#         cur_small = small_endian_cont(cur_small, last_big, sd[1], sd[0])
#         # n_b = new_belief(last_big[0], cur_small[0], sd[0], sd[1])
#         # print(f'belief: {n_b}, big: {last_big[0]}, small: {cur_small[0]}')
#         print(f'big: {cur_big[0]}, small: {cur_small[0]}')
#         print('------------------------ITERATION ENDS---------------------------')
#         # n_b = new_belief(cur_big[0], cur_small[0], sd[0], sd[1]) 
#         # print(f'belief: {n_b}')
#         if (i>0 and (np.linalg.norm(cur_big-last_big) <= 0.001) and (np.linalg.norm(cur_small-last_small) <= 0.001)):
#             print(f'sim converges at ITERATION {i}')
#             # n_b = new_belief(cur_big[0], cur_small[0], sd[0], sd[1]) 
#             # print(f'belief: {n_b}, big: {cur_big[0]}, small: {cur_small[0]}')
#             print(f'big: {cur_big[0]}, small: {cur_small[0]}')
#             break 
#         big_arr.append(cur_big[0])
#         small_arr.append(cur_small[0])
#     print('SIM ENDS')

#     iters = [i for i in range(len(big_arr))]
#     y = [0 for _ in range(len(big_arr))]
#     s_arr=np.arange(10, 10 + 10*len(big_arr),10)

#     # plt.scatter(big_arr, y, c = 'blue', s = s_arr, label='big')
#     # plt.scatter(small_arr, y, c = 'red', s = s_arr, label='small')

#     plt.scatter(big_arr, iters, c = 'blue', label='big')
#     plt.scatter(small_arr, iters, c = 'red', s = 20, label='small')
#     plt.yticks(iters, iters)
#     plt.xlabel('points chosen')
#     plt.ylabel('iteration')
#     plt.title('Points Chosen at Each Iteration')
#     plt.grid()
#     plt.legend()
#     plt.show()

def bne_check(x_arr, i, sigma, n, nb_func):
    global pts, targets
    print(f'Point assigned: {x_arr[i]:.3f}')

    z_other = []
    min_loss = -1
    num = 10 # number of points distributed around x_i
    min_ind = -1
    # distribute points for other agents
    for k in range(n - 1):
        z_other.append(np.random.normal(x_arr[i], 1, num).tolist())
    
    distances = np.array([(pt - x_arr[i])**2 for pt in pts])
    d_inds = distances.argsort()
    for ind in d_inds:
        cumu_loss = 0
        for j in range(num):
            temp_arr = []
            pdf_product = 1
            for k in range(n - 1):
                pdf_product = pdf_product * norm.pdf(z_other[k][j], x_arr[i], sigma)
            temp_arr.extend([z_other[y][j] for y in range(n - 1)])
            temp_arr.insert(i, pts[ind])
            cumu_loss += pdf_product * (nb_func(temp_arr[i], temp_arr, i) - targets[i]) ** 2
        expect_loss = cumu_loss
        # expect_loss = cumu_loss / num
        if min_loss == -1 or expect_loss < min_loss - 0.1:
            min_loss = expect_loss
            min_ind = ind
    print(f'BNE point: {pts[min_ind]:.3f}')

def discrete_bne(x_arr, i, delta, n, nb_func):
    global pts, targets
    cur_loss = (nb_func(x_arr[i], x_arr, i) - targets[i]) ** 2 
    print(f'Point assigned to player {i}: {x_arr[i]:.3f}')

    z_arr = [x_arr[i] - 2 * delta, x_arr[i] - delta, x_arr[i], x_arr[i] + delta, x_arr[i] + 2 * delta]
    pr_arr = [0.1, 0.2, 0.4, 0.2, 0.1]
    ind_combs = list(permutations([0, 1, 2, 3, 4], n-1))

    min_loss = -1
    min_ind = -1
    distances = np.array([(pt - x_arr[i])**2 for pt in pts])
    d_inds = distances.argsort() 

    for ind in d_inds:
       expect_loss = 0
       for comb in ind_combs:
           pr_multiplier = math.prod([pr_arr[comb[j]] for j in range(len(comb))])
           temp_arr = [z_arr[comb[j]] for j in range(len(comb))]
           temp_arr.insert(i, pts[ind])
           expect_loss += pr_multiplier * ((nb_func(temp_arr[i], temp_arr, i) - targets[i]) ** 2) 
       if min_loss == -1 or expect_loss < min_loss - 0.1:
            min_loss = expect_loss
            min_ind = ind 
    print(f'BNE point for player {i}: {pts[min_ind]:.3f}')

def main():
    global targets,sig
    x_arr = np.random.normal(1, 1, 4).tolist()
    # sim_discrete(points, 10)
    # print(new_belief_mean_w_penalty(-60.0, x_arr, 0))
    # loss_plt(x_arr, 0, new_belief_mean_w_penalty)
    
    n = len(x_arr)
    for i in range(n):
        discrete_bne(x_arr, i, 0.3, n, new_belief_mean_w_penalty)
        print(f'Target of player {i}: {targets[i]}')
        print()
    # bne_check(x_arr, 0, 1, n, new_belief_mean_w_penalty)
    # for i in range(10):
    #     x_arr = np.random.normal(1, 1, 2).tolist()
    #     n = 2
    #     discrete_bne(x_arr, 0, 0.2, n, new_belief_mean_w_penalty)
    #     print()
    # sim(x_arr, n, new_belief_mean_w_penalty)
    
    # pairs = list(zip(x_arr, targets))
    # perms = permutations(pairs)
    # believes = []
    # for perm in perms:
    #     x_perm, targets_perm = zip(*perm)
    #     x_list = list(x_perm)
    #     targets = list(targets_perm)
    #     believes.append(sim(x_list, n))

    # print(believes)
    
    # plt.hist(believes)
    # plt.title('Hist of Believes')
    # plt.xlabel('Belief')
    # plt.ylabel('Frequency')
    # plt.show()
    
    
    # sd=[10,10]
    
    # sim_cont(sd)
    

    # weights =[g_fct(np.linalg.norm(point-cur_belief),0,20) for point in points]
    # for i in range(len(points)):
    #     print(f'point: {points[i][0]}, weight: {weights[i]}')
    # sim2(points)


    
if __name__ == "__main__":
    main()