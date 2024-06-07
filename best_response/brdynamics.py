import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import copy
import random

"""
TIE BREAKING BY MOVING TO THE CLOSEST POINT TO THE PREV ITER

"""
# cur_belief = np.array([0.0, 0.0])
bound = [-100.0, 100.0]
pts = np.arange(-100, 100, 0.01)

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

def loss_plt(x_arr, targets, i):
    n = len(x_arr)
    pts = [(random.uniform(-100, 110), 0)[0] for _ in range(1000)]
    new_belief = []
    for pt in pts:
        x_arr[i] = pt
        new_belief.append(np.median(x_arr))
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

def bf_opt(x_arr, i, target):
    global pts
    cur_loss = (np.median(x_arr) - target)**2
    min_loss = cur_loss
    res = x_arr[i]
    distance = np.array([(pt - x_arr[i])**2 for pt in pts])
    d_inds = distance.argsort()
    arr = copy.deepcopy(x_arr)

    for ind in d_inds:
        arr[i] = pts[ind]
        new_belief = np.median(arr)
        loss = (target - new_belief)**2
        if loss < min_loss - 0.1:
            min_loss = loss
            res = pts[ind]

    # new_belief = []    
    # for pt in pts:
    #     arr[i] = pt
    #     new_belief.append(np.median(arr))

    # loss = np.array([(target - nb) ** 2 for nb in new_belief])
    # inds = loss.argsort()
    # res = pts[inds[0]] 
    # if(abs(loss[inds[0]] - cur_loss) < 0.0001):
    #     res = x_arr[i]
    # print(res)
    # print(loss[inds[0]])
    return res

"""Given a distance, calculates its weight using Gaussian distribution"""
def g_fct(x, sd):
    global cur_belief
    c = np.exp(-(x - cur_belief[0]) ** 2 / (2 * sd ** 2))
    return c

def new_belief_gaussian(x, y, sd1, sd2, d):
    # x = x - d + 2
    c_1 = g_fct(x, sd1)
    c_2 = g_fct(y, sd2) 
    ret = (cur_belief[0] + c_1 * x + c_2 * y) / (1 + c_1 + c_2)
    # print(f'new belief: {ret}')
    return ret

def new_belief_mean(x_i, x_arr, i):
    ret = (sum(x_arr) + x_i - x_arr[i]) / len(x_arr)
    return ret

def new_belief_median(x_i, x_arr, i):
    # concatenated = np.concatenate(x_arr)
    return np.median(x_arr)

def distance(x_i, coefs):
    # return (new_belief_median(x_i, coefs[0], coefs[1]) - coefs[2]) ** 2
    return (new_belief_mean(x_i, coefs[0], coefs[1]) - coefs[2]) ** 2

def best_points(func, x_i, coefs):
    global bound
    x0 = x_i
    result = opt.minimize(func, x0, args=coefs, method = 'Nelder-Mead')
    # print(result.message)
    x0 = result.x[0]
    if x0 < bound[0]:
        x0 = bound[0]
    if x0 > bound[1]:
        x0 = bound[1]
    return x0
    # return bf_opt(coefs[0], coefs[1], coefs[2])
    
def sim(x_arr, targets, n):
    print('SIM STARTS')
    print(x_arr)
    print('-----------------------\n')
    pts_arr = [[] for _ in range(n)]
    cur_belief = new_belief_mean(x_arr[0], x_arr, 0)
    # cur_belief = new_belief_median(x_arr[0], x_arr, 0)

    for i in range(n):
        pts_arr[i].append(x_arr[i])
    print('-----------------------')
    print(f'Belief: {cur_belief}')

    for i in range(100):
        last_x = copy.deepcopy(x_arr)
        print(f'------------------------ITERATION {i+1}---------------------------') 
        for j in range(n):
            coefs = [x_arr, j, targets[j]] 
            x_arr[j] = best_points(distance, last_x[j], coefs)
            # if j == 2:
            #     loss_plt(last_x, targets, j)
            print(f'{x_arr}\n')
            print(f'Belief: {new_belief_mean(x_arr[0], x_arr, 0)}\n')
        print('------------------------ITERATION ENDS---------------------------\n')
        if(i > 0):
            if all(abs(last_x[j] - x_arr[j]) <= 0.001 for j in range(n)):
                print(f'sim converges at ITERATION {i}\n')
                print(f'Belief: {new_belief_mean(x_arr[0], x_arr, 0)}\n')
                break
        for j in range(n):
            pts_arr[j].append(x_arr[j])
    
    print('---------------------------------\n')
    for i in range(n):
        print(f'Influncer {i}\'s sequence: {pts_arr[i]}')

    iters = [j for j in range(len(pts_arr[0]))] 
    sizes = [5 + 20 * i for i in range(n)]
    sizes.reverse()
    for i in range(n):
        # iters = [j + 0.05*i for j in range(len(pts_arr[0]))]
        plt.scatter(pts_arr[i], iters, s=sizes[i], label = f'influencer{i}')
    plt.yticks(iters, iters)
    plt.xlabel('points chosen')
    plt.ylabel('iterations')
    plt.title('Points Chosen at Each Iteration')
    plt.grid()
    plt.legend()
    plt.show()   




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

def main():
    # points = getdata('best_response/data3.txt')
    # plotdata(pointsi

    # sim_discrete(points, 10)

    targets = np.arange(-50, 50, 20.0).tolist()
    # x_arr = [-50.0, -30.0, -10.0, 10.0, 30.0, 50.0]
    # x_arr = [random.uniform(-100, 110) for _ in range(len(targets))]
    x_arr = copy.deepcopy(targets)

    # loss_plt(x_arr, targets, 2)
    sim(x_arr, targets, len(x_arr))
    # bf_opt(x_arr, 0, targets[0])
    
    # sd=[10,10]
    
    # sim_cont(sd)
    

    # weights =[g_fct(np.linalg.norm(point-cur_belief),0,20) for point in points]
    # for i in range(len(points)):
    #     print(f'point: {points[i][0]}, weight: {weights[i]}')
    # sim2(points)


    
if __name__ == "__main__":
    main()