import matplotlib.pyplot as plt
import numpy as np
import math

cur_belief = np.array([45, 0])
big_target = np.array([25, 0])
small_target = np.array([75, 0])

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

def plotdata(points):
   plt.scatter(points[:,0], points[:,1], label='sources')
   plt.scatter(big_target[0], big_target[1], c='purple', s=100, label='big_target')
   plt.scatter(small_target[0], small_target[1], c='orange', s=100, label="small_target")
   plt.scatter(cur_belief[0], cur_belief[1], c='green', s=100, label='current_belief')
   plt.title('Data Points Distribution')
   plt.legend()
   plt.show() 
   return

def g_fct(x, mean, variance):
    pi = math.pi
    ret = np.exp(-(x - mean)**2 / (2 * variance**2))
    return ret

def best_points(points, weighted_pos, target, num_points):
    global cur_belief
    distance = 100
    index = -1
    best_v = np.array([-1.0, 0.0])
    weighted_p = np.array([-1.0, 0.0])
    for k in range(num_points):
        prev_i = index
        for j in range(len(points)):

            if(j == prev_i):
                continue
            
            u = np.array([-1.0, 0.0]) 
            cur_d = np.linalg.norm(points[j] - cur_belief)
            #print(f'cur_d: {cur_d}')
            fct = g_fct(cur_d, 0, 20)
            #print(f'fct: {big_fct}')
            cur_d = cur_d * fct
            print(f'{j+1}--distantce to current belief: {cur_d}')
            if(points[j][0] > cur_belief[0]):
                #print(f'rcur_point: {points[j]}')
                u[0] = cur_belief[0] + cur_d
                #print(u[0])
            else:
                #print(f'lcur_point: {points[j]}')
                u[0] = cur_belief[0] - cur_d
                #print(u[0]) 
            if(weighted_pos[0] == -1):
                v = (u + cur_belief) / 2
            else: 
                v = (u + cur_belief + weighted_pos) / 3
            if(np.linalg.norm(v - target) < distance):
                index = j 
                distance = np.linalg.norm(v - target)
                weighted_p = u
                best_v = v
                print(f'wp: {weighted_p} | pt: {points[index]}') 
            print('----------------------------------------')
    cur = points[index]
    cur_belief = best_v
        #print(f'big: {cur_big}')
    return cur, weighted_p 

def big_endian(points, weighted_pos, num_points):
    global big_target
    print('::::::::::::::::::::::BIG::::::::::::::::::::::::')
    return best_points(points, weighted_pos, big_target, num_points)

def small_endian(points, weighted_pos, num_points):
    global small_target
    print(':::::::::::::::::::::::SMALL:::::::::::::::::::::::')
    return best_points(points, weighted_pos, small_target, num_points)

def sim(points):
    cur_big = np.array([-1, 0])
    cur_small = np.array([-1, 0])
    weighted_pos_s = np.array([-1, 0]) 
    print('SIM STARTS')
    for i in range(8):
        last_cur_big = cur_big
        last_cur_small = cur_small
        print(f'------------------------ITERATION {i}---------------------------')
        cur_big, weighted_pos_b = big_endian(points, weighted_pos_s, 1)
        print(f'big: {cur_big} weighted big: {weighted_pos_b} current: {cur_belief}')
        cur_small, weighted_pos_s = small_endian(points, weighted_pos_b, 1)
        print(f'small: {cur_small} weighted small: {weighted_pos_s} current: {cur_belief}')
        #print(f'small: {cur_small}')
        print('------------------------ITERATION ENDS---------------------------') 
        if (last_cur_big[0] == cur_big[0] and last_cur_small[0] == cur_small[0]):
            print(f'CONVERGES at the {i}th iteration')
            print(f'final big: {cur_big}, final small: {cur_small}, final belief: {cur_belief}')
            break
        # print(f'big: {cur_big}')
        # print(f'small: {cur_small}') 
        # mid = (cur_big + cur_small) / 2
        #print(f'mid: {mid}')
        # plt.scatter(big_target[0], big_target[1], color='red', s=80)
        # plt.scatter(small_target[0], small_target[1], color='purple', s=80)
        # plt.scatter(points[:,0], points[:,1])
        # plt.scatter(cur_big[0], cur_big[1], color = 'red')
        # print(cur_big[0])
        # plt.scatter(cur_small[0], cur_small[1], color = 'purple')
        # plt.scatter(mid[0], mid[1], color = 'green')
        # print(cur_big)
        # plt.plot(cur_big, cur_small, '-o')
        # plt.show()
    print('SIM ENDS')
def main():
    points = getdata('best_response/data2.txt')
    plotdata(points)
    sim(points)


    
if __name__ == "__main__":
    main()