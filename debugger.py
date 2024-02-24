import random
import matplotlib.pyplot as plt
import numpy as np

def new_belief(x_arr, y, vb, sd1, sd2):
    c_1 = [np.exp(-(x - vb) ** 2 / (2 * sd1 ** 2)) for x in x_arr]
    c_2 = np.exp(-(y - vb) ** 2 / (2 * sd2 ** 2))
    # print(f'new belief: {ret}')
    return [(vb + c_1[i] * x_arr[i] + c_2 * y) / (1 + c_1[i] + c_2) for i in range(len(x_arr))]

def loss_calc(x_arr, y, vb, sd1, sd2, tar):
    nbs = new_belief(x_arr, y, vb, sd1, sd2)
    return [(nb - tar)**2 for nb in nbs]

def main():
    pt = [(random.uniform(-20, 10), 0)[0] for _ in range(1000)]
    y = 5.625
    vb = 0.0
    sd1 = 10
    sd2 = 5
    tar = -50.0
    nb = new_belief(pt, y, vb, sd1, sd2)
    loss = loss_calc(pt, y, vb, sd1, sd2, tar)
    figure, (ax1, ax2) = plt.subplots(2) 
    ax1.plot(pt, loss, 'o')
    ax1.set(ylabel='loss')
    ax2.plot(pt, nb, 'o')
    ax2.set(xlabel='point chosen', ylabel='new belief')
    plt.show()
if __name__ == "__main__":
    main()