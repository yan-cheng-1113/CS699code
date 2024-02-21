import random
import matplotlib.pyplot as plt
import numpy as np


def loss_calc(x_arr, y, vb, v, tar):
    c_1 = [np.exp(-(x - vb) ** 2 / (2 * v ** 2)) for x in x_arr]
    c_2 = np.exp(-(y - vb) ** 2 / (2 * v ** 2))
    nbs = [(vb + c_1[i] * x_arr[i] + c_2 * y) / (1 + c_1[i] + c_2) for i in range(len(x_arr))]
    return [(nb - tar)**2 for nb in nbs]

def main():
    pt = [(random.uniform(-100, 100), 0)[0] for _ in range(1000)]
    loss = loss_calc(pt, 0.0, 30.0, 10, 55.0)
    plt.plot(pt, loss, 'o')
    plt.xlabel('point chosen')
    plt.ylabel('loss')
    plt.show()
if __name__ == "__main__":
    main()