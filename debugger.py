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

def plot():
    pt = [(random.uniform(-100, 0), 0)[0] for _ in range(1000)]
    y = 50.0
    vb = -70.0
    sd1 = 10
    sd2 = 10
    tar = -50.0
    nb = new_belief(pt, y, vb, sd1, sd2)
    loss = loss_calc(pt, y, vb, sd1, sd2, tar)
    figure, (ax1, ax2) = plt.subplots(2) 
    ax1.plot(pt, loss, 'o')
    ax1.set(ylabel='loss')
    ax2.plot(pt, nb, 'o')
    ax1.set(xticks=np.arange(-50, 50, 10.0))
    ax2.set(xticks=np.arange(-50, 50, 10.0))
    ax2.set(xlabel='point chosen', ylabel='new belief')
    plt.show()

def plot_2():
    pt = [(random.uniform(-30, 30), 0)[0] for _ in range(1000)] 
    y_arr = [-11.25, 5.62]
    vb = 0.0
    sd_arr = [5, 10]
    tars = [50.0, -50.0] 
    nbs = [new_belief(pt, y_arr[i], vb, sd_arr[i], sd_arr[i]) for i in range(len(y_arr))]
    losses = [loss_calc(pt, y_arr[i], vb, sd_arr[i], sd_arr[i], tars[i]) for i in range(len(y_arr))]
    figure, axs = plt.subplots(2, 2)
    axs[0, 0].plot(pt, losses[0], 'o')
    axs[0, 0].set(ylabel='loss')
    axs[0, 0].set_title('Big')
    axs[0, 1].plot(pt, losses[1], 'o')
    axs[0, 1].set_title('Small')
    
    axs[1, 0].plot(pt, nbs[0], 'o')
    axs[1, 0].set(ylabel='new belief')
    axs[1, 1].plot(pt, nbs[1], 'o')

    axs[0,0].set(xticks=np.arange(-30, 30, 10.0))
    axs[0,1].set(xticks=np.arange(-30, 30, 10.0))
    axs[1,0].set(xticks=np.arange(-30, 30, 10.0))
    axs[1,1].set(xticks=np.arange(-30, 30, 10.0))
    plt.show()

def main():
    plot()
    # plot_2()

if __name__ == "__main__":
    main()