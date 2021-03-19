#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_data(file):
    data = np.loadtxt(file)
    return data

def plot(title):
    if 'czech' in title.lower():
        data0 = get_data('./prc/cs0')
        data1 = get_data('./prc/cs1')
    else:
        data0 = get_data('./prc/en0')
        data1 = get_data('./prc/en1')

    x = np.arange(0,1.001,0.1) 

    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.plot(x, data0, marker='o', label='baseline')
    plt.plot(x, data1, marker='^', label='constrained')
    plt.legend(loc="upper right")
    axes = plt.gca()
    # axes.set_xlim([xmin,xmax])
    axes.set_ylim([0,1])
    plt.show()


if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('PDF')
    if sys.argv[1] == 'cs':
        plot('czech')
        plt.savefig("cs.pdf", bbox_inches='tight')
    else:
        plot('english')
        plt.savefig("en.pdf", bbox_inches='tight')
