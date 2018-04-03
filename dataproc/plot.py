import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib import pylab


def plot_and_show(data, x, y, labels=None, axis=None, style='.-', size=None):
    handles = [plt.plot(line[x], line[y], style)[0] for line in data]
    if axis:
        plt.axis(axis)
    if labels:
        plt.legend(handles, labels,
                   bbox_to_anchor=(0., 1.02, 1., .102), loc=10,
                   ncol=3, borderaxespad=0., frameon=False, fontsize=8)
    fig = plt.gcf()
    if size:
        fig.set_size_inches(size)
    plt.show()
