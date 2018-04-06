"""Plotting tools."""
import matplotlib.pyplot as plt  # pylint: disable=import-error


def plot_and_show(data, x_column, y_column, *,
                  labels=None, axis=None, style='.-', size=None):
    """Plots and shows data: for use in Jupyter Notebook."""
    handles = [plt.plot(line[x_column], line[y_column], style)[0]
               for line in data]
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
