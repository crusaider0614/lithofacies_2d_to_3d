import matplotlib.pyplot as plt
import numpy as np


def read_binary(data_path, n1):
    with open(data_path, "rb") as file:
        field = np.fromfile(file, dtype=np.float32)
    data = field.reshape(-1, n1)
    data = data.transpose()
    return data


def show_2d_array(data, scale=100, cmap="seismic", vmin=-1.0, vmax=1.0, is_show=True, **kwargs):
    fig = plt.figure()
    fig.set_size_inches((data.shape[1] / scale, data.shape[0] / scale))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    if is_show:
        plt.show()