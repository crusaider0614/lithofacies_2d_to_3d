import numpy as np
import os
from utils.data import show_2d_array, read_binary, get_project_root


a = read_binary(os.path.join("data", "southsea", "3dn.bin"), 2001)
a = np.reshape(a, (2001, 1200, 2667))
# print(a.shape)
# print(a.max(), a.min())
a = a / a.max() * 2
show_2d_array(a[:, 1000])
exit()
a = a / a.max()
print(a.shape)
a = np.transpose(a, (0, 2, 1))
show_2d_array(a[:, 1000])

a = np.load(os.path.join("data", "southsea", "3dn_process.npy"))
print(a.shape)
show_2d_array(a[:, 1000])