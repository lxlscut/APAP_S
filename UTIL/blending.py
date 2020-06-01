import numpy as np


def blending_average(my_local, origin_result):
    my_local = my_local.astype(np.float)
    for i in range(my_local.shape[0]):
        for j in range(my_local.shape[1]):
            if np.sum(my_local[i, j, :]) > 0 and np.sum(origin_result[i, j, :]) > 0:
                my_local[i, j, :] = (my_local[i, j, :] + origin_result[i, j, :]) / 2
            elif np.sum(my_local[i, j, :]) > 0 and np.sum(origin_result[i, j, :]) == 0:
                my_local[i, j, :] = my_local[i, j, :]
            else:
                my_local[i, j, :] = origin_result[i, j, :]
    my_local = my_local.astype(np.uint8)
    return my_local
