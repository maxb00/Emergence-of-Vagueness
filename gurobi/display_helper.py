import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Dict
from matplotlib import colormaps as cmp
from matplotlib import colors as col
import matplotlib.patches as mpatches


def rotate_aboutZ(array : NDArray) -> NDArray:
    # assume: array is 3x3x3
    # array = np.asarray(array)
    new_array = np.zeros_like(array, dtype="<U9")
    sz = array.shape[0] - 1    
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            new_array[y, sz-x, :] = array[x, y, :]

    return new_array


def rotate_aboutY(array : NDArray) -> NDArray:
    new_array = np.zeros_like(array, dtype='<U9')
    sz = array.shape[0] - 1
    for x in range(array.shape[0]):
        for z in range(array.shape[2]):
            new_array[sz-z, :, x] = array[x, :, z]

    return new_array


def rotate_aboutX(array : NDArray) -> NDArray:
    new_array = np.zeros_like(array, dtype='<U9')
    sz = array.shape[0] - 1
    for y in range(array.shape[1]):
        for z in range(array.shape[2]):
            new_array[:, z, sz-y] = array[:, y, z]

    return new_array

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


def show(colors_array : NDArray, legend : list[tuple[str,str]]):
    filled = np.ones(colors_array.shape)

    # upscale the image, leaving gaps
    filled_2 = explode(filled)
    colors_2 = explode(colors_array)

    # shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    # create artists for legend
    patches = []
    for bucket in legend:
        patch = mpatches.Patch(color=bucket[0], label=bucket[1])
        patches.append(patch)

    # handle display
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(x, y, z, filled_2, facecolors=colors_2)
    ax.set_aspect('equal')
    ax.legend(handles=patches, bbox_to_anchor=(0,1))

    plt.show()


def cube_from_lookup(
        lookup : Dict[int, int],
        size : int,
        centers : list[int],
        colormap : str = 'Pastel1') -> NDArray:
    color_lookup = cmp[colormap]
    cube = np.zeros((size,size,size), dtype='<U9')
    legend = []
    visited = set()
    i = 0
    for x in range(size):
        for y in range(size):
            for z in range(size):
                bucket = lookup[i]
                try:
                    bucket_color = color_lookup.colors[bucket]
                except IndexError:
                    raise IndexError(f"{bucket} isn't on the chosen color palatte.")
                hex_color = col.to_hex(bucket_color)
                if bucket not in visited:
                    legend.append((hex_color, f"Bucket {centers[bucket]+1}"))
                    visited.add(bucket)
                cube[x,y,z] = hex_color
                i += 1

    return cube, legend