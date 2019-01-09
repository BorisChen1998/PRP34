import numpy as np
from skimage.util.shape import *
from operator import mul, sub
from math import floor, sqrt, log10
import sys
from scipy.sparse.linalg import svds
from scipy.stats import chi2
from skimage.util import pad
import timeit
from functools import reduce

from Functions import *
from PIL import Image
from scipy.misc import imsave
from scipy.misc import imshow


import matplotlib.pyplot as plt


window_shape = (16,16)
resize_shape = (256, 256)
step = 16
name = r'C:\Users\siwei\Desktop\prp\codeLab\K-SVD-py\timg3.jpg'
original_image = np.asarray(Image.open(name).convert('L').resize(resize_shape))

img = original_image

patches = view_as_windows(img, window_shape, step=step)
# loaded from from skimage.util.shape
# spilt pic into patche
cond_patches = np.zeros((reduce(mul, patches.shape[2:4]), reduce(mul, patches.shape[0:2])))
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        print(j+patches.shape[1]*i)
        cond_patches[:, j+patches.shape[1]*i] = np.concatenate(patches[i, j], axis=0)



plt.imshow(Image.fromarray(np.uint8(cond_patches)))
plt.show()
print(cond_patches.shape)


# patch spilting?
