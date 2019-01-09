import numpy as np
from os import *
from PIL import Image
import matplotlib.pyplot as plt
from operator import mul, sub



def image_reconstruction_windows(mat_shape, patch_mat, patch_sizes, step):
    img_out = np.zeros(mat_shape)
    for l in range(patch_mat.shape[1]):
        i, j = divmod(l, patch_sizes[1])
        temp_patch = patch_mat[:, l].reshape((patch_sizes[2], patch_sizes[3]))
        img_out[i*step:(i+1)*step, j*step:(j+1)*step] = temp_patch[:step, :step].astype(int)
    return img_out

#settings
chdir(r"C:\Users\siwei\OneDrive\prp\codeLab\K-SVD-py")
shape_factor = 10
window_shape = (shape_factor , shape_factor)    # Patches' shape
sigma = 10                 # Noise standard dev.
resize_shape = (window_shape[0]*window_shape[0], window_shape[0]*window_shape[0])  # Resized image's shape
step = int(shape_factor/5)                # Patches' step
ratio = 1             # Ratio for the dictionary (training set).
ksvd_iter = 5              # Number of iterations for the K-SVD

#load dictionary
dictionary = np.load(r".\TestTemp\dictionary.npy")
phi = np.load(r".\TestTemp\phi.npy")
plt.imshow(Image.fromarray(np.uint8(dictionary)))
plt.show()
plt.imshow(Image.fromarray(np.uint8(phi)))
plt.show()

#Reconstruct
ans = phi.dot(dictionary)
padded_denoised_image = image_reconstruction_windows(((120, 120)),ans,(23, 23, 10, 10),step)
shrunk_0, shrunk_1 = tuple(map(sub,padded_denoised_image.shape, window_shape))
ans = np.abs(padded_denoised_image)[window_shape[0]:shrunk_0, window_shape[1]:shrunk_1]
plt.imshow(Image.fromarray(np.uint8(ans)))
plt.show()

#show
