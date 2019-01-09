#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 0. libraries -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
from time import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version

from math import *
from Functions import *
from PIL import Image
from scipy.misc import imsave
from os import *

import cv2
import math


class Hog_descriptor():
    def __init__(self, img, cell_size=16, bin_size=10):
        self.img = img
        self.img = np.sqrt(img / np.max(img))
        self.img = img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        # assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        # print (cell_gradient_vector.shape)
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        #print (gradient_values_x.shape)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        #print (gradient_values_y.shape)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image




#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 1. Initial settings. -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

# some of the varias in comments  are for further used
chdir(r"C:\Users\siwei\OneDrive\prp\codeLab\K-SVD-py")
# !! change to the adrress of working file
shape_factor = 16 # !! root factor of pic reading
window_shape = (shape_factor , shape_factor)    # !! Patches' shape
#sigma = 10                 # Noise standard dev.
resize_shape = (window_shape[0]*window_shape[0], window_shape[0]*window_shape[0])  # Resized image's shape
#step = int(shape_factor/2)                # Patches' step
#ratio = 1             # Ratio for the dictionary (training set).
#ksvd_iter = 5              # Number of iterations for the K-SVD
dic_num = 1024
#-------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------- 2. Image import. ----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

name = r'\edge.jpg'
read_add = r'.\TestCases'
original_image = np.asarray(Image.open(read_add + name).convert('L').resize(resize_shape))
learning_image = np.asarray(Image.open(read_add + name).convert('L').resize(resize_shape))

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- 3. Image processing. --------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
face = original_image
# make face as the image imported


# Convert from uint8 representation with values between 0 and 255 to
# a floating point representation with values between 0 and 1.
face = face / 255.0


# downsample for higher speed
# make every 4 pixels into one
# 1 1   ->    4
# 1 1
#old_face = face
#face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
#face = face / 4.0
height, width = face.shape



#-------------------# we don't need distort------------------------
# Distort the right half of the image
print('Distorting image...')
distorted = face.copy()
#.copy of np is a deep copy
# distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)
#==================================================================


#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 4. generate Dictionary. -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
# Extract all patches from theimage
print('Extracting reference patches...')
t0 = time()
patch_size = (shape_factor, shape_factor)
data = extract_patches_2d(distorted[:, :], patch_size)
#data = extract_patches_2d(distorted[:, :width // 2], patch_size)
data = data.reshape(data.shape[0], -1)
print(data.shape)

data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
print('done in %.2fs.' % (time() - t0))
print('Learning the dictionary...')

t0 = time()
dico = MiniBatchDictionaryLearning(n_components=dic_num, alpha=1, n_iter=500)
V = dico.fit(data).components_
dt = time() - t0
print('done in %.2fs.' % dt)
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:dic_num]):
    plt.subplot( ceil(dic_num**0.5), ceil(dic_num**0.5), i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())


plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:dic_num]):
    plt.subplot( ceil(dic_num**0.5), ceil(dic_num**0.5), i + 1)
    dic_ele = comp.reshape(patch_size)
    hog = Hog_descriptor(dic_ele, cell_size=4, bin_size=20)
    vector, image = hog.extract()
    plt.imshow(image, cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
# cell_size, the size of the detection window
# bin_size, the number of areas divided in the 360 degree




plt.suptitle('Dictionary learned from face patches\n' +
             'Train time %.1fs on %d patches' % (dt, len(data)),
             fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)#left, right, bottom, top, wspace, hspace

#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 4. generate phi and reconstruct. -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
transform_algorithms =('Orthogonal Matching Pursuit\n1 atom', 'omp',{'transform_n_nonzero_coefs': 1})
data = extract_patches_2d(face, patch_size)
data = data.reshape(data.shape[0], -1)
print(data.shape)
code = dico.transform(data)
patches = np.dot(code, V)
patches = patches.reshape(len(data), *patch_size)
reconstructions= face.copy()
reconstructions[:,:] = reconstruct_from_patches_2d(patches, (height, width))
reconstructions = reconstructions*255.0
plt.figure('original_image & reconstructions')
plt.subplot(1,2,1)
plt.imshow(face,cmap=plt.cm.gray)
plt.subplot(1,2,2)
plt.imshow(reconstructions,cmap=plt.cm.gray)
plt.show()
