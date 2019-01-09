import numpy as np
from scipy.misc import imsave
from PIL import Image
from skimage.util.shape import view_as_windows as pic2windows

window_shape = (16, 16)

np.set_printoptions(threshold=np.inf)
file = open("dictionary.npy",'rb')
phi = np.load(file)
print(phi.shape)
dictionary = np.load(file)
print(dictionary.shape)
pad = np.load(file)
file.close()

#from file load data generated in main fun

new = pad
#new = np.mat(phi)*np.mat(dictionary)
new = np.array(new)
patches = pic2windows(new,window_shape)
imsave('new'+'.jpg', Image.fromarray(np.uint8(np.abs(new))))

file = open('dictionary.txt','w')
file.write(str(phi))
file.write('\n\n')
file.write(str(dictionary))
file.write("---------------------------------patches---------------------------------")
for i in patches:
    #file.write(str(i))
    print("-----------------")
file.close()
