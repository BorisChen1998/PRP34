from cv2 import bilateralFilter as bilateral
from cv2 import imread, imwrite,imshow
import numpy as np
import os

os.chdir(r'C:\Users\siwei\OneDrive\prp\codeLab\bilateralFilter-py\code')
originImagine = imread('case1.png')
filteredImagine =np.hstack([ bilateral(originImagine, 15, 30, 30),
 bilateral(originImagine, 75, 150, 150),
 bilateral(originImagine, 150, 250, 250)])
imshow("Testing",filteredImagine)
imwrite("before.jpg",np.hstack([originImagine ,originImagine ,originImagine ]))
imwrite("after.jpg", filteredImagine, params=None)
