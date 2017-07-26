from pdb import set_trace as st
import os
import numpy as np
import cv2

fold = 'handdraw'

l = ['2_60', '3_54', '5_29', '5_41', '5_62', '8_53', '9_28', '9_39', '9_40', '11_08', '11_20', '11_30', '13_05', '13_17', '15_44', '17_07', '17_51']

num_imgs = len(l)
for n in range(num_imgs):
    print('['+str(n)+'/'+str(num_imgs)+']')
    name = '{:02d}'.format(n+1)
    path_A = os.path.join(fold, l[n]+'_A.png')
    path_B = os.path.join(fold, l[n]+'_B.png')
    #path_C = os.path.join(fold, l[n]+'_C.png')
    if os.path.isfile(path_A) and os.path.isfile(path_B):
        path_ABC = os.path.join(fold, name+'.png')
        im_A = cv2.imread(path_A)
        im_B = cv2.imread(path_B)
        #im_C = cv2.imread(path_C)
        h, w, c = im_A.shape
        gap = np.ones((h, 5, c))*255
        im_ABC = np.concatenate([im_A, gap, im_B], 1)
        cv2.imwrite(path_ABC, im_ABC)

