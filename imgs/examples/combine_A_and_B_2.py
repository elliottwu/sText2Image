from pdb import set_trace as st
import os
import numpy as np
import cv2

fold = 'mismatch/gender/m2f'

l = ['2_22', '2_30', '3_14', '4_17', '5_24', '8_54', '8_60', '9_09', '9_13', '9_64', '10_27', '14_60', '15_13', '15_21', '15_43', '16_58']

num_imgs = 16
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

