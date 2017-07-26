from pdb import set_trace as st
import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold', dest='fold', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

num_imgs = 8
for n in range(num_imgs):
    print('['+str(n)+'/'+str(num_imgs)+']')
    name = '{:02d}'.format(n+1)
    path_A = os.path.join(args.fold, name+'_A.png')
    path_B = os.path.join(args.fold, name+'_B.png')
    path_C = os.path.join(args.fold, name+'_C.png')
    if os.path.isfile(path_A) and os.path.isfile(path_B):
        path_ABC = os.path.join(args.fold, name+'.png')
        im_A = cv2.imread(path_A)
        im_B = cv2.imread(path_B)
        im_C = cv2.imread(path_C)
        im_ABC = np.concatenate([im_A, im_B, im_C], 1)
        cv2.imwrite(path_ABC, im_ABC)

