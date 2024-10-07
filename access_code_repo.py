computer_vision=False
com_path=False
pytorch=False
tensorflow=False

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

sys.path.append(f'UtilityBox')

if tensorflow:
    import utils.tf_gpu_utils as tfgu

if pytorch:
    import utils.torch_gpu_utils as tgu

if computer_vision:
    import utils.image_utils as iu
    import ocv
    import cv2
    import Shapely
    
if com_path:
    from tifffile import imread, imsave
    import pma as pu
    import compath.tissue as tu
    from pma_python import core

import utils.utils as u
import load
import s3

 
'''
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
hierarchy = [[[ 1, -1, -1, -1],  # Contour 0: [Next, Prev, First_Child, Parent]
              [-1, -1, -1,  0],  # Contour 1: [Next, Prev, First_Child, Parent]
              [-1, -1, -1,  0]]] # Contour 2: [Next, Prev, First_Child, Parent]
'''
