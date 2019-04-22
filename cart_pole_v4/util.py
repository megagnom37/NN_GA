##########################################################
#                       Utilities                        #
##########################################################
# Author:                                Ivan Skripachev #
# Create date:                                21.04.2019 #
# Organization:                                     ITMO #
# Version:                                         1.0.1 #
##########################################################

import numpy as np
from PIL import Image

def rgb2gray(img_mtrx):
    # Convert rgb image mtrx to grayscale image mtrx
    return np.dot(img_mtrx[:,:,:3], [0.2989, 0.5870, 0.1140])

def resize(img_mtrx, img_size):
    # Resize image matrix to target size
    height, width = img_size
    img = Image.fromarray(np.uint8(img_mtrx))
    img = img.resize((width, height), Image.BICUBIC)
    
    return np.array(img)