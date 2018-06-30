# -*- coding: utf-8 -*-
"""
Created on Tue May 29 22:58:45 2018

@author: front
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#
##from ..utils.data_utils import get_file
##import numpy as np
#
#from PIL import Image
# 
#
##import codecs
#
#PhotoPath = Image.open('C:\\Users\\front\\Desktop\\FirstNeuro\\test_Neuro\\nine_test.png')
#PhotoTxtFile = codecs.open(PhotoPath, "r", "ansi")
#PhotoTxt = PhotoTxtFile.read()
#PhotoTxtFile.close()
#PhotoBinary = bin(int.from_bytes(PhotoTxt.encode("ansi"), 'big'))[2:]
#PhotoBinary = PhotoBinary.zfill(8 * ((len(PhotoBinary) + 7) // 8))
#print(PhotoBinary)


 

import numpy as np
from PIL import Image

def main():
#        PathImg = input('Enter the image path (image must be BMP): ')
        img = Image.open('C:\\Users\\front\\Desktop\\FirstNeuro\\test.bmp')
#        img = Image.open(PathImg)
#        print('img = ', img)
        arr = np.asarray(img, dtype='uint8')
        
#        for()
#        PhotoBinary = bin(arr[1,1,1])
        
        
        arr = arr.reshape(1, 784)
        
        arr = arr / 255
        
#        print('arr = ', arr)
        
        return arr 

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    