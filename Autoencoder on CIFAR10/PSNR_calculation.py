from math import log10, sqrt
import cv2
import numpy as np

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20*log10(max_pixel / sqrt(mse))
    return psnr

original = cv2.imread('/home/lucciffer/WORK/AI/CVG/networks/AE-CIFAR10/try1/original.png')
compressed = cv2.imread('/home/lucciffer/WORK/AI/CVG/networks/AE-CIFAR10/try1/reconstructed.png',1)
compressed = cv2.resize(compressed,(714,252))
val = PSNR(original,compressed)
print(f"PSNR Value is {val} dB")



