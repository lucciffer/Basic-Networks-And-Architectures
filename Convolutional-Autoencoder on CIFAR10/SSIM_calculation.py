# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as ssim
import argparse
import imutils
import cv2

imageA = cv2.imread('original.png')
imageB = cv2.imread('reconstructed.png')

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

print("SSIM: {}".format(score))
# print("Diff: {}".format(diff))
