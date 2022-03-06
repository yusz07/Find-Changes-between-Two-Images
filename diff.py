import cv2
import numpy as np
from cv2.xfeatures2d import matchGMS

# Locations of taking two images have slight differences in my dataset
# Perform SIFT+BF feature matching & homography transformation to align same parts of two images

def filter(image):  #No significant effects on my dataset
    #image = cv2.medianBlur(src=image, ksize=5)
    image = cv2.GaussianBlur(image, (5,5), 0)
    #image = cv2.blur(image,(3,3))
    return image

def cut_image(imageA,imageB):
    h1 = imageA.shape[0]
    h2 = imageB.shape[0]
    imageA = imageA[100:(h1 - 100), :]
    imageB = imageB[100:(h2 - 100), :]
    return imageA,imageB


def matchimage(img1,img2):
    pt1 = []
    pt2 = []
    sift = cv2.xfeatures2d.SIFT_create()  #SIFT (SURF is also efficient)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)  #BF matching
    #matches1 = bf.knnMatch(des1, des2,2)
    #matches = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches, withScale=True, withRotation=True,
    #                        thresholdFactor=3)  #GMS, the subtract result became worse when added on my dataset
    if len(matches) != 0:
        for i, m in enumerate(matches):
            pt1.append(kp1[m.queryIdx].pt)
            pt2.append(kp2[m.trainIdx].pt)
        pt1 = np.float32(pt1)
        pt2 = np.float32(pt2)  #match point pairs
        H, status = cv2.findHomography(pt1, pt2, cv2.RANSAC, 6.0) #Homomorphic transformation
        return matches, H, status

    return None

def subtract(imageA, imageB):
    if len(imageA.shape) ==3:
        imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    if len (imageB.shape) ==3:
        imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # #将两幅图像对齐后相减
    # Align the corresponding positions of the two images and subtract

    matches, H, status = matchimage(imageA,imageB)
    if matches is None and H is None:
        return None

    resultA = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

    hb = imageB.shape[0]
    wb = imageB.shape[1]
    resultA = resultA[:hb, :wb]    #Only keep the area corresponding to imageB / 只保留和图B对应的区域
    #cv2.imwrite('./re/resultA.jpg', resultA)
    diff = cv2.absdiff(resultA, imageB)
    #diff = cv2.subtract(resultA, imageB)
    return diff

if __name__ == '__main__':

  imageA = cv2.imread('./before.jpg' ,0)
  imageB = cv2.imread('./after.jpg' ,0)
  imageA, imageB = cut_image(imageA, imageB)
  diff = subtract(imageA,imageB)
  cv2.imwrite('./re/diff.jpg', diff)
