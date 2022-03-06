import cv2
from timeit import time
from detect import find_change_and_detect
from yolo import YOLO

if __name__ == '__main__':
    find = find_change_and_detect(threshold_for_binary=20,  #binary threshold
                                  ck1=3,  #window size of eroding
                                  ck2=5,  #window size of dilating
                                  yolo=YOLO())  # object detection

    previousframe = cv2.imread('./before.jpg') #image taken before
    currentframe  = cv2.imread('./after.jpg') #image taken now

    img_contour, location = find.detect_difference(img_before = previousframe,
                                                   img_after = currentframe)#return contours of changing areas and their location
    cv2.imwrite('./re/contours.jpg', img_contour
                )
    for key,value in location.items(): #due to downsampling in detect.py, change coordinates to original values
        key = [i * 2 for i in key]
        x,y,w,h = key

        cv2.rectangle(currentframe, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(currentframe, value, org=(x, y-15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=0.50, color=(255, 0, 0), thickness=1)

    cv2.imwrite('./re/result.jpg',currentframe)
    #cv2.imshow('re',currentframe) #display
    #cv2.waitKey(500)


