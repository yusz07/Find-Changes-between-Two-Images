import cv2
import numpy as np
from diff import cut_image, subtract, filter
from PIL import Image

#Find contours, do image processing and use YOLOv3 to detect targets.

#我的数据集上有图像扭曲的情况，所以采用腐蚀和膨胀去除作差得图像上的干扰区域
# There exists image distortion caused by atmosphere turbulence on my images,
# so I perform image eroding then dilating to remove interferences on image subtract results

class find_change_and_detect(object):
    def __init__(self, threshold_for_binary,ck1,ck2,yolo):
        self.threshold_for_binary = threshold_for_binary # binary threshold
        self.ck1 = (ck1 , ck1) # window size of eroding
        self.ck2 = (ck2 , ck2) # window size of dilating, 7*7 performs best
        self.yolo = yolo


    def find_contours(self, img_diff,imageB):
        #在作差得到的图像上找轮廓和外接矩形框，并在后一张图像上显示
        #Find contours and bounding rectangle on previous subtract results, then display on later image
        location = {}

        # Get binary image
        _, binary = cv2.threshold(img_diff, self.threshold_for_binary, 255, cv2.THRESH_BINARY)
        #Dilating on binary image，number of iterations can be adjusted 迭代次数可调整
        binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.ck2),iterations=1)
        #cv2.imwrite('./re/erod_binar.jpg', binary)

        #find contours
        image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i , c in enumerate(contours):
            # get bounding box and its coordinates
            x, y, w, h = cv2.boundingRect(c)
            # calculate area
            area = cv2.contourArea(c)
            #select bounding boxes to further decrease the effects of image distortion
            #筛选矩形框以进一步去除图像扭曲的影响
            if 200 < area <10000 and x !=0 and y != 0:
              if 0.3 <w/h <3:
                bbox = imageB[y-5:y+h+5,x-5:x+w+5]
                bbox = Image.fromarray(cv2.cvtColor(bbox, cv2.COLOR_BGR2RGB))
                #draw bounding boxes on substract results 差值图上画出矩形框
                cv2.rectangle(img_diff, (x, y), (x + w, y + h), (255, 255, 255), 1)
                cv2.drawContours(img_diff, contours, -1, (255, 255, 255), 1)

                #text = self.match_template(bbox) #对有变化的区域进行模板匹配
                text = self.yolo_detect(bbox)
                location[(x,y,w,h)] = text #return location & classes / 返回字典，包含位置和对应的类别

                # draw bounding boxes on imageB
                #cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 255, 0), 1)
                #cv2.putText(imageB, text, org=(x, y-15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #    fontScale=0.50, color=(255, 0, 0), thickness=1)

        return img_diff,location

    def yolo_detect(self,img):
        # yolov3-tiny object detection based on qqwweee's code (https://github.com/qqwweee/keras-yolo3)

        # I did not train a new net due to limited dataset, so I only just let what I need display.
        #由于数据集有限，没有重新训练，这里只是让我需要的类别显示

        text = self.yolo.detect_image(img)
        if text == 'person' or text == 'car' or text == 'truck':
            name = text
        else:
            name = None
        return name

    def gamma(self,img):  # gaamma contrast enhancement
        output = np.power(img / 255.0, 1.2)
        return (output * 255).astype(np.uint8)

    def detect_difference(self, img_before,img_after):

        #img_before, img_after = cut_image(img_before, img_after)

        img_before = cv2.pyrDown(img_before)#2x downsampling
        img_after = cv2.pyrDown(img_after)

        diff = subtract(img_before,img_after) #substract result
        #diff = gamma(diff)

        conv_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.ck1)  # 3*3 image eroding
        img_erod = cv2.erode(diff, conv_kernel, iterations=2)
        conv_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, self.ck2)    #7*7 image dilating
        img_dilate = cv2.dilate(img_erod, conv_kernel1)

        # get contours of changing targets and their location,draw on the later image(imageB)
        img_contour, location = self.find_contours(img_diff = img_dilate, imageB = img_after)


        return img_contour, location


