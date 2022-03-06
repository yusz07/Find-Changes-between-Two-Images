import cv2
import numpy as np
import os


#Due to my limited dataset and poor effects for weights trained on public datasets,
# I added template matching and cut objects from my own dataset as templates to classify changed targets.

def match_template(img, temp_path_1, temp_path_2, threshold1,threshold2):
        list1 = []
        list2 = []
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        file_list = os.listdir(temp_path_1)
        for file in file_list:
            #print(file)
            template = cv2.imread(os.path.join(temp_path_1,file),0) #read templates(gray image)
            w, h = template.shape[::-1]
            #print(w,h)
            img_n = cv2.resize(img,(w,h))
            res = cv2.matchTemplate(img_n, template, cv2.TM_CCOEFF_NORMED) #template matching
            list1.append(res)
        maxVal1 = max(list1)
        print(maxVal1)
        print('now to another class')
        #detect another class
        file_list = os.listdir(temp_path_2)
        for file in file_list:
         #   print(file)
            template = cv2.imread(os.path.join(temp_path_2, file), 0)
            w, h = template.shape[::-1]
            img_n = cv2.resize(img, (w, h))
            res = cv2.matchTemplate(img_n, template, cv2.TM_CCOEFF_NORMED)
            list2.append(res)
         #   print(res)
        maxVal2 = max(list2)
        print(maxVal2)
        if maxVal1 > threshold1:
            text = 'person'
        elif maxVal2 > threshold2:
            text = 'car'
        else:
            text = None
        print(text)

        return text


if __name__ == '__main__':
    img = cv2.imread('bbox/1.jpg')#changed areas, may include object I need
    temp_path_1 = 'temp/template_img/car/'  #path to templates
    temp_path_2 = 'temp/template_img/infred_person/'
    text = match_template(img,temp_path_1,temp_path_2,0.5,0.5)#threshold value can be adjusted
