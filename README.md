Find Changes between Two Images
===============================

Introduction
------------

Compare two images taken at the same location in different time to find changed areas, and then detect whether
there exists some targets (person & car in my project) in changed areas.

**The camera position & angle have little difference between previous and later images in my project.**

1. Use SIFT feature matching and homography transformation to align the same parts of the two images.

2. Subtract two images, then perform image eroding & dilation (to decrease influence of image distortion) on the subtracted results.
   Next, find contours to obtain changed areas and display on the later image.

3. Detect whether there have objects I need in changed areas by using YOLOv3-tiny.
   **You need to download the weight file yourself or use other weight/net to get better results.**
   (based on qqwweee's code (https://github.com/qqwweee/keras-yolo3)

4. Template matching in OpenCV can be used to further detect objects. (See template.py)

Step
----

1. Environment: OpenCV == 3.4.2, (If use YOLOv3)Tensorflow-gpu == 1.6.0 & Keras == 2.2.1

2. Run **results.py** to get results saved in **./re** file.
