#!/usr/bin/env python

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2 as cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

image_path = "empty"

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self.browse_button.clicked.connect(self.SLOT_browse_button)     # add functionality to our button for image selection, go to function below

        self._cam_id = 0         # for a live screen display, "0" is typically the id of the laptop webcam
        self._cam_fps = 10      # if you want to have the camera update faster
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        # camera resolution is set to 320x240
        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

    # helps us select an image
    def SLOT_browse_button(self):
        global image_path       # allows the global variable to be changed from inside this function
        
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        
        # get a path to our image
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        image_path = self.template_path
        # print(image_path)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):

        sift = cv2.xfeatures2d.SIFT_create()

        # Camera Frame
        ret, frame = self._camera_device.read()                         # get camera image
        kp_frame, desc_frame = sift.detectAndCompute(frame, None)       # get keypoints and descriptions
        frame = cv2.drawKeypoints(frame, kp_frame, frame)    # this will draw the keypoints on the camera captured frame
        pixmap_frame = self.convert_cv_to_pixmap(frame)
        self.live_image_label.setPixmap(pixmap_frame)       # this can only print pixelmaps, prints frame with keypoints
        
        # Selected Image
        # #print(image_path)
        img = cv2.imread(image_path, 0)
        kp_image, desc_image = sift.detectAndCompute(img, None)
        #img = cv2.drawKeyPoints(img, kp_image, img) # draw the keypoints on our image, pass it where you want to draw, the keypoints, and outer image (img)
        
        # Feature Matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_image, desc_frame, k=2)

        good_points = []
        # m and n are arrays, m holds the original image, and n holds the camera cap. grayframe
        for m, n in matches:
            # to avoid many false results, take descriptors that have short distances between them
            # play with this constant in front of n.distance: 0.6, 0.8
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        img3 = cv2.drawMatches(img, kp_image, frame, kp_frame, good_points, frame)
        cv2.imshow("Matches", img3)

        # Homography
        # if we find at least 10 matches, we will draw homography
        # anywhere that mentions query, i mean the img, and train refers to the camera captured frame
        if len(good_points) > 10:
            # queryIdx gives us the points of the query image (from our m array)
            # the .reshape just changes the shape of the numpy array
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

            # matrix shows object from its perspective?
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()        # extract points from mask and put into a list

            # Perspective transforms, helps with homography
            h, w = img.shape        # height and width of original image
            #print(h)
            #print(w)

            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)    # points gets h and w of image. does not work with int32, but float32 works
            dst = cv2.perspectiveTransform(pts, matrix)

            # convert to an integer for pixel pointers, (you can't point to a decimal of a pixel)
            # True is for "closing the lines"
            # next is the colour we select, in bgr, we have selected blue
            # thickness = 3
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

            cv2.imshow("Homography", homography)
        else:
            cv2.imshow("Regular Frame", frame)


    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())

