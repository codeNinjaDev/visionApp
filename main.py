import kivy
kivy.require('1.10.1') # replace with your current kivy version !
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty, ReferenceListProperty,\
    ObjectProperty
from kivy.clock import Clock
import cv2
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import numpy as np
from kivy.uix.slider import Slider
import math
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
#Calculated by hand
HFOV = 55.794542
VFOV = 50.92
#Calculated in program
H_FOCAL_LENGTH = 808
V_FOCAL_LENGTH = 707


class VisionWidget(Image):
    # define range of blue (my notebook) in HSV
    maskingButton = ObjectProperty(None)
    resultButton = ObjectProperty(None)
    contourButton = ObjectProperty(None)

    closeHPopUpButton = ObjectProperty(None)
    hViewInput = ObjectProperty(None)
    hFocalInput = ObjectProperty(None)

    state = 'rgb'
    hMin = 0
    hMax = 180
    sMin = 0
    sMax = 255
    vMin = 0
    vMax = 255
    def returnThreshold(self, frame):
        lower_color = np.array([self.hMin, self.sMin, self.vMin])
        upper_color = np.array([self.hMax, self.sMax, self.vMax])
        #lower_color = np.array([100, 157, 66])
        #upper_color = np.array([116, 255, 255])
        print("Lower Threshold: " + str(lower_color))
        print("Upper Threshold: " + str(upper_color))


        mask = self.threshold_video(frame, lower_color, upper_color)
        

        

        return mask        
    def mask(self, frame):
        mask = self.returnThreshold(frame)

        screenHeight, screenWidth, channels = frame.shape
        hue = "H: " + str(self.hMin) + "-" + str(self.hMax)
        value = "V: " + str(self.vMin) + "-" + str(self.vMax)
        saturation = "S: " + str(self.sMin) + "-" + str(self.sMax)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask,hue,(math.floor(screenWidth * 3.8/ 5),math.floor(screenHeight / 10)), font, .75,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(mask,saturation,(math.floor(screenWidth * 3.8/ 5),math.floor(screenHeight / 5) - 8), font, .75,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(mask,value,(math.floor(screenWidth * 3.8/ 5),math.floor(screenHeight*3 / 10) - 12), font, .75,(255,255,255),2,cv2.LINE_AA)
        self.turnFrameIntoImage(mask, 'luminance')


        
    def res(self, frame, mask):
        res = cv2.bitwise_and(frame,frame, mask= mask)
        screenHeight, screenWidth, channels = frame.shape
        hue = "H: " + str(self.hMin) + "-" + str(self.hMax)
        value = "V: " + str(self.vMin) + "-" + str(self.vMax)
        saturation = "S: " + str(self.sMin) + "-" + str(self.sMax)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(res,hue,(math.floor(screenWidth * 3.8/ 5),math.floor(screenHeight / 10)), font, .75,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(res,saturation,(math.floor(screenWidth * 3.8/ 5),math.floor(screenHeight / 5) - 8), font, .75,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(res,value,(math.floor(screenWidth * 3.8/ 5),math.floor(screenHeight*3 / 10) - 12), font, .75,(0,255,0),2,cv2.LINE_AA)
        self.turnFrameIntoImage(res, 'bgr')

        return res
    def contour(self, frame, mask):
        contourImage = self.findContours(frame, mask)
        screenHeight, screenWidth, channels = frame.shape
        hue = "H: " + str(self.hMin) + "-" + str(self.hMax)
        value = "V: " + str(self.vMin) + "-" + str(self.vMax)
        saturation = "S: " + str(self.sMin) + "-" + str(self.sMax)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(contourImage,hue,(math.floor(screenWidth * 3.8/ 5),math.floor(screenHeight / 10)), font, .75,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(contourImage,saturation,(math.floor(screenWidth * 3.8/ 5),math.floor(screenHeight / 5) - 8), font, .75,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(contourImage,value,(math.floor(screenWidth * 3.8/ 5),math.floor(screenHeight*3 / 10) - 12), font, .75,(0,255,0),2,cv2.LINE_AA)
        self.turnFrameIntoImage(contourImage, 'bgr')

        return contourImage
    def moveToMasking(self):
        self.state = 'mask'
        self.resultButton.size = self.maskingButton.size
        self.resultButton.text = "Result->"

        self.maskingButton.size = (0, 0)
        self.maskingButton.text = ""
    def moveToResult(self):
        self.contourButton.size = self.resultButton.size
        self.contourButton.text = "Contour->"

        self.resultButton.size = (0, 0)
        self.resultButton.text = ""
        self.state = 'result'
            
    def moveToContour(self):
        #if(math.isnan(self.hFocalInput.text) is not True):
        #    H_FOCAL_LENGTH = self.hFocalInput.text
        #if(math.isnan(self.hViewInput.text) is not True):
        #    HFOV = self.hViewInput.text
        self.state = 'contour'
        self.contourButton.size = (0, 0)
        self.contourButton.text = ""

        
    def update(self, dt):
        ret, frame = self.cap.read()
        #should get changed to actual mask before being used
        threshold = self.returnThreshold(frame)

        if(self.state == 'mask'):
            self.mask(frame)
        elif(self.state == 'result'):
            res = self.res(frame, threshold)
        elif(self.state == 'contour'):
            contour = self.contour(frame, threshold)
        else:
            self.turnFrameIntoImage(frame, 'bgr')



    def turnFrameIntoImage(self, frame, colorFormat):
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt=colorFormat)
        image_texture.blit_buffer(buf, colorfmt=colorFormat, bufferfmt='ubyte')
        # display image from the texture
        self.texture = image_texture
    # Masks the video based on a range of hsv colors
    # Takes in a frame, returns a masked frame
    def threshold_video(self, frame, lower_hsv, upper_hsv):
        print("Thresholding...")
        #Gets the shape of video
        print("Frame shape: " + str(frame.shape))
        screenHeight, screenWidth, channels = frame.shape

        #Gets center of height and width
        centerX = (screenWidth / 2) - .5
        centerY = (screenHeight / 2) - .5
        blur = cv2.medianBlur(frame, 5)
        print("Blurring...")

        # Convert BGR to HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        print("Converting to HSV...")

        # hold the HSV image to get only red colors
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        print("Masking...")
        #cv2.imshow("mask", mask)
        # Returns the masked imageBlurs video to smooth out image
        
        return mask   
    #Finds the contours from the masked image and displays them on original stream
    def findContours(self, frame, mask):
        #Finds contours
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        screenWidth, screenHeight, channels = frame.shape
        # Take each frame
        #Flips the frame so my right is the image's right (probably need to change this
        #frame = cv2.flip(frame, 1)
        #Gets the shape of video
        screenHeight, screenWidth, channels = frame.shape
        #Gets center of height and width
        centerX = (screenWidth / 2) - .5
        centerY = (screenHeight / 2) - .5
        #Copies frame and stores it in image
        image = frame.copy()
        #Processes the contours, takes in (contours, output_image, (centerOfImage) #TODO finding largest
        if len(contours) != 0:
            self.processLargestContour(contours, image, centerX, centerY)
        return image

    #Draws and calculates properties of largest contour
    def processLargestContour(self,contours, image, centerX, centerY):
        screenHeight, screenWidth, channels = image.shape

        if len(contours) != 0:

            cnt = self.findBiggestContours(contours)
            #Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            #Get convex hull (bounding polygon on contour)
            hull = cv2.convexHull(cnt)
            #Calculate Contour area
            cntArea = cv2.contourArea(cnt)
            #calculate area of convex hull
            hullArea = cv2.contourArea(hull)
            #Filters contours based off of size
            if (self.checkContours(cntArea, hullArea)):
                #Gets the centeroids of contour
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0

                #Gets rotated bounding rectangle of contour
                rect = cv2.minAreaRect(cnt)
                #Creates box around that rectangle
                box = cv2.boxPoints(rect)
                #Not exactly sure
                box = np.int0(box)
                #Gets center of rotated rectangle
                center = rect[0]
                #Gets rotation of rectangle; same as rotation of contour
                rotation = rect[2]
                #Gets width and height of rotated rectangle
                width = rect[1][0]
                height = rect[1][1]
                #Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
                rotation = self.translateRotation(rotation, width, height)
                #Gets smaller side
                if width > height:
                    smaller_side = height
                else:
                    smaller_side = width
                #Calculates yaw of contour (horizontal position in degrees)
                yaw = self.calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                #Calculates yaw of contour (horizontal position in degrees)
                pitch = self.calculatePitch(cy, centerY, V_FOCAL_LENGTH)
                #
                #Adds padding for text
                padding  = -8 - math.ceil(.5*smaller_side)
                #Draws rotated rectangle
                cv2.drawContours(image, [box], 0, (23, 184, 80), 3)

                #Draws a vertical white line passing through center of contour
                cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
                #Draws a white circle at center of contour
                cv2.circle(image, (cx, cy), 6, (255, 255, 255))
                #Puts the rotation on screen
                cv2.putText(image, "Rotation: " + str(rotation), (cx + 40, cy + padding), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
                #Puts the yaw on screen
                cv2.putText(image, "Yaw: " + str(yaw), (cx+ 40, cy + padding -16), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
                #Puts the Pitch on screen
                cv2.putText(image, "Pitch: " + str(pitch), (cx+ 80, cy + padding -42), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))


                #Draws the convex hull
                #cv2.drawContours(image, [hull], 0, (23, 184, 80), 3)
                #Draws the contours
                cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

                #Gets the (x, y) and radius of the enclosing circle of contour
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                #Rounds center of enclosing circle
                center = (int(x), int(y))
                #Rounds radius of enclosning circle
                radius = int(radius)
                #Makes bounding rectangle of contour
                rx, ry, rw, rh = cv2.boundingRect(cnt)

                #Draws countour of bounding rectangle and enclosing circle in green
                cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)
                cv2.circle(image, center, radius, (23, 184, 80), 1)

                return image
    #Draws and calculates contours and their properties
    def processContours(self, contours, image, centerX, centerY):

        #Loop through all contours
        for cnt in contours:
            #Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            #Get convex hull (bounding polygon on contour)
            hull = cv2.convexHull(cnt)
            #Calculate Contour area
            cntArea = cv2.contourArea(cnt)
            #calculate area of convex hull
            hullArea = cv2.contourArea(hull)
            #Filters contours based off of size
            if (self.checkContours(cntArea, hullArea)):
                #Gets the centeroids of contour
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0

                #Gets rotated bounding rectangle of contour
                rect = cv2.minAreaRect(cnt)
                #Creates box around that rectangle
                box = cv2.boxPoints(rect)
                #Not exactly sure
                box = np.int0(box)
                #Gets center of rotated rectangle
                center = rect[0]
                #Gets rotation of rectangle; same as rotation of contour
                rotation = rect[2]
                #Gets width and height of rotated rectangle
                width = rect[1][0]
                height = rect[1][1]
                #Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
                rotation = translateRotation(rotation, width, height)
                #Gets smaller side
                if width > height:
                    smaller_side = height
                else:
                    smaller_side = width
                #Calculates yaw of contour (horizontal position in degrees)
                yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                #Calculates yaw of contour (horizontal position in degrees)
                pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)
                #
                #Adds padding for text
                padding  = -8 - math.ceil(.5*smaller_side)
                #Draws rotated rectangle
                cv2.drawContours(image, [box], 0, (23, 184, 80), 3)

                #Draws a vertical white line passing through center of contour
                cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
                #Draws a white circle at center of contour
                cv2.circle(image, (cx, cy), 6, (255, 255, 255))
                #Puts the rotation on screen
                cv2.putText(image, "Rotation: " + str(rotation), (cx + 40, cy + padding), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
                #Puts the yaw on screen
                cv2.putText(image, "Yaw: " + str(yaw), (cx+ 40, cy + padding -16), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
                #Puts the Pitch on screen
                cv2.putText(image, "Pitch: " + str(pitch), (cx+ 80, cy + padding -42), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))


                #Draws the convex hull
                #cv2.drawContours(image, [hull], 0, (23, 184, 80), 3)
                #Draws the contours
                cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

                #Gets the (x, y) and radius of the enclosing circle of contour
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                #Rounds center of enclosing circle
                center = (int(x), int(y))
                #Rounds radius of enclosning circle
                radius = int(radius)
                #Makes bounding rectangle of contour
                rx, ry, rw, rh = cv2.boundingRect(cnt)

                #Draws countour of bounding rectangle and enclosing circle in green
                cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)
                cv2.circle(image, center, radius, (23, 184, 80), 1)

                return image
    #Checks if contours are worthy based off of contour area and (not currently) hull area
    def checkContours(self, cntSize, hullSize):
        if(cntSize > 10000):
            return True
        else:
            return False;


    def translateRotation(self, rotation, width, height):
        if (width < height):
            rotation = -1 * (rotation - 90)
        if (rotation > 90):
            rotation = -1 * (rotation - 180)
        rotation *= -1
        return rotation
    def calculateDistance(self, heightOfCamera, heightOfTarget, pitch):
        heightOfCameraFromTarget = heightOfTarget - heightOfCamera

        #Uses trig and pitch to find distance to target
        '''
        d = distance
        h = height between camera and target
        a = angle/pitch
        
        tan a = h/d (opposite over adjacent)
        
        d = h / tan a
        
                            .                 
                            /|        
                        / |       
                        /  |h        
                        /a  |       
                camera -----
                        d         
        '''
        distance = math.fabs(heightOfCameraFromTarget / math.tan(math.radians(pitch)))

        return distance
    #Uses trig and focal length of camera to find yaw.
    #Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
    def calculateYaw(self, pixelX, centerX, hFocalLength):
        yaw = math.degrees(math.atan((pixelX - centerX) / hFocalLength))
        return yaw
    #Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
    def calculatePitch(self, pixelY, centerY, vFocalLength):
        pitch = math.degrees(math.atan((pixelY - centerY) / vFocalLength))
        #Just stopped working have to do this:
        pitch *= -1
        return pitch
    def sendImportantContourInfo(self, contourX, contourY, contourYaw, contourPitch, contourRotation, contourArea):
        print("Contour X: " + str(contourX))
        print("Contour Y: " + str(contourY))
        print("Contour Yaw: " + str(contourYaw))
        print("Contour Pitch: " + str(contourPitch))
        print("Contour Rotation: " + str(contourRotation))
        print("Contour Area: " + str(contourArea))

    def findBiggestContours(self, contours):
        if len(contours) != 0:
            c = max(contours, key = cv2.contourArea)
            return c
        else:
            return None;     
    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

    def openCamera(self):
        self.cap = cv2.VideoCapture(0)

    def setHueMin(self, *args):
        self.hMin = int(self.clamp(int(args[1]), 0, 180))
        
    def setHueMax(self, *args):
        self.hMax = int(self.clamp(int(args[1]), 0, 180))
    def setSaturationMin(self, *args):
        self.sMin = int(self.clamp(int(args[1]), 0, 255))
    def setSaturationMax(self, *args):
        self.sMax = int(self.clamp(int(args[1]), 0, 255))
    def setValueMin(self, *args):
        self.vMin = int(self.clamp(int(args[1]), 0, 255))
    def setValueMax(self, *args):
       self.vMax = int(self.clamp(int(args[1]), 0, 255))

class VisionApp(App):
    def build(self):
        vision = VisionWidget()
        vision.openCamera()
        Clock.schedule_interval(vision.update, 1.0 / 120.0)
        return vision


if __name__ == '__main__':
    VisionApp().run()