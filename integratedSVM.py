import numpy as np
import sys
import cv2
import csv
import os
import shutil
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
# import joblib
from skimage import color
from PIL import Image
import time
import pickle
from PIL import ImageChops
from pims import ImageSequence
# from numba import vectorize

config = 'predictor.sav'
imageStore = os.listdir("./imageStore")
svm = pickle.load(open(config, 'rb'))

def preProcessLocation(imPath, frameNum): 
    #Loading of origional image
    sphere = False
    # rawImage = cv2.imread(imPath)

    # rawImage = cv2.fastNlMeansDenoisingColored(rawImage,None,1,1,7,21)

    rawImage = imPath

    # if sphere:
    #     norm = cv2.normalize(rawImage, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #     cv2.imshow("origional", norm)
    #     cv2.waitKey(0) 
    #     cv2.destroyAllWindows()

    if rawImage.size > 100000:
        print(rawImage.shape)
        rawImage = cv2.resize(rawImage, (int(0.3*(np.shape(rawImage)[1])), int(0.3*(np.shape(rawImage)[0]))))
    brightImage = rawImage.copy()
    gray = cv2.cvtColor(brightImage, cv2.COLOR_RGB2GRAY)
    brightImage = cv2.cvtColor(brightImage, cv2.COLOR_RGB2BGR)
    


    # Uncomment to see grayscaled Immage

    # cv2.imshow("origional", gray)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    #----------------------------------------
    # Find area of the image with the largest intensity value

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    
    # maxLoc = minLoc
    
    maxLoc = np.transpose(maxLoc)
    maxLoc = tuple([maxLoc[1], maxLoc[0]])

    print("Location of brightest pixel: ", maxLoc)
    # cv2.circle(brightImage, maxLoc, 1, (255, 0, 0), 2)

    print("Intensity: ", gray[maxLoc])

    print("Median: ", np.median(gray))
    if np.median(gray) > 100:
        rawImage = (255-rawImage)
        intensityInvert = True
    else:
        intensityInvert = False
    

    # cv2.imshow("origional", gray)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

    # display where the brightest area of pixels is

    # cv2.imshow("Brightest Spot", brightImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #----------------------------------------
    # Preprocessing (using a bilateral filter)

    height = int(rawImage.shape[0])
    width = int(rawImage.shape[1])


    if gray[maxLoc] <= 50 and not intensityInvert:
        print("Low intensity image - concentrating image contours...")
        kernel = np.ones((5,5),np.uint8)  
        bilateral_filtered_image = cv2.bilateralFilter(rawImage, 5, height, width) # For dark images 
        edge_detected_image = cv2.Canny(bilateral_filtered_image, 0, 10)
        edge_detected_image = cv2.dilate(edge_detected_image ,kernel,iterations = 3)
        edge_detected_image = cv2.medianBlur(edge_detected_image, 11)
        edge_detected_image = cv2.erode(edge_detected_image,kernel,iterations = 2)
    else:    
        bilateral_filtered_image = cv2.bilateralFilter(rawImage, 1, height, width) # For bright images
        edge_detected_image = auto_canny(bilateral_filtered_image, True, intensityInvert)

   
    
    # cv2.imshow('Bilateral', bilateral_filtered_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #----------------------------------------
    # Edge detection
    # 75 and 200 default min/max for canny edge detection


    
    print("Detecting islands.....")
    
   #----------------------------------------
    corners = cv2.goodFeaturesToTrack(edge_detected_image, 500, .0001, 7, useHarrisDetector=True)

    corners = np.int0(corners) 

    cornerList = []
  
    for i in corners: 
        x, y = i.ravel() 
        cornerList.append((x,y))
    #----------------------------------------
        # cv2.circle(brightImage, (x, y), 5, (255,0,0), 1) 
        
# uncomment to see edge detected image
    cv2.imshow('Edge detected image', edge_detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#     shutil.rmtree('./Temp Images')
#     os.mkdir("./Temp Images")
    #----------------------------------------
    # Finding Contours
   
    contours, _ = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# print(contours)
    

    contour_list = []
    for contour in contours:        
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
    #     if area > 0:
    #         print("Pixel area: ", area)
        if ((len(approx) > 1) or (area > 5) ):  # len 8, area 30 are default
            contour_list.append(contour)
    #----------------------------------------
    # convert the grayscale image to binary image
    # ret,thresh = cv2.threshold(gray,127,255,0)

    # calculate moments of binary image
    os.chdir("./imageStore")
    imCount = 1
    for cnt in contour_list:
        M = cv2.moments(cnt)
        if int(M["m00"] != 0):
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            
            x,y,w,h = cv2.boundingRect(cnt)
            xCent = int((x+x+w)/2)
            yCent = int((y+y+h)/2)
            cv2.rectangle(rawImage,(x,y),(x+w,y+h),(255,0,0),2)

            if (y-1 == -1) or (x-1 == -1):	
            	cropped = edge_detected_image[y:y+h+1, x:x+w+1]
            	fileName = "edgeImage" + str(imCount) + ".png"
            	# print(y, x)
            else:
            	cropped = edge_detected_image[y-1:y+h+1, x-1:x+w+1]
            	fileName = "croppedImage" + str(imCount) + ".png"
#             print(fileName)
            cv2.imwrite(fileName, cropped)
            imCount += 1

            # cv2.imshow('Cropped island hits', cropped)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if classifyObject(fileName) == [1]:
                suppPath = '../suppTrain/hits'
                cv2.imwrite(os.path.join(suppPath, fileName), cropped)
                # cv2.rectangle(brightImage, (xCent, yCent), (xCent, yCent), (0,255,255),6) # Draw rectangle centers
                # cv2.circle(brightImage, (cX, cY), 2, (0, 0, 255), 2)   # Draw centers in relation to moments
                cv2.rectangle(brightImage, (x,y), (x+w,y+h),(255,0,0),2)
                relevant = [xCent, yCent, w+h/4, frameNum]
                writer.writerow(relevant)

            elif classifyObject(fileName) == [2]:
                suppPath = '../suppTrain/groups'
                cv2.imwrite(os.path.join(suppPath, fileName), cropped)
                cv2.rectangle(brightImage, (x,y), (x+w,y+h),(0,255,255),2)
                for corner in cornerList:
                    a = corner[0]
                    b = corner[1]
                    within = (x < a < x+w) and (y < b < y+h)
                    # if within: 
                        # cv2.circle(brightImage, (a, b), 5, (255,0,0), 1) 
                        





            else: 
            	suppPath = '../suppTrain/negs'
            	cv2.imwrite(os.path.join(suppPath, fileName), cropped)

    # cv2.imshow("Rects", rawImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
         
           

    #----------------------------------------

    # Displaying Resuts
    # cv2.imshow('Library Detected Image',rawImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('detectedImg.png', brightImage)
    cv2.imshow('Objects Detected',brightImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preProcessFeatures(islandHit):
	# if os.getcwd())
	orientations = 9
	pixels_per_cell = (8, 8)
	cells_per_block = (2, 2)
	threshold = .3
	try:
		img = Image.open(islandHit)
	except OSError:
		print("Bad file: ", islandHit)
		return np.array([[1], [1]])

	img = img.resize((64,128))
	gray = img.convert('L') 
	# HOG for positive features
	fd, hog_image = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True, visualize=True)


	# cv2.imshow("hogFilter", hog_image)
	# cv2.waitKey(0) 
	# cv2.destroyAllWindows()

	X_new = np.array(fd, ndmin=2)
	return X_new




def classifyObject(image):

	X_new = preProcessFeatures(image)
	if X_new.shape[1] == 1:
		return -1
	return(svm.predict(X_new))
# @jit
def auto_canny(image, sphere, intensityInvert):
	# compute the median of the single channel pixel intensities
    v = np.median(image)
    if intensityInvert and v < 15:
        image = cv2.bilateralFilter(image,1,75,75)
        sigma = 1.9
        print("path 1: ", v)
    elif 0 <= v <= 10:
        image = cv2.bilateralFilter(image,6,75,75)
        sigma = 1
        print("path 2")
    elif 10 < v <= 12:
        image = cv2.bilateralFilter(image,6,75,75)
        sigma = 8
        print("path 3")
    elif 12 < v <= 15:
        image = cv2.bilateralFilter(image,6,75,75)
        sigma = 5
        print("path 4")
    elif 15 < v <= 25:
        image = cv2.bilateralFilter(image,3,75,75)
        sigma = 0.22
        print("path 5")
    elif 25 < v <= 35:
        image = cv2.bilateralFilter(image,6,75,75)
        sigma = 2
        print("path 6")
        if sphere:
            kernel = np.ones((5,5),np.uint8)
            image = cv2.fastNlMeansDenoisingColored(image,None,5,5,7,21)
            # image = cv2.dilate(image ,kernel,iterations = 5)
            # image = cv2.medianBlur(image, 5)
            # image = cv2.erode(image,kernel,iterations = 2)
    elif 35 < v <= 70:
        image = cv2.bilateralFilter(image,4,75,75)
        sigma = 1
    # elif 65 < v:
    #     image = cv2.bilateralFilter(image, 1, 75, 75)
    #     sigma = 1.3
    #     print("path 7: ", v)

	# apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    
	# return the edged image
    return edged
     

shutil.rmtree('./imageStore')
time.sleep(.0001)
os.mkdir("./imageStore")

# imagePath = "./images/compare5.tif"
# imagePath = "./images/keithTest1.tif" 
# imagePath = "./images/chainformation4.png" 
# imagePath = "./images/chainformation3.png" 
# imagePath = "./images/chainformation2.png"
# imagePath = "./images/chainformation1.png" 
# imagePath = "./Images/islandtest3.tif"   
# imagePath = "./Images/islandtest1.tif"  
# imagePath = "./Images/islandtest2.tif"
# imagePath = "./Images/macro_image_09233.tiff"
imagePath = "C:/Users/Keith/Desktop/SVM-Droplet-Tracking/Data/"

os.remove("./islandCenters.csv")
islandCenters = open("./islandCenters.csv", "w", newline='')
writer = csv.writer(islandCenters)
indexes = ['x','y','radius','frame']
writer.writerow(indexes)

rawImgs = ImageSequence(imagePath+'*.tif')


for counter,i in enumerate(rawImgs):
    preProcessLocation(i,counter)

islandCenters.close
t = time.process_time()
print('The run time was ', t, ' seconds.')