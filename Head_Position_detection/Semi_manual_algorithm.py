import cv2
import os
import numpy as np
import math
import dlib
from skimage.segmentation import slic # sckit-image implementation
from skimage.color import label2rgb
from skimage.future import graph

#The algorithm is executed via 4 stages.
#1.Pixel-based skin colour detection
#2.Contour detection
#3.Facial landmark detection
#4.Head position detection

#Dataset for the Haar-cascade face detection : haarcascade_frontalface_default.xml
#From https://github.com/opencv/opencv/tree/master/data/haarcascades

#IBUG 300-W dataset for the facial landmark detection : shape_predictor_68_face_landmarks.dat
#From https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

CASC_PATH = r'/home/pi/Downloads/haarcascade_frontalface_default.xml'  
PREDICTOR_PATH= r'/home/pi/Downloads/shape_predictor_68_face_landmarks.dat'  

#Function
def Load():
  global Img,onlyfiles
  #Open a file
  path=r'/home/pi/Desktop/TEST2/Horim'
  #dirs = os.listdir(path) # returns a list containing the name of the entries in the directory given by path
  onlyfiles = [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) ]
  #isfile(path) : return true if path is an existing regular file

  Img= np.empty(len(onlyfiles), dtype=object) #dtype=object means all other things
  for n in range(0,len(onlyfiles)):
    Img[n] = cv2.imread(os.path.join(path,onlyfiles[n]),1)
    #os.path.join(path,*paths) : join one or more path components
    #i.e.os.path.join("c:","foo"):c:foo'

# Create the Haar cascade for Haar-cascade face detection & predictor for facial landmark
faceCascade = cv2.CascadeClassifier(CASC_PATH)   
predictor = dlib.shape_predictor(PREDICTOR_PATH)  

# Call the function to Load
Load() # Load the image

## Pixel-based skin colour detection

for J in range(len(Img)):
  Data1 =Img[J]
  
# Make a temporary mask
  Mask=cv2.cvtColor(Data1,cv2.COLOR_BGR2GRAY)
  ret,Mask=cv2.threshold(Mask,10,255,cv2.THRESH_BINARY)
  Mask[:]=0

# Construct the channels,with extracted the normalised RGB & HSV Channels
  [x,y,_]=Data1.shape
  seg=np.zeros((x,y,3),np.float32)
  image=np.zeros((x,y,3),np.uint8)
  b=Data1[:,:,0] #Blue channel
  g=Data1[:,:,1] #Green channel
  r=Data1[:,:,2] #Red channel
  HSV=cv2.cvtColor(Data1,cv2.COLOR_BGR2HSV)  
  summ=b+g+r #Summation of red, bule, and green channel
  seg[:,:,1]=g/summ #Normalised green channel
  seg[:,:,0]=r/summ #Normalised red channel

# Skin colour classification
# Explictly defined skin boundary model: ratio of the normalised green channel and red channel & HSV Filter
  for z in range(0,x):
   for k in range(0,y):
    if not((seg[z,k,0]/seg[z,k,1]>1.15) and(HSV[z,k,0]>0 and HSV[z,k,0]<30)or((HSV[z,k,0]>=335 and HSV[z,k,0]<=360))):
           continue 
    else:
       image[z,k,:]= Data1[z,k,:]
       Mask[z,k]=255 # compute the skin colour region as 255 in the mask
       
## Perform Contour detection 
# Find the contour
   _, contours, _ = cv2.findContours(Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  Area=[] # to save the area
  hull=[] # to save the calculated coordinates
  
# Calculate the Area of the contours and save them in a list
  for I in range(len(contours)):
    Area.append(cv2.contourArea(contours[I]))
  Max=max(Area) # to isolate the larger region
  for j in range(len(Area)):
    if Area[j]==Max:
      cnt=contours[j]
          
# Get the coordinates from the contour
# Extract the extreme points
  extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
  extBot = tuple(cnt[cnt[:, :, 1].argmax()][0]) 
  extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
  extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
  
# Extract the coordinates of the outer, surrounding the contour 
  hull=(cv2.convexHull(cnt))
  
#Get the outline of the contour
  Con=cv2.approxPolyDP(hull,0.01*cv2.arcLength(hull,True),True)
  #cv2.approxPlyDP(curve,epsilon,closed,approxCurve) : to find the arc length of the contour
  #cv2.arcLength(curve,closed):
  #curve: Input vector of 2D points
  #closed : Flag indicating whether the curve is closed or not
  #epsilon: accouracy parameter: ,0.01*cv2.arcLength(hull,True)
  #curve : Input vector of a 2D Point
  #approx Curve : result of the approximation
  #closed:if True, the aproximated curve is closed == first and last vertices are connected
  # using the index of the contour, find the circle
    
#Get the coordinates & radius, using the contour
  (x,y),radius=cv2.minEnclosingCircle(cnt)
  center=(int(x),int(y))

# Reset the previous mask
  A=Mask.copy()
  A[:]=0
  
# Draw the largest contour
  cv2.drawContours(A,[Con],-1,(255.255),-1)

# Apply the new mask to the image to segment the head
  face=cv2.bitwise_and(image,image,mask=A)
  
## Head position detection
  gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  

# Detect faces in the image  
  faces = faceCascade.detectMultiScale(
       gray,  
       scaleFactor=1.05,  
       minNeighbors=5,  
       minSize=(70, 70),  
       flags=cv2.CASCADE_SCALE_IMAGE)  

  if (len(faces)>0): # if the face is detected
   for (x, y, w, h) in faces:  
      dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h)) # Convert the OpenCV rectangle coordinates to Dlib rectangle

# Perform the facial landmark detection    
   detected_landmarks = predictor(face, dlib_rect).parts()  
   
   landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
   Left_eye_corner=landmarks[37] 
   Right_eye_corner=landmarks[46]
   Nose=landmarks[34]
   Chin=landmarks[9]
   Left_mouth_corner=landmarks[49]
   Right_mouth_corner=landmarks[55]
   
# Construct the detected facial landmark points as an array  
   image_points = np.array([
                             Nose.tolist()[0],     # Nose tip
                             Chin.tolist()[0],     # Chin
                             Left_eye_corner.tolist()[0],     # Left eye left corner
                             Right_eye_corner.tolist()[0],     # Right eye right corne
                             Left_mouth_corner.tolist()[0],     # Left Mouth corner
                             Right_mouth_corner.tolist()[0]      # Right mouth corner
                           ], dtype="double")

   p2 = ( int(image_points[1][0]),int(image_points[1][1])) # chin
   pp1 = ( int(image_points[2][0]),int(image_points[2][1])) # Left eye left corner
   pp2 = ( int(image_points[3][0]),int(image_points[3][1])) # Right eye Right corner
   p3= ((pp1[0]+pp2[0])//2,(pp1[1]+pp2[1])//2) # center point between left eye and right eye
   p4 = ( int(image_points[0][0]),int(image_points[0][1])) # nose


# Calculate the ratio for the head position detection
   Le=landmarks[2].tolist()
   Ri=landmarks[16].tolist()

# Calculate the ratio, Left/Right to classify left or right side
   dL=(Le[0][1]-p4[1]) # From center to upper
   dR=(Ri[0][1]-p4[1]) # From center to bottom
   if dL==0 or dR==0 :
     ratio2=0
   else:
     ratio2=np.abs(dL/dR)
     
# Calculate the ratio of the Upper/Bottom to classify up or down
   dU=(p4[0]-p3[0])
   dB=(p4[0]-p2[0])
   if dU==0 or dB==0:
     ratio=0
   else:
     ratio=np.abs(dU/dB)

# Perform the head position detection
# Classify the head position in Up / Down / Front
# Front into Left / Right

   title='detected %dth'%J   
   if (ratio>0.18 and ratio<=0.27):
      print(title,'Head is down and check other measurements')
   if ((ratio>1.6 and ratio<2) or (ratio>0.4 and ratio<0.5) or (ratio>0.72 and ratio<0.75) or (ratio>=0.8 and ratio<1) or (ratio>0.16 and ratio<=0.18)):
      print(title,'Head is Up and check other measurements')
   if ((0.75<=ratio and ratio<0.8) or (0.5<=ratio and ratio<=0.72) or (ratio<=0.16) or (ratio>0.27 and ratio<=0.4) or (ratio>=2) or (ratio>=1 and ratio<=1.6)):
       if ((ratio2<=0.231) or (0.5<=ratio2 and ratio2<=0.54) or (0.86<=ratio2 and ratio2<=2)):
           print(title,'Head is front')
       elif (0.54<ratio2 and ratio2<=0.86):
           print(title,'Head is tilt in left direction')
       elif (ratio2>2) or (ratio2>0.231 and ratio2<0.5):
           print(title,'Head is tilt in right direction')              
  else:

## Head position detection
# Calculate a ratio(Upper/Bottom) and ratio(Left/Right)
          dCU=(extTop[0]-center[0])
          dCB=(extBot[0]-center[0])
          dCL=(extLeft[1]-center[1])
          dCR=(extRight[1]-center[1])

          if dCU==0 or dCB==0:
            Cratio=0
          else:
            Cratio=np.abs(dCU/dCB)
            
          if dCL==0 or dCR==0:
            CLR=0
          else:
            CLR=np.abs(dCL/dCR)
          
          title='%dth'%J
          if ((Cratio>=0.07 and Cratio<0.17) or (Cratio>0.193 and Cratio<=0.3) or (0.33<Cratio and Cratio<0.4) or (0.5<=Cratio and Cratio<=0.6) or (Cratio>=0.8 and Cratio<1.3) or (Cratio>=2)):
            if ((CLR>0.27 and CLR<0.4) or (CLR>=0.6 and CLR<=0.65)):
               print(title,'Head is tilt in left direction')
            elif ((CLR<=0.27) or (CLR>=0.4 and CLR<=0.44) or (CLR>=0.57 and CLR<0.6) or (0.81<CLR and CLR<=0.82) or (CLR>1)):
               print(title,'Head is Front')
            elif ((0.44<CLR and CLR<0.57) or (0.65<CLR and CLR<=0.81) or (0.82<=CLR and CLR<=1)):
               print(title,'Head is tilt in right direction')
               
          elif((Cratio<=0.07) or (Cratio>=0.4 and Cratio<=0.47) or (Cratio>0.7 and Cratio<=0.84)):
            print(title,'head is up, please check other measurements')
        
          elif((Cratio>0.17 and Cratio<=0.193) or (0.3<Cratio and Cratio<=0.33) or (0.47<Cratio and Cratio<=0.5) or (0.6<Cratio and Cratio<=0.7) or (1.3<=Cratio and Cratio<2)):
            print(title,'head is down, please check other measurements')
