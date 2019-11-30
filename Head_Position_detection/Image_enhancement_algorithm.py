import cv2
import os
import numpy as np
import math
import dlib
from skimage.segmentation import slic # sckit-image implementation
from skimage.color import label2rgb
from skimage.future import graph

#The algorithm is executed via 4 stages.
#1.Image enhancement
#2.Region-based skin colour detection
#3.Contour detection
#4.Facial landmark detection
#5.Head position detection

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
  path=r'/home/pi/Desktop/TEST/Horim'
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

##1.Image enhancement

# Call the function to Load
Load() # Load the image

for J in range(len(Img)):
  hist,bins = np.histogram(Img[J].flatten(),256,[0,256])

  cdf = hist.cumsum()
  cdf_normalized = cdf * hist.max()/ cdf.max()
  cdf_m = np.ma.masked_equal(cdf,0)
  cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
  cdf = np.ma.filled(cdf_m,0).astype('uint8')
  Img[J] = cdf[Img[J]]
  Data1 =Img[J]

## Region-based skin colour detection
# Make a temporary mask
  Mask=cv2.cvtColor(Data1,cv2.COLOR_BGR2GRAY) # Convert the image in gray-scale
  ret,Mask=cv2.threshold(Mask,10,255,cv2.THRESH_BINARY) # Thresholding to binarize the image
  Mask[:]=0 # reset the Mask 

# SLIC Segmentation with number of segment: 250 , Sigma =3, Compactness : 30
  segment_slic = slic(Data1,n_segments=250,sigma=3,compactness=30) # SLIC Segmentation
# Calculate the mean-color, corresponding to the original image and result of the SLIC Segmentation.
  g = graph.rag_mean_color(Data1,segment_slic) 
# Use the pre-defined threshold to cut off the connection of the region and group the regions, having the value less than 17
  label=graph.cut_threshold(segment_slic,g,17)
# Convert the Label(contains the pixel-info) into the RGB Color space
  rgb=label2rgb(label,Data1,kind='avg')
# Construct the channels,with extracted the normalised RGB & HSV Channels
  [x,y]=segment_slic.shape
  seg=np.zeros((x,y,3),np.float32)
  r=rgb[:,:,0] #Red channel
  g=rgb[:,:,1] #Green channel
  b=rgb[:,:,2] #Blue channel
  HSV=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)  
  summ=b+g+r #Summation of red, bule, and green channel
  seg[:,:,1]=g/summ #Normalised green channel
  seg[:,:,0]=b/summ #Normalised blue channel

#Additional Variable to save the result of the Loop
  sldx=[] # save indexs of the skin
  
# Skin colour classification
# Explictly defined skin boundary model: ratio of the normalised green channel and blue channel & HSV Filter
  for z in range(0,x):
   for k in range(0,y):
    if not(((seg[z,k,0]/seg[z,k,1]<=2.3) and (seg[z,k,0]/seg[z,k,1]>=1.095)) and(HSV[z,k,0]<=23)or(161<=HSV[z,k,0])): #previous >=161 and <=18
           continue 
    else:
      sldx.append([z,k]) # Skin region index

  tmp=np.zeros([x,y],np.uint8) # to save the skin part
  
# Using the index, extract the skin region
  for i in range(len(sldx)):
    tmp[sldx[i][0],sldx[i][1]]=(segment_slic[sldx[i][0],sldx[i][1]])

# According to the value of the label, compute the mask as 255 (white) and 0 (black)
  for i in range(tmp.shape[0]):
    for j in range(tmp.shape[1]):
      if (tmp[i,j]>=30) and (tmp[i,j]<=135): #30
        Mask[i,j]=255
      else:
        Mask[i,j]=0
  image= cv2.bitwise_and(Data1,Data1,mask=Mask)

## Perform Contour detection 
# Find the contour
  _, contours, _ = cv2.findContours(Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  Area=[] # to save the area
  hull=[] # to save the calculated coordinates

# Calculate the Area of the contours and save them in a list
  for I in range(len(contours)):
    Area.append(cv2.contourArea(contours[I]))
  Max=max(Area) # to isolate the larger region

# Find the largest area of the contour, among them
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
    
# Perform the contour approximation with given outer to obatin a outline
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
    
# Get the coordinates & radius, using the contour
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

# Detect face in the image  
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
   dL=(Le[0][1]-p4[1]) # From center to left side
   dR=(Ri[0][1]-p4[1]) # From center to right side
   if dL==0 or dR==0:
       ratio2=0
   else:
       ratio2=np.abs(dL/dR)
       
# Calculate the ratio of the Upper/Bottom to classify up or down
   dU=((p4[0]-p3[0])) # From center to upper
   dB=((p4[0]-p2[0])) # From center to bottom
   ratio=np.abs(dU/dB)
   
# Perform the head position detection
# Classify the head position in Up / Down / Front
# Front into Left / Right

   title='detected %dth'%J
   if ((ratio>0.5 and ratio<=0.55) or (ratio>=0.67 and ratio<=0.73) or (ratio>2 and ratio<=3.5)):
       print(title,'Head is down and check other measurements')
   if (ratio>0.55 and ratio<0.6) or (ratio>0.73 and ratio<1.0):
       print(title,'Head is Up and check other measurements')
   if ((ratio<=0.5) or (ratio>3.5) or (ratio>=0.6 and ratio<0.67) or (ratio>=1.0 and ratio<=2.0)):
       if ((0.3<=ratio2 and ratio2<=0.38) or (0.4<=ratio2 and ratio2<=0.5) or (ratio2<=0.25) or (ratio2>3.5)):
           print(title,'Head is front')
       elif ((ratio2<0.3 and ratio2>0.25) or (ratio2>0.38 and ratio2<0.4)):
           print(title,'Head is tilt in left direction')
       elif (ratio2>0.5 and ratio2<=3.5):
           print(title,'Head is tilt in right direction')              
  else:
  
## Head position detection
# Calculate a ratio(Upper/Bottom) and ratio(Left/Right)
          dCU=(extTop[0]-center[0])
          dCB=(extBot[0]-center[0])
          dCL=(extLeft[1]-center[1])
          dCR=(extRight[1]-center[1])
          CLR=np.abs(dCL/dCR)
          Cratio=np.abs(dCU/dCB)
          title='%dth'%J
          if ((0.43<Cratio and Cratio<=0.48) or (Cratio<=0.1) or ((Cratio>0.9)and(Cratio<=1.03)) or (Cratio>0.27 and Cratio<=0.35) or (Cratio>2.4 and Cratio<3) or (Cratio>5 and Cratio<=6) or (Cratio>0.59 and Cratio<=0.67)):
            if (CLR>=0.76 and CLR<0.82):
               print(title,'Head is tilt in left direction')
            elif ((0.82<=CLR and CLR<1) or (CLR>=0.3 and CLR<0.76)):
               print(title,'Head is Front')
            elif (CLR>=1 or CLR<0.3):
               print(title,'Head is tilt in right direction')
               
          elif ((Cratio>0.1 and Cratio<=0.23) or (Cratio>0.35 and Cratio<=0.37) or (Cratio>6) or (Cratio>0.48 and Cratio<=0.52) or (Cratio>1.03 and Cratio<=2) or (Cratio>0.75 and Cratio<=0.9) or (Cratio>=3 and Cratio<5)):
            print(title,'head is up, please check other measurements')
        
          elif((Cratio>0.23 and Cratio<=0.27) or (Cratio>0.37 and Cratio<=0.43) or (Cratio>0.52 and Cratio<=0.59) or (Cratio>0.68 and Cratio<=0.75) or (Cratio>2 and Cratio<=2.4)):
            print(title,'head is down, please check other measurements')
