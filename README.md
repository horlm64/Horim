# Head Segmentation & Position Detection

Project : Segmentation of human heads, using a commercial low-cost camera combined with a Raspberry Pi
 
This code is implemented, based on the Raspberry Pi and Raspberry Pi Camera.

it is divided into head segmentation and head position detection 

One is head segmentation from images of Raspberry Pi Camera
- Region-based color segmentation: Superpixel algorithm, multiple thresholding, skin-color classification.
- Contour detection to obtain the head among other contours by comparison.
- Contour approximation to smooth contour as much as possible and update a Mask, obtained by contour detection
- Apply the updated Mask to the given image from Raspberry Pi Camera.

Another is head position detection, based on segmented face
- When head face detection doesn't work: set the criteria, based on the coordinates, extracted from contour detection.
- Once Haar cascade face detection recognizes the face, perform facial landmark detection to set the criteria
- Perform Head position detection 

You can request me for more details : horim64@naver.com
