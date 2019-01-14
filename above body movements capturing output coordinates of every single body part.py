
# coding: utf-8

# In[16]:



# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
 
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
 
# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])
 
# initialize the first frame in the video stream
firstFrame = None


# In[17]:


# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	text = "Unoccupied"
 
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break
 
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue


# In[18]:


from imutils.video import VideoStream
from imutils.video import FPS
import datetime
import imutils
import time
import cv2
import os 

os.system("mkdir record")
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
fps = FPS().start()


i = 0
while i < 2000:
    frame = vs.read()
    if frame is None:
        print 'read none'
        continue
    timestamp = datetime.datetime.now()
    cv2.imwrite("record/%d.jpg" % i, frame)
    print i
    fps.update()
    i += 1

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
vs.stop()


# In[21]:


from imutils.video import VideoStream
from imutils.video import FPS
import datetime
import imutils
import time
import cv2
import os 

os.system("mkdir record")
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
fps = FPS().start()


i = 0
while i < 2000:
    frame = vs.read()
    if frame is None:
        continue
    timestamp = datetime.datetime.now()
    cv2.imwrite("record/%d.jpg" % i, frame)
    print i
    fps.update()
    i += 1

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
vs.stop()


# In[20]:


# import the necessary packages
import imutils
import cv2
 
class RGBHistogram:
	def __init__(self, bins):
		# store the number of bins the histogram will use
		self.bins = bins
 
	def describe(self, image):
		# compute a 3D histogram in the RGB colorspace,
		# then normalize the histogram so that images
		# with the same content, but either scaled larger
		# or smaller will have (roughly) the same histogram
		hist = cv2.calcHist([image], [0, 1, 2],
			None, self.bins, [0, 256, 0, 256, 0, 256])
 
		# normalize with OpenCV 2.4
		if imutils.is_cv2():
			hist = cv2.normalize(hist)
 
		# otherwise normalize with OpenCV 3+
		else:
			hist = cv2.normalize(hist,hist)
 
		# return out 3D histogram as a flattened array
		return hist.flatten()


# In[ ]:


# Pyhton program to implement  
# WebCam Motion Detector 
  
# importing OpenCV, time and Pandas library 
import cv2, time, pandas 
# importing datetime class from datetime library 
from datetime import datetime 
  
# Assigning our static_back to None 
static_back = None
  
# List when any moving object appear 
motion_list = [ None, None ] 
  
# Time of movement 
time = [] 
  
# Initializing DataFrame, one column is start  
# time and other column is end time 
df = pandas.DataFrame(columns = ["Start", "End"]) 
  
# Capturing video 
video = cv2.VideoCapture(0) 
  
# Infinite while loop to treat stack of image as video 
while True: 
    # Reading frame(image) from video 
    check, frame = video.read() 
  
    # Initializing motion = 0(no motion) 
    motion = 0
  
    # Converting color image to gray_scale image 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    # Converting gray scale image to GaussianBlur  
    # so that change can be find easily 
    gray = cv2.GaussianBlur(gray, (21, 21), 0) 
  
    # In first iteration we assign the value  
    # of static_back to our first frame 
    if static_back is None: 
        static_back = gray 
        continue
  
    # Difference between static background  
    # and current frame(which is GaussianBlur) 
    diff_frame = cv2.absdiff(static_back, gray) 
  
    # If change in between static background and 
    # current frame is greater than 30 it will show white color(255) 
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1] 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 
  
    # Finding contour of moving object 
    (_, cnts, _) = cv2.findContours(thresh_frame.copy(),  
                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
  
    for contour in cnts: 
        if cv2.contourArea(contour) < 10000: 
            continue
        motion = 1
  
        (x, y, w, h) = cv2.boundingRect(contour) 
        # making green rectangle arround the moving object 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 
  
    # Appending status of motion 
    motion_list.append(motion) 
  
    motion_list = motion_list[-2:] 
  
    # Appending Start time of motion 
    if motion_list[-1] == 1 and motion_list[-2] == 0: 
        time.append(datetime.now()) 
  
    # Appending End time of motion 
    if motion_list[-1] == 0 and motion_list[-2] == 1: 
        time.append(datetime.now()) 
  
    # Displaying image in gray_scale 
    cv2.imshow("Gray Frame", gray) 
  
    # Displaying the difference in currentframe to 
    # the staticframe(very first_frame) 
    cv2.imshow("Difference Frame", diff_frame) 
  
    # Displaying the black and white image in which if 
    # intencity difference greater than 30 it will appear white 
    cv2.imshow("Threshold Frame", thresh_frame) 
  
    # Displaying color frame with contour of motion of object 
    cv2.imshow("Color Frame", frame) 
  
    key = cv2.waitKey(1) 
    # if q entered whole process will stop 
    if key == ord('q'): 
        # if something is movingthen it append the end time of movement 
        if motion == 1: 
            time.append(datetime.now()) 
        break
  
# Appending time of motion in DataFrame 
for i in range(0, len(time), 2): 
    df = df.append({"Start":time[i], "End":time[i + 1]}, ignore_index = True) 
  
# Creating a csv file in which time of movements will be saved 
df.to_csv("Time_of_movements.csv") 
  
video.release() 
  
# Destroying all the windows 
cv2.destroyAllWindows()

