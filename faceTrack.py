#!/usr/bin/env python
#Below we are importing functionality to our Code, OPEN-CV, Time, and Pimoroni Pan Tilt Hat Package of particular note.
import cv2, sys, time, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import dlib
from pantilthat import *

# Set up the the faceDetector, shapePredictor and faceRecognizer to
# optimize speed as they are nedded by both functions.
faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
faceRecognizer = dlib.face_recognition_model_v1(
    "dlib_face_recognition_resnet_model_v1.dat"
)

mpl.use('tkagg')

def inrole_data(faceDetector, shapePredictor, faceRecognizer):
    """This function creates a face descriptors of (1x128) for each face
    in the images and stores them in in a NumPy array. It also creates a dictionary
    to store the the index of array to names of the celebrity also in NumPy array.

    Args:
        faceDetector Dlib: used to detect faces in an image
        shapePredictor Dlib: identifies the locations of import facial landmarks
        faceRecognizer Dlib:  maps human faces into 128D vectors
    """

    # create a dictionary to uses as a index for each face descriptors to celebrity name.
    index = {}
    i = 0
    # create a NumPy array to store face descriptors
    faceDescriptors = None

    # loop though the images in folders
    for images in os.listdir("celeb_mini"):
        imagefiles = os.listdir(os.path.join("celeb_mini", images))
        for image in imagefiles:
            imagePath = os.path.join("celeb_mini", images, image)

            #  read each image and convert to a format form Dlib
            img = cv2.imread(imagePath)
            imDli = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # look for faces in image
            faces = faceDetector(imDli)

            # Create descriptor and index for each image
            for face in faces:

                # Find facial landmarks for each detected face
                shape = shapePredictor(imDli, face)

                # Compute face descriptor using neural network defined in Dlib.
                faceDescriptor = faceRecognizer.compute_face_descriptor(img, shape)

                # Convert face descriptor from Dlib's format to list, then a NumPy array
                faceDescriptorList = [x for x in faceDescriptor]
                faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
                faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

                # add face descriptors to the faceDescriptor Numpy array.
                if faceDescriptors is None:
                    faceDescriptors = faceDescriptorNdarray
                else:
                    faceDescriptors = np.concatenate(
                        (faceDescriptors, faceDescriptorNdarray), axis=0
                    )

                # map celebrity name corresponding to face descriptors and stored in NumPy Array
                index[i] = np.load("celeb_mapping.npy", allow_pickle=True).item()[
                    images
                ]
                i += 1
    # save
    np.save("index.npy", index)
    np.save("faceDescriptors.npy", faceDescriptors)

# to save time and not recreate the descriptors skip if the folder and index already exist
if not os.path.exists("index.npy") or not os.path.exists("faceDescriptors.npy"):
    print("building face descriptors")
    inrole_data(faceDetector, shapePredictor, faceRecognizer)


# Load the BCM V4l2 driver for /dev/video0. This driver has been installed from earlier terminal commands. 
#This is really just to ensure everything is as it should be.
os.system('sudo modprobe bcm2835-v4l2')
# Set the framerate (not sure this does anything! But you can change the number after | -p | to allegedly increase or decrease the framerate).
os.system('v4l2-ctl -p 40')

# Frame Size. Smaller is faster, but less accurate.
# Wide and short is better, since moving your head up and down is harder to do.
# W = 160 and H = 100 are good settings if you are using and earlier Raspberry Pi Version.
FRAME_W = 320
FRAME_H = 200

# Default Pan/Tilt for the camera in degrees. I have set it up to roughly point at my face location when it starts the code.
# Camera range is from 0 to 180. Alter the values below to determine the starting point for your pan and tilt.
cam_pan = 40
cam_tilt = 20

# Set up the Cascade Classifier for face tracking. This is using the Haar Cascade face recognition method with LBP = Local Binary Patterns. 
# Seen below is commented out the slower method to get face tracking done using only the HAAR method.
cascPath = 'haarcascade_frontalface_default.xml' # sys.argv[1]
# cascPath = '/usr/share/opencv/lbpcascades/lbpcascade_frontalface.xml'
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

# Start and set up the video capture with our selected frame size. Make sure these values match the same width and height values that you choose at the start.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200);
time.sleep(2)

# Turn the camera to the Start position (the data that pan() and tilt() functions expect to see are any numbers between -90 to 90 degrees).
pan(cam_pan-90)
tilt(cam_tilt-90)
light_mode(WS2812)

# Light control down here. If you have a LED stick wired up to the Pimoroni HAT it will light up when it has located a face.
def lights(r,g,b,w):
    for x in range(18):
        set_pixel_rgbw(x,r if x in [3,4] else 0,g if x in [3,4] else 0,b,w if x in [0,1,6,7] else 0)
    show()

lights(0,0,0,50)

 # load face descriptors and the names index
faceDescriptors = np.load("faceDescriptors.npy")
index = np.load("index.npy", allow_pickle="TRUE").item()

#Below we are creating an infinite loop, the system will run forever or until we manually tell it to stop (or use the "q" button on our keyboard)
while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    # This line lets you mount the camera the "right" way up, with neopixels above
    frame = cv2.flip(frame, -1)
    
    if ret == False:
      print("Error getting image")
      continue

    # Convert to greyscale for easier+faster+accurate face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist( gray )

    # Do face detection to search for faces from these captures frames
    faces = faceCascade.detectMultiScale(frame, 1.1, 3, 0, (10, 10))
   
    # Slower method (this gets used only if the slower HAAR method was uncommented above. 
    '''faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 20),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE | cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv2.cv.CV_HAAR_DO_ROUGH_SEARCH
    )'''
    
    lights(50 if len(faces) == 0 else 0, 50 if len(faces) > 0 else 0,0,50)

    #Below draws the rectangle onto the screen then determines how to move the camera module so that the face can always be in the centre of screen. 

    for (x, y, w, h) in faces:
        # Draw a green rectangle around the face (There is a lot of control to be had here, for example If you want a bigger border change 4 to 8)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

        # Track face with the square around it
        
        # Get the centre of the face
        x = x + (w/2)
        y = y + (h/2)

        # Correct relative to centre of image
        turn_x  = float(x - (FRAME_W/2))
        turn_y  = float(y - (FRAME_H/2))

        # Convert to percentage offset
        turn_x  /= float(FRAME_W/2)
        turn_y  /= float(FRAME_H/2)

        # Scale offset to degrees (that 2.5 value below acts like the Proportional factor in PID)
        turn_x   *= 2.5 # VFOV
        turn_y   *= 2.5 # HFOV
        cam_pan  += -turn_x
        cam_tilt += turn_y

        print(cam_pan-90, cam_tilt-90)

        # Clamp Pan/Tilt to 0 to 180 degrees
        cam_pan = max(0,min(180,cam_pan))
        cam_tilt = max(0,min(180,cam_tilt))

        # Update the servos
        pan(int(cam_pan-90))
        tilt(int(cam_tilt-90))

        break
    
    #Orientate the frame so you can see it.
    frame = cv2.resize(frame, (540,300))
    frame = cv2.flip(frame, 1)
   
    # Display the video captured, with rectangles overlayed
    # onto the Pi desktop 
    cv2.imshow('Video', frame)

    #If you type q at any point this will end the loop and thus end the code.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    key = cv2.waitKey(20)
    if key == 13:
        cv2.destroyWindow("Video")
        if ret:
            imDli = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # detect faces
            faces = faceDetector(imDli)

            # create descriptor and index for each image
            for face in faces:
                shape = shapePredictor(imDli, face)

                faceDescriptor = faceRecognizer.compute_face_descriptor(frame, shape)

                faceDescriptorList = [x for x in faceDescriptor]
                faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
                faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

                # calculate the distances of the new face wiht face descriptors of celebrities
                distances = np.linalg.norm(faceDescriptors - faceDescriptorNdarray, axis=1)
                argmin = np.argmin(distances)
                minDistance = distances[argmin]

                # find an acceptable lookalike
                if minDistance <= 0.8:
                    label = index[argmin]
                else:
                    label = "unknown"

                celeb_name = label

                # load celebrity images from celeb_mini folder
                for images in os.listdir("celeb_mini"):
                    imagefiles = os.listdir(os.path.join("celeb_mini", images))

                    if (
                        np.load("celeb_mapping.npy", allow_pickle=True).item()[images]
                        == celeb_name
                    ):
                        for im in imagefiles:
                            img_cele = cv2.imread(os.path.join("celeb_mini", images, im))
                            img_cele = cv2.cvtColor(img_cele, cv2.COLOR_BGR2RGB)
                            break

            # show images one at a time
            plt.subplot(121)
            plt.imshow(imDli)
            plt.title("test img")

            plt.subplot(122)
            plt.imshow(img_cele)
            plt.title("Celeb Look-Alike={}".format(celeb_name))
            plt.show()
            break

# When everything is done, release the capture information and stop everything
video_capture.release()
cv2.destroyAllWindows()
