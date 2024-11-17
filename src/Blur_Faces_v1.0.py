# Fadil Eldin
# Aug/2021
# https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/

import time
import os
import sys
# %pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#---------------------
import face_recognition
# To install face_recognition run
# pip install cmake
# pip install dlib
# if fails then https://stackoverflow.com/questions/41912372/dlib-installation-on-windows-10
# i.e.
# Install Dlib from .whl
# Dlib 19.7.0
# pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f
# You can test it, downloading an example from the site, for example SVM_Binary_Classifier.py and running it on your machine.
# Read Here
# https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f
# Note: if this message occurs you have to build dlib from source:
# dlib-19.7.0-cp36-cp36m-win_amd64.whl is not a supported wheel on this platform
# Install Dlib from source (If the solution above doesn't work)
# Windows Dlib > 19.7.0
# Download the CMake installer and install it: https://cmake.org/download/
#----------------------------------
# Add CMake executable path to the Enviroment Variables:
# set PATH="%PATH%;C:\Program Files\CMake\bin"
# note: The path of the executable could be different from C:\Program Files\CMake\bin, just set the PATH accordingly.
# note: The path will be set temporarily, to make the change permanent you have to set it in the “Advanced system settings” → “Environment Variables” tab.
# Restart The Cmd or PowerShell window for changes to take effect.
# Download the Dlib source(.tar.gz) from the Python Package Index : https://pypi.org/project/dlib/#files extract it and enter into the folder.
# Check the Python version: python -V. This is my output: Python 3.7.2 so I'm installing it for Python3.x and not for Python2.x
# note: You can install it for both Python 2 and Python 3, if you have set different variables for different binaries i.e: python2 -V, python3 -V
# Run the installation: python setup.py install
#------------------------------------
import cv2
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

cwd = os.getcwd()
print(f"cwd {cwd}")

out_path=os.path.abspath(os.path.join(cwd, '../output'))
config_path=os.path.abspath(os.path.join(cwd,'../config'))
input_path=os.path.abspath(os.path.join(cwd,'../input'))
#------------------------------------------------------------------------------
def blur_face_from_camera():   # WORKING
    # From Camera Every Frame

    # Initialize some variables
    face_locations = []
    face_encodings = []
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        #rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = frame[:, :, ::-1]

        # could also use haar cascade
        # https://www.analyticsvidhya.com/blog/2019/03/opencv-functions-computer-vision-python/
        face_locations = face_recognition.face_locations(rgb_small_frame)
        #print('Face Locations {}'.format(face_locations))
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            #top *= 4
            #right *= 4
            #bottom *= 4
            #left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # extract the face ROI
            face = frame[top:bottom, left:right]
            #mgplot = plt.imshow(face)
            #plt.show()
            #----------- GaussianBlur --------------------------------
            face = cv2.GaussianBlur(face, (23, 23), 30)
            # impose this blurred image on original image to get final image
            frame[top:bottom, left:right] = face
            #----------- OR pixelated  --------------------------------
            #(B, G, R) = [int(x) for x in cv2.mean(face)[:3]]
            #cv2.rectangle(frame, (left, top), (right, bottom), (B, G, R), -1)


            time.sleep(0.1)
        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle
    video_capture.release()
    cv2.destroyAllWindows()

    print('Done blur_face_from_camera')
    return
#------------------------------------------------------------------------------
def HAAR_blur_face_from_camera():
    # https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html
    xml=os.path.join(config_path,'haarcascade_frontalface_alt.xml')
    face_detect = cv2.CascadeClassifier(xml)
    outfilename=os.path.join(out_path, 'blurred-from-camera.mp4')
    if os.path.exists(outfilename):
        print('Output exists, deleting')
        os.remove(outfilename)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outfilename, fourcc, 20.0, (640, 480))

    # First download xml from here
    # https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    # Tutorial
    # https://towardsdatascience.com/computer-vision-detecting-objects-using-haar-cascade-classifier-4585472829a9
    # From Camera Every Frame
    #video_capture = cv2.VideoCapture(0)
    #https://stackoverflow.com/questions/60007427/cv2-warn0-global-cap-msmf-cpp-674-sourcereadercbsourcereadercb-termina
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_data = face_detect.detectMultiScale(gray, 1.04, 4)
        #print('Face face_data {}'.format(face_data))

        # Draw rectangle around the faces which is our region of interest (ROI)
        for (x, y, w, h) in face_data:
            #print('x y {} {}'.format(x,y))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = frame[y:y + h, x:x + w]
            # applying a gaussian blur over this new rectangle area
            face = cv2.GaussianBlur(face, (23, 23), 30)
            #face = cv2.GaussianBlur(face, (51, 51), 0)
            # impose this blurred image on original image to get final image
            frame[y:y + face.shape[0], x:x + face.shape[1]] = face

        # Write the frame to the output
        out.write(frame)
        # Display the resulting image
        cv2.imshow('Video', frame)
        #time.sleep(0.2)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

    print('Done HAAR_blur_face_from_camera')
    return
#------------------------------------------------------------------------------
def HAAR_blur_face_from_video_file(source):
    # https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html
    xml=os.path.join(config_path,'haarcascade_frontalface_alt.xml')
    face_detect = cv2.CascadeClassifier(xml)
    basename = os.path.basename(source)
    basenamenoextension = os.path.splitext(basename)[0]
    outfilename = os.path.join(out_path, 'blurred_'+basenamenoextension+'.mp4')
    if os.path.exists(outfilename):
        print('Output exists, deleting')
        os.remove(outfilename)
    props = get_video_properties(source)
    #duration = props["duration"]
    fps = props["fps"]
    frames = props["frames"]
    width = int(props["width"])
    height = int(props["height"])

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outfilename, fourcc, fps, (width, height))

    video_capture = cv2.VideoCapture(source)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_data = face_detect.detectMultiScale(gray, 1.04, 4)
        #print('Face face_data {}'.format(face_data))

        # Draw rectangle around the faces which is our region of interest (ROI)
        for (x, y, w, h) in face_data:
            #print('x y {} {}'.format(x,y))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = frame[y:y + h, x:x + w]
            # applying a gaussian blur over this new rectangle area
            face = cv2.GaussianBlur(face, (23, 23), 30)
            #face = cv2.GaussianBlur(face, (51, 51), 0)
            # impose this blurred image on original image to get final image
            frame[y:y + face.shape[0], x:x + face.shape[1]] = face

        # Write the frame to the output
        out.write(frame)
        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

    print('Done HAAR_blur_face_from_video_file')
    return
#------------------------------------------------------------------------------
def silent_mark(source, outfilename,  AllMarkXes):
    if os.path.exists(outfilename):
        os.remove(outfilename)
    props = get_video_properties(source)
    duration = props["duration"]
    fps = props["fps"]
    frames = props["frames"]
    width = props["width"]
    height = props["height"]
    height_of_bar = int(height / 30)
    Y=height-height_of_bar
    thickness = 4
    frame_number = 0
    bg_color = (20, 20, 0)
    # https://www.rapidtables.com/web/color/RGB_Color.html
    # GET RGB then put them as BRG
    marker_color = (0,255, 255)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outfilename, fourcc, fps, (width, height))

    video_capture = cv2.VideoCapture(source)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # https://www.programcreek.com/python/example/110706/cv2.drawMarker
        for markX in AllMarkXes:
            #frame=cv2.rectangle(frame, (0, Y), (width, Y), color=bg_color, thickness=50)
            # Marker types https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html
            # line types: https://docs.opencv.org/3.4/d0/de1/group__core.html#gaf076ef45de481ac96e0ab3dc2c29a777
            frame2=cv2.drawMarker(frame, (markX,Y),
                                 color=marker_color,
                                 markerType=cv2.MARKER_TRIANGLE_UP,
                                 thickness=15,
                                 line_type=4)
        out.write(frame2)
    return
# ------------------------------------------------------------------------------
def get_startx_endx(source, marker_at_msec):
    import math
    props = get_video_properties(source)
    duration = props["duration"]
    fps = props["fps"]
    frames = props["frames"]
    width = props["width"]
    height = props["height"]
    video_capture = cv2.VideoCapture(source)
    markX=-1
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        pos_msec=math.ceil(video_capture.get(cv2.CAP_PROP_POS_MSEC))
        frame_number=int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        if pos_msec==0 and frame_number>1:
            continue
        gg=(frame_number / frames) * width
        frame_x = round((frame_number / frames) * width)
        # print(f"pos_msec={pos_msec} frame_x:{frame_x} gg:{gg}")
        # Millisecond counter not Consecutive, looking for a range match instant exact match
        if pos_msec in range(marker_at_msec-200,marker_at_msec+200):
            markX = frame_x

    # Release handle
    video_capture.release()
    cv2.destroyAllWindows()
    return markX
#------------------------------------------------------------------------------
def markers_on_progress_bar(source,  marker_msec_list):
    if not os.path.exists(source):
        print(f"{bcolors.FAIL}Error : File not found {source} ! {bcolors.ENDC}")
        return

    basename = os.path.basename(source)
    basenamenoextension = os.path.splitext(basename)[0]
    outfilename = os.path.join(video_path, 'markers_' + basenamenoextension + '.mp4')
    if os.path.exists(outfilename):
        os.remove(outfilename)

    AllMarkXes=[]
    # https://karobben.github.io/2021/06/01/Python/opencv-progressbar/
    for at_msec in marker_msec_list:
        markX=get_startx_endx(source, at_msec)
        print(f"Second:{at_msec} markX:{markX}")
        AllMarkXes.append(markX)

    silent_mark(source, outfilename, AllMarkXes)
    # Jump https://stackoverflow.com/questions/57198119/how-to-play-a-video-file-from-a-particular-timstamp-in-opencv-using-python
    # https://pypi.org/project/moviepy/
    print('Done markers_on_movie')
    return outfilename
#------------------------------------------------------------------------------
def get_video_properties(source):
    if not os.path.exists(source):
        print(f"{bcolors.FAIL}Error : File not found {source} ! {bcolors.ENDC}")
        return

    import cv2
    import datetime

    video = cv2.VideoCapture(source)
    # count the number of frames
    frames=video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps=int(video.get(cv2.CAP_PROP_FPS))

    # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    seconds = int(frames / fps)
    duration = str(datetime.timedelta(seconds=seconds))
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_props = {
        "duration": duration,
        "frames": frames,
        "fps": fps,
        "height":height,
        "width":width
    }
    print(f"duration:{duration} frames:{frames} fps:{fps} height:{height} width:{width}")
    return video_props
#------------------------------------------------------------------------------
if __name__ == "__main__":

    blur_face_from_camera()
    HAAR_blur_face_from_camera()
    exit(0)

    source = os.path.join(input_path, 'Faces.mp4')
    if not os.path.exists(source):
        print(f"File does not exist:{source}")
        exit(1)

    props=get_video_properties(source)
    print('Width:{} Hight:{}'.format(props["width"],props["height"]))

    HAAR_blur_face_from_video_file(source)

    # Put markers on movie at different seconds
    #outfile=markers_on_progress_bar(source, [3500,12780,16970])
# The End
#------------------------------------------------------------------------------
# Later
    #import vlc
    # https://www.geeksforgeeks.org/vlc-module-in-python-an-introduction/
    # Note: In order to use the vlc module in python, the user system
    # should have vlc media player already installed on the machine.
    # https://get.videolan.org/vlc/3.0.16/win64/vlc-3.0.16-win64.exe
    # Timeline Markers
    # https://docs.blender.org/api/current/bpy.types.TimelineMarkers.html#bpy.types.TimelineMarkers