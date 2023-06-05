import cv2
import dlib
import time
import math
import numpy as np
import os
import face_recognition
import winsound
import datetime
import imutils
from pushbullet import Pushbullet

l=[]

#sending notifications for detected anomalies
def notif(msg):
    ph_key="o.lXdMmcLOmjD6dAodbbFy2r4fQureaDuD"
    pb=Pushbullet(ph_key)
    phone = pb.devices[0]
    pb.push_sms(phone, "+917338935190",msg)

#face recognition to detect unknown faces



    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    tharun_image = face_recognition.load_image_file("known_faces\\tharun.jpg")
    tharun_face_encoding = face_recognition.face_encodings(tharun_image)[0]

    # Load a second sample picture and learn how to recognize it.
    karthick_image = face_recognition.load_image_file("known_faces\karthick.jpg")
    karthick_face_encoding = face_recognition.face_encodings(karthick_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        tharun_face_encoding,
        karthick_face_encoding
    ]
    known_face_names = [
        "tharun",
        "karthick"
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()







video_path = '1.mp4'
cap = cv2.VideoCapture(video_path)









def classify_day_night(frame):
        # Convert the frame to graysca le
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the average brightness of the frame
        average_brightness = cv2.mean(gray_frame)[0]
    
        # Define a threshold value to classify day and night
        threshold = 100

        # Classify the frame as day or night based on the average brightness  
        if average_brightness > threshold:
            l.append("Day")
        else:
            l.append("Night")

    # Open the video file 


while cap.isOpened():
        # Read the next frame from the video
    ret, frame = cap.read()

        # Check if frame is read successfully
    if not ret:
        break

        # Classify the frame as day or night
    classify_day_night(frame)
        # print("Frame classification:", classification)
        # Display the frame
    cv2.imshow('Video', frame)

        # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
if(l.count("Day")>l.count("Night")):
    k='Day'
else:
    k='Night'

cap.release()
cv2.destroyAllWindows()

if(k=='Night'):

    
    #Theft detection
    
    p=0
    # Constants for motion detection
    MIN_AREA = 500  # Minimum area for considering motion
    THRESHOLD_SENSITIVITY = 30  # Sensitivity threshold for detecting motion

    # Create video capture object (use 0 for webcam)
    cap = cv2.VideoCapture('1.mp4')

    # Read the first frame for initialization
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Failed to read video file")

    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Loop over the video frames
    while cap.isOpened():
        # Read the current frame
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Convert the current frame to grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between the current and previous frames
        frame_diff = cv2.absdiff(curr_gray, prev_gray)

        # Apply thresholding to highlight the regions with significant differences
        _, thresh = cv2.threshold(frame_diff, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)

        # Perform morphological operations to remove noise and fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=2)

        # Find contours of the remaining regions
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check for motion by analyzing each contour
        for contour in contours:
            if cv2.contourArea(contour) > MIN_AREA:
                # Detected significant motion (potential theft)
                cv2.drawContours(curr_frame, [contour], -1, (0, 0, 255), 2)
                p+=1
                

        # Display the current frame with motion detection
        cv2.imshow('Theft Detection', curr_frame)
        if(p>100):
            print('Motion detected at night Alert Needed')
            notif('Motion detected at night Alert Needed')
            break

        # Update the previous frame
        prev_gray = curr_gray

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
    
    #car overspeeding





    #Classifier File
    carCascade = cv2.CascadeClassifier("vech.xml")

    #Video file capture
    video = cv2.VideoCapture("1.mp4")

    # Constant Declaration
    WIDTH =1280
    HEIGHT = 720

    #estimate speed function
    def estimateSpeed(location1, location2):
        d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
        ppm = 8.8
        d_meters = d_pixels / ppm
        fps = 18
        speed = d_meters * fps * 3.6
        return speed

    #tracking multiple objects
    def trackMultipleObjects():
        rectangleColor = (0, 255, 255)
        frameCounter = 0
        currentCarID = 0
        fps = 0

        carTracker = {}
        carNumbers = {}
        carLocation1 = {}
        carLocation2 = {}
        speed = [None] * 1000


        out = cv2.VideoWriter('outTraffic.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH, HEIGHT))

        while True:
            start_time = time.time()
            rc, image = video.read()
            if type(image) == type(None):
                break

            image = cv2.resize(image, (WIDTH, HEIGHT))
            resultImage = image.copy()

            frameCounter = frameCounter + 1
            carIDtoDelete = []

            for carID in carTracker.keys():
                trackingQuality = carTracker[carID].update(image)

                if trackingQuality < 7:
                    carIDtoDelete.append(carID)

            
            for carID in carIDtoDelete:
                print("Removing carID " + str(carID) + ' from list of trackers. ')
                print("Removing carID " + str(carID) + ' previous location. ')
                print("Removing carID " + str(carID) + ' current location. ')
                carTracker.pop(carID, None)
                carLocation1.pop(carID, None)
                carLocation2.pop(carID, None)

            
            if not (frameCounter % 10):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

                for (_x, _y, _w, _h) in cars:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)

                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h

                    matchCarID = None

                    for carID in carTracker.keys():
                        trackedPosition = carTracker[carID].get_position()

                        t_x = int(trackedPosition.left())
                        t_y = int(trackedPosition.top())
                        t_w = int(trackedPosition.width())
                        t_h = int(trackedPosition.height())

                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                            matchCarID = carID

                    if matchCarID is None:
                        # print(' Creating new tracker' + str(currentCarID))

                        tracker = dlib.correlation_tracker()
                        tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                        carTracker[currentCarID] = tracker
                        carLocation1[currentCarID] = [x, y, w, h]

                        currentCarID = currentCarID + 1

            for carID in carTracker.keys():
                trackedPosition = carTracker[carID].get_position()

                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())

                cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

                carLocation2[carID] = [t_x, t_y, t_w, t_h]

            end_time = time.time()

            if not (end_time == start_time):
                fps = 1.0/(end_time - start_time)

            for i in carLocation1.keys():
                if frameCounter % 1 == 0:
                    [x1, y1, w1, h1] = carLocation1[i]
                    [x2, y2, w2, h2] = carLocation2[i]

                    carLocation1[i] = [x2, y2, w2, h2]

                    if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                        if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                            speed[i] = estimateSpeed([x1, y1, w1, h1], [x1, y2, w2, h2])

                        if speed[i] != None and y1 >= 180:
                            if int(speed[i])>20:
                                #cv2.putText(resultImage, str(int(speed[i])) + "km/h", (int(x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 100) ,2)
                                cv2.putText(resultImage, str(int(speed[i])) + "km/h overspeeding", (int(x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 100) ,2)
            cv2.imshow('result', resultImage)

            out.write(resultImage)

            if cv2.waitKey(1)==27:
                break

        
        cv2.destroyAllWindows()
        out.release()

    if __name__ == '__main__':
        trackMultipleObjects()


    #gun detection
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 100  # Set Duration To 1000 ms == 1 second

    gun_cascade = cv2.CascadeClassifier('cascade.xml')
    camera = cv2.VideoCapture('data/gun.mp4')

    # initialize the first frame in the video stream
    firstFrame = None

    # loop over the frames of the video

    gun_exist = False

    while True:
        (grabbed, frame) = camera.read()

        # if the frame could not be grabbed, then we have reached the end of the video
        if not grabbed:
            break

        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        gun = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize = (100, 100))
        
        if len(gun) > 0:
            gun_exist = True
            winsound.Beep(frequency, duration)
            
        for (x,y,w,h) in gun:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]    

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        # draw the text and timestamp on the frame
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        key = cv2.waitKey(1) & 0xFF

    if gun_exist:
        print("Guns detected")
        notif("Guns detected Alert Needed")
        
    else:
        print("No guns detected")

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
