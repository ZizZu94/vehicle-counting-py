# ------------------------- created by: Zihadul Azam -------------------

# import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import vehicles # class file
import time

# /#/#/#/#/#/#/#/#/#/#/#/# <> Global Variables /#/#/#/#/#/#/#/#/#/#/#/#

input_video_path = 'samples/sample_3.mp4'
cap = cv2.VideoCapture(input_video_path)

ratio = .45
cnt_up = 0
cnt_down = 0

# colors
heavy_vehicle_color = (0, 165, 255)
normal_vehicle_color = (0, 255, 0)

# Get width and height of video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * ratio
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * ratio

frameArea = height * width
areaTH = frameArea / 400 # tracking height
type_of_vehicle_area_treshold = 10000

# Lines (four tracking lines)
up_limit = int(6 * (height / 10))
line_up = int(7 * (height / 10))
line_down = int(8 * (height / 10))
down_limit = int(9 * (height / 10))

line_down_color = (255, 0, 0)
line_up_color = (255, 0, 255)

# points of down line
pt1 = [0, line_down]
pt2 = [width, line_down]
pts_L1 = np.array([pt1, pt2], np.int32)
pts_L1 = pts_L1.reshape((-1, 1, 2))
# points of up line
pt3 = [0, line_up]
pt4 = [width, line_up]
pts_L2 = np.array([pt3, pt4], np.int32)
pts_L2 = pts_L2.reshape((-1, 1, 2))
# points of up limit line
pt5 = [0, up_limit]
pt6 = [width, up_limit]
pts_L3 = np.array([pt5, pt6], np.int32)
pts_L3 = pts_L3.reshape((-1, 1, 2))
# points of down limit line
pt7 = [0, down_limit]
pt8 = [width, down_limit]
pts_L4 = np.array([pt7, pt8], np.int32)
pts_L4 = pts_L4.reshape((-1, 1, 2))

# Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Kernal
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

font = cv2.FONT_HERSHEY_DUPLEX
my_vehicles = []
max_p_age = 5
pid = 1

# /#/#/#/#/#/#/#/#/#/#/#/# </> End Global Variables /#/#/#/#/#/#/#/#/#/#/#/#


# /#/#/#/#/#/#/#/#/#/#/#/# <> Start Frame Analysis /#/#/#/#/#/#/#/#/#/#/#/#
while(cap.isOpened()):
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't show the video. Exiting ...")
        break

    for i in my_vehicles:
        i.age_one()

    # resize image (too big)
    image = cv2.resize(frame, (0, 0), None, ratio, ratio)
    
    # converts image to gray
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    # subtract frames to take out the changes
    fgmask = fgbg.apply(image)

    # Binarization
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # OPening i.e First Erode the dilate
    o_mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernel)

    # Closing i.e First Dilate then Erode
    c_mask = cv2.morphologyEx(o_mask, cv2.MORPH_CLOSE, kernel)

    # increase the white region in the image.
    #dilation = cv2.dilate(c_mask, kernel)

    # Find Contours
    #_,all_contours, hierarchy = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, all_contours, hierarchy = cv2.findContours(c_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #---------------------------
    normal_car = True
    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            ####Tracking######
            m = cv2.moments(cnt)
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            x, y, w, h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit, down_limit):
                for i in my_vehicles:
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        if i.going_UP(line_down, line_up) == True:
                            cnt_up += 1
                            print("ID:", i.getId(),
                                    'crossed going up at', time.strftime("%c"))
                        elif i.going_DOWN(line_down, line_up) == True:
                            cnt_down += 1
                            print("ID:", i.getId(),
                                    'crossed going up at', time.strftime("%c"))
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()
                    if i.timedOut():
                        index = my_vehicles.index(i)
                        my_vehicles.pop(index)
                        del i

                if new == True:  # If nothing is detected,create new
                    p = vehicles.Car(pid, cx, cy, max_p_age)
                    my_vehicles.append(p)
                    pid += 1

            # classify vehicle by area: <normal, heavy>
            if area > type_of_vehicle_area_treshold:
                normal_car = False
            rect_color = normal_vehicle_color if normal_car else heavy_vehicle_color

            # draw rectangle
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
            img = cv2.rectangle(image, (x, y), (x + w, y + h), rect_color, 2)
            
            # draw rectangle lable
            label = 'Normal' if normal_car else 'Heavy'
            labelSize, baseLine = cv2.getTextSize(label, font, 0.5, 1)
            cv2.rectangle(image, (x - 1, y - labelSize[1] - 8), (x + labelSize[0] + 2, y), rect_color, cv2.FILLED)
            cv2.putText(image, label, (x, y-6), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # print car Id
    for i in my_vehicles:
        cv2.putText(image, str(i.getId()), (i.getX(), i.getY()),
                    font, 0.3, i.getRGB(), 1, cv2.LINE_AA)

    # print texts and lines
    str_up = 'UP: ' + str(cnt_up)
    str_down = 'DOWN: ' + str(cnt_down)
    image = cv2.polylines(image, [pts_L1], False, line_down_color, thickness=2)
    image = cv2.polylines(image, [pts_L2], False, line_up_color, thickness=2)
    image = cv2.polylines(image, [pts_L3], False, (255, 255, 255), thickness=1)
    image = cv2.polylines(image, [pts_L4], False, (255, 255, 255), thickness=1)
    cv2.putText(image, str_up, (10, 40), font, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str_up, (10, 40), font,
                0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(image, str_down, (10, 90), font,
                0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str_down, (10, 90), font,
                0.5, (255, 0, 0), 1, cv2.LINE_AA)

    
    # ######################## Displays images and transformations #########################
    cv2.imshow('Frame', image)
    cv2.moveWindow("Frame", 0, 0)

    cv2.imshow("Subtraction and Binarization (threshold)", imBin)
    cv2.moveWindow("Subtraction and Binarization (threshold)", int(width), 0)

    cv2.imshow("Closing Morphology", c_mask)
    cv2.moveWindow("Closing Morphology", 0, int(height))

    cv2.imshow("Opening Morphology", o_mask)
    cv2.moveWindow("Opening Morphology", int(width), int(height))

    # control video speed
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()