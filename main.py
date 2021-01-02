# ------------------------- created by: Zihadul Azam -------------------

# import libraries
import cv2
import _vehicles as vehicles # class file
import utils
import time

# /#/#/#/#/#/#/#/#/#/#/#/# <> Global Variables /#/#/#/#/#/#/#/#/#/#/#/#

input_video_path = 'samples/sample_1.mp4'
cap = cv2.VideoCapture(input_video_path)

ratio = .45

# Get width and height of video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * ratio
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * ratio

frameArea = height * width
areaTH = frameArea / 400 # tracking area
type_of_vehicle_area_treshold = 10000

# Lines (detection lines and limit lines)
up_limit = int(5 * (height / 10))
line_up = int(6 * (height / 10))
line_down = int(7 * (height / 10))
down_limit = int(8 * (height / 10))

# Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Kernal
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

my_vehicles = []
count_up = 0
count_down = 0
count_heavy = 0
count_normal = 0
max_vehicle_age = 5
vehicle_id = 1

# /#/#/#/#/#/#/#/#/#/#/#/# </> End Global Variables /#/#/#/#/#/#/#/#/#/#/#/#


# /#/#/#/#/#/#/#/#/#/#/#/# <> Start Frame Analysis /#/#/#/#/#/#/#/#/#/#/#/#

while(cap.isOpened()):
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Total vehicles passed: ", (count_up + count_down))
        print("Can't show the video. Exiting ...")
        break

    for v in my_vehicles:
        v.age_one()

    # resize image (too big)
    image = cv2.resize(frame, (0, 0), None, ratio, ratio)
    
    # converts image to gray
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    # subtract frames to take out the changes
    fgmask = fgbg.apply(image)

    # Binarization
    ret, binary_img = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # OPening i.e First Erode the dilate
    o_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # Closing i.e First Dilate then Erode
    c_mask = cv2.morphologyEx(o_mask, cv2.MORPH_CLOSE, kernel)

    # increase the white region in the image.
    #dilation = cv2.dilate(c_mask, kernel)

    # Find Contours
    #_,all_contours, hierarchy = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, all_contours, hierarchy = cv2.findContours(c_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    normal_vehicle = True
    for contour in all_contours:
        area = cv2.contourArea(contour)
        if area > areaTH:
            ####Tracking######
            m = cv2.moments(contour)
            cx = int(m['m10'] / m['m00']) # X of centroid
            cy = int(m['m01'] / m['m00']) # Y of centroid

            bounding_rect_x, bounding_rect_y, bounding_rect_w, bounding_rect_h = cv2.boundingRect(contour)

            new = True
            # if the contour is inside the detection (count) zone
            if cy in range(up_limit, down_limit):
                # check if current vehicle is already into the list
                for v in my_vehicles:
                    if abs(bounding_rect_x - v.getX()) <= bounding_rect_w and abs(bounding_rect_y - v.getY()) <= bounding_rect_h:
                        new = False
                        v.updateCoords(cx, cy)
                        v.updateArea(area)
                        v.updateType(type_of_vehicle_area_treshold)

                        if v.going_UP(line_up) == True:
                            count_up += 1
                            count_heavy += 1 if v.type == 'heavy' else 0 
                            count_normal += 1 if v.type == 'normal' else 0 
                            print("ID:", v.getId(), 'crossed going [UP] at', time.strftime("%c"))
                        elif v.going_DOWN(line_down) == True:
                            count_down += 1
                            count_heavy += 1 if v.type == 'heavy' else 0 
                            count_normal += 1 if v.type == 'normal' else 0 
                            print("ID:", v.getId(), 'crossed going [DOWN] at', time.strftime("%c"))
                        break
                    # set done if current vehicle already passed the up or down detection line
                    if v.getState() == '1':
                        if v.getDir() == 'down' and v.getY() > down_limit:
                            v.setDone()
                        elif v.getDir() == 'up' and v.getY() < up_limit:
                            v.setDone()
                # create new
                if new == True:  
                    new_Vehicle = vehicles.Vehicle(vehicle_id, cx, cy, area, max_vehicle_age)
                    my_vehicles.append(new_Vehicle)
                    vehicle_id += 1

            # classify vehicle by area: <normal, heavy>
            if area > type_of_vehicle_area_treshold:
                normal_vehicle = False

            # draw vehicle boundary rectangle and center point
            utils.drawVehicleCenterPoint(image, cx, cy)
            utils.drawVehicleBoundaryRect(image, bounding_rect_x, bounding_rect_y, bounding_rect_w, bounding_rect_h, normal_vehicle)

    # Print vehicle Id if not timedOut else delete
    for v in my_vehicles:
        if not v.timedOut():
            utils.printVehicleId(image, v)
        else:
            index = my_vehicles.index(v)
            my_vehicles.pop(index)
            del v
            

    # Print detection lines
    utils.printDetectionLines(image, line_up, line_down, up_limit, down_limit, width)

    # Print output texts (Total counts)
    utils.printOutputTexts(image, count_up, count_down, count_normal, count_heavy)
    
    # Displays Image and Transformations 
    utils.displayImageAndTransformations(image, binary_img, c_mask, o_mask, height, width)

    # control video speed
    if cv2.waitKey(20) & 0xff == ord('q'):
        break

# /#/#/#/#/#/#/#/#/#/#/#/# </> End Frame Analysis /#/#/#/#/#/#/#/#/#/#/#/#

cap.release()
cv2.destroyAllWindows()