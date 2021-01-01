
import cv2
import numpy as np

# variables
font = cv2.FONT_HERSHEY_DUPLEX
bg_rect_color = (56,50,38)

line_down_color = (254,90,61)
line_up_color = (87,0,245)

normal_vehicle_color = (118,230,0)
heavy_vehicle_color = (34, 87, 255)

def getDetectionLinePointsMatrix(lineY, width):
    point_1 = [0, lineY]
    point_2 = [width, lineY]
    point_matrix = np.array([point_1, point_2], np.int32)
    return point_matrix.reshape((-1, 1, 2))

def printDetectionLines(image, line_up, line_down, up_limit, down_limit, width):
    # up line
    line_up_points_cord = getDetectionLinePointsMatrix(line_up, width)
    image = cv2.polylines(image, [line_up_points_cord], False, line_up_color, thickness=2)
    # down line
    line_down_points_cord = getDetectionLinePointsMatrix(line_down, width)
    image = cv2.polylines(image, [line_down_points_cord], False, line_down_color, thickness=2)
    # up limit line
    limit_line_up_points_cord = getDetectionLinePointsMatrix(up_limit, width)
    image = cv2.polylines(image, [limit_line_up_points_cord], False, (224,224,224), thickness=1)
    # down limit line
    limit_line_down_points_cord = getDetectionLinePointsMatrix(down_limit, width)
    image = cv2.polylines(image, [limit_line_down_points_cord], False, (224,224,224), thickness=1)


def printOutputTexts(image, count_up, count_down, count_normal, count_heavy):
    
    cv2.rectangle(image, (5, 5), (200, 140), bg_rect_color, cv2.FILLED)
    
    str_up = 'Going UP: ' + str(count_up)
    str_down = 'Going DOWN: ' + str(count_down)
    str_normal = 'Normal vehicles: ' + str(count_normal)
    str_heavy = 'Heavy vehicles: ' + str(count_heavy)
    str_total = 'Total vehicles: ' + str(count_up + count_down)

    cv2.putText(image, str_up, (15, 25), font,
                0.45, line_up_color, 1, cv2.LINE_AA)

    cv2.putText(image, str_down, (15, 45), font,
                0.45, line_down_color, 1, cv2.LINE_AA)
    
    cv2.putText(image, str_normal, (15, 75), font,
                0.45, normal_vehicle_color, 1, cv2.LINE_AA)
    
    cv2.putText(image, str_heavy, (15, 95), font,
                0.45, heavy_vehicle_color, 1, cv2.LINE_AA)

    cv2.putText(image, str_total, (15, 125), font,
                0.45, (255,255,255), 1, cv2.LINE_AA)

def printVehicleId(image, vehicle):
    #cv2.putText(image, str(vehicle.getId()), (vehicle.getX(), vehicle.getY()),
    #                    font, 0.5, vehicle.getRGB(), 1, cv2.LINE_AA)
    cv2.putText(image, str(vehicle.getId()), (vehicle.getX(), vehicle.getY()),
                        font, 0.5, (57,220,205), 1, cv2.LINE_AA)

def drawVehicleBoundaryRect(image, bounding_rect_x, bounding_rect_y, bounding_rect_w, bounding_rect_h, normal_vehicle):
    rect_color = normal_vehicle_color if normal_vehicle else heavy_vehicle_color
    # draw rectangle
    cv2.rectangle(image, (bounding_rect_x, bounding_rect_y), (bounding_rect_x + bounding_rect_w, bounding_rect_y + bounding_rect_h), rect_color, 2)
    # draw rectangle lable
    label = 'Normal' if normal_vehicle else 'Heavy'
    labelSize, baseLine = cv2.getTextSize(label, font, 0.5, 1)
    cv2.rectangle(image, (bounding_rect_x - 1, bounding_rect_y - labelSize[1] - 8), (bounding_rect_x + labelSize[0] + 2, bounding_rect_y), rect_color, cv2.FILLED)
    cv2.putText(image, label, (bounding_rect_x, bounding_rect_y-6), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def drawVehicleCenterPoint(image, cx, cy):
    # draw rectangle
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

def showMoveWindow(image, text, x, y):
    cv2.imshow(text, image)
    cv2.moveWindow(text, x, y)

def displayImageAndTransformations(image, imBin, c_mask, o_mask, height, width):
    showMoveWindow(image, 'Frame', 0, 0)
    showMoveWindow(imBin, 'Subtraction and Binarization (threshold)', int(width), 0)
    showMoveWindow(c_mask, 'Closing Morphology', 0, int(height))
    showMoveWindow(o_mask, 'Opening Morphology', int(width), int(height))
