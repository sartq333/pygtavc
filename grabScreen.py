# https://stackoverflow.com/questions/50963283/opencv-imshow-doesnt-need-convert-from-bgr-to-rgb
import math
import numpy as np
import cv2
import time
import mss
from ultralytics import YOLO
from directkeys import PressKey, ReleaseKey, PressMouse, ReleaseMouse, W, A, S, D, MoveMouseRelative

sct = mss.mss()
monitor = {"top": 200, "left": 0, "width": 800, "height": 475} # this much part of the screen i want to record
IMG_WIDTH = 320
IMG_HEIGHT = 320
IMG_CENTER_X = IMG_WIDTH//2
IMG_CENTER_Y = IMG_HEIGHT//2
PLAYER_AREA_DISTANCE = 84 # this is a variable, try tinkering with it

def process_img(original_img):
    processed_img = cv2.resize(original_img, (320, 320)) # resizing the original feed coming from game, all the
    return processed_img                                 # operations will be done on this resized/processed image

def detect_objects(model, processed_img):
    result = model.predict(processed_img, imgsz=(320, 320), iou=0.7, verbose=False, conf=0.5, max_det=5)
    return result

def draw_person_boxes(processed_img, boxes):
    # boxes = result[0].boxes
    nearest_person_distance = float("inf")
    second_nearest_person_distance = float("inf")
    idx = 0
    box_idx = None
    second_nearest_box_idx = None
    for box in boxes:
        if int(box.cls)==0: # 0 indicates "person" class, refer this: https://stackoverflow.com/questions/77477793/class-ids-and-their-relevant-class-names-for-yolov8-model
            # use "box.conf[0]" to get confidence score of the "detected person" by the model
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1+x2)//2
            center_y = (y1+y2)//2
            distance_from_player = math.sqrt((center_x-IMG_CENTER_X)**2 + (center_y-IMG_CENTER_Y)**2)
            if distance_from_player<PLAYER_AREA_DISTANCE: # if the player itself is detected then ignore it, not works perfectly at the moment
                continue                      # (assumption is that player will always be in the center of the image/screen)
            
            if distance_from_player<nearest_person_distance:
                second_nearest_person_distance = nearest_person_distance
                second_nearest_box_idx = box_idx
                nearest_person_distance = distance_from_player
                box_idx = idx
            
            cv2.rectangle(processed_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        idx += 1
    # now one possible thing which can be done is that instead of shooting the nearest person 
    # (since it can be player itself also)  we can shoot the second nearest person
    if box_idx is not None:
        # redraw the nearest person with different color (red) to identify him
        x1, y1, x2, y2 = boxes[box_idx].xyxy[0].cpu().numpy()
        cv2.rectangle(processed_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) # red, nearest one
    return processed_img, second_nearest_box_idx

def shoot(processed_img, boxes, second_nearest_box_idx):
    if second_nearest_box_idx is not None:
        x1, y1, x2, y2 = boxes[second_nearest_box_idx].xyxy[0].cpu().numpy()
        cv2.rectangle(processed_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2) # blue, this will get hit
        # now since the coordinates of this is know then we can move the player coordinate location from whatever direction 
        # it is in to the direction of center of [(x1, y1) and (x2, y2)] with help of "W", "A", "S" and "D" key bindings (mouse movement is not working in the game that's why keys)
        # once the player is alinged with the desired coordinate location, we can start shooting
    return processed_img

def main():
    cv2.namedWindow("object_detection window", cv2.WINDOW_NORMAL)
    model = YOLO("models/yolov8n.pt")
    model.to("cpu")

    while True:
        sct_img = sct.grab(monitor) # grabbing a particular section of screen
        """
        print(type(sct_img))
        <class 'mss.screenshot.ScreenShot'>
        print(sct_img)
        <ScreenShot pos=0,200 size=800x475>
        """
        frame = np.array(sct_img)[:, :, :3] # converting it into numpy array
        processed_img = process_img(frame) # resizing the image into (320*320)
        result = detect_objects(model, processed_img) # here we are running the inference for object detection
        if result is not None: # check if any object is detected
            # (the if condition is written here and not inside the functions because it's benificial that we can use boxes for both draw_person_boxes
            # and shoot function, and don't have to unpack boxes from result inside them)
            boxes = result[0].boxes
            processed_img, second_nearest_box_idx = draw_person_boxes(processed_img, result) # if object (person, in this case) is detected in 
                                                                    # the frame/image then draw a rectangle around it
            processed_img = shoot(processed_img, boxes, second_nearest_box_idx)
        
        cv2.imshow("object_detection window", processed_img)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            sct.close()
            break

main()