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
MOUSE_SENSITIVITY = 0.0001
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
    idx = 0
    nearest_box_idx = None
    for box in boxes:
        if int(box.cls)==0: # 0 indicates "person" class, refer this: https://stackoverflow.com/questions/77477793/class-ids-and-their-relevant-class-names-for-yolov8-model
            # use "box.conf[0]" to get confidence score of the "detected person" by the model
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1+x2)//2
            center_y = (y1+y2)//2

            if 120 <= center_x <= 200:  # ignore bounding boxes in middle screen range - because most of the time its player itself which is being detected
                continue
            
            distance_from_player = math.sqrt((center_x-IMG_CENTER_X)**2 + (center_y-IMG_CENTER_Y)**2)
            if distance_from_player<PLAYER_AREA_DISTANCE: # if the player itself is detected then ignore it, not works perfectly at the moment
                continue                      # (assumption is that player will always be in the center of the image/screen)
            
            if distance_from_player<nearest_person_distance:
                nearest_person_distance = distance_from_player
                nearest_box_idx = idx
            
            cv2.rectangle(processed_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) # green
        idx += 1
    # now one possible thing which can be done is that instead of shooting the nearest person 
    # (since it can be player itself also)  we can shoot the second nearest person
    # if nearest_box_idx is not None:
    #     # redraw the nearest person with different color (red) to identify him
    #     x1, y1, x2, y2 = boxes[nearest_box_idx].xyxy[0].cpu().numpy()
    #     cv2.rectangle(processed_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) # red, nearest one
    return processed_img, nearest_box_idx

def shoot(processed_img, boxes, box_idx, player_center_x=160, player_center_y=160):
    if box_idx is not None:
        x1, y1, x2, y2 = boxes[box_idx].xyxy[0].cpu().numpy()
        cv2.rectangle(processed_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) # red, this will get shot
        # now since the coordinates of this is know then we can move the player coordinate location from whatever direction 
        # it is in to the direction of center of [(x1, y1) and (x2, y2)] with help of "W", "A", "S" and "D" key bindings (mouse movement is not working in the game that's why keys)
        # once the player is aligned with the desired coordinate location, we can start shooting
        
        # setting up coordiantes where we want the player coordinate direction to be 
        target_center_x = (x1+x2)/2
        target_center_y = y1 # for headshot otherwise this can also be (y1+y2)/2
        
        # just using PressKey(W) for 0.0001 seconds might possibly fix the issue faced here,
        # since it would straighten up the orientation of the player (hypothesis, might NOT work, or might work, hehe)
        # key problem faced here:
        # this assumes (offset_x and offset_y) that player would be in the center but his orientation in the center can be different, like it can be in any direction
        
        offset_x = target_center_x-IMG_CENTER_X
        offset_y = target_center_y-IMG_CENTER_Y

        # test by using PressKey(W) to fix orientation
        PressKey(W)
        # time.sleep(0.0001)
        ReleaseKey(W)

        threshold = 5 # this needs to be tested properly

        # for now i'm just focusing on horizontal alignment - also play around and test with the time.sleep present inside it
        if abs(offset_x)>threshold:
            holdout_time = max(0.05, MOUSE_SENSITIVITY*abs(offset_x))
            if offset_x>0:
                print(f"box which needs to be shot detected at: {target_center_x, target_center_y}. moving right.", flush=True)
                PressKey(D)
                time.sleep(holdout_time)
                ReleaseKey(D)
            else:
                print(f"box which needs to be shot detected at: {target_center_x, target_center_y}. moving left.", flush=True)
                PressKey(A)
                time.sleep(holdout_time)
                ReleaseKey(A)
                
            # it is assumed that the orientation/things are adjusted now, and we are good to shoot the person present in
            # the blue bounding box 
            PressMouse(1)
            time.sleep(0.05)
            ReleaseMouse(1)

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
            processed_img, nearest_box_idx = draw_person_boxes(processed_img, boxes) # if object (person, in this case) is detected in 
                                                                    # the frame/image then draw a rectangle around it
            
            processed_img = shoot(processed_img, boxes, nearest_box_idx, 160, 160)
        
        cv2.imshow("object_detection window", processed_img)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            sct.close()
            break

main()