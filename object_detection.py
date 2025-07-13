import math
import cv2
import time
from ultralytics import YOLO
from directkeys import PressKey, ReleaseKey, PressMouse, ReleaseMouse, W, A, S, D, MoveMouseRelative

def load_model():
    model = YOLO("models/yolov8n.pt")
    model.to("cpu")
    return model

def detect_objects(model, processed_img):
    result = model.predict(processed_img, imgsz=(320, 320), iou=0.7, verbose=False, conf=0.5, max_det=5)
    return result

def draw_bounding_boxes(processed_img, object_number, boxes, MOUSE_SENSITIVITY, ANGULAR_SENSITIVITY, IMG_CENTER_X, IMG_CENTER_Y):
    # boxes = result[0].boxes
    nearest_person_distance = float("inf")
    idx = 0
    nearest_box_idx = None
    for box in boxes:
        if int(box.cls)==object_number: # 0 indicates "person" class, refer this: https://stackoverflow.com/questions/77477793/class-ids-and-their-relevant-class-names-for-yolov8-model
            # use "box.conf[0]" to get confidence score of the "detected person" by the model
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1+x2)//2
            center_y = (y1+y2)//2

            # if 120 <= center_x <= 200:  # ignore bounding boxes in middle screen range - because most of the time its player itself which is being detected
            #     continue
            
            distance_from_player = math.sqrt((center_x-IMG_CENTER_X)**2 + (center_y-IMG_CENTER_Y)**2)
            # if distance_from_player<PLAYER_AREA_DISTANCE: # if the player itself is detected then ignore it, not works perfectly at the moment
            #     continue                      # (assumption is that player will always be in the center of the image/screen)
            
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

def shoot(processed_img, boxes, box_idx, MOUSE_SENSITIVITY, ANGULAR_SENSITIVITY, IMG_CENTER_X=160, IMG_CENTER_Y=160):
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

        threshold = 5 # this needs to be tested properly
        theta = math.atan2(160-target_center_y, target_center_x-160)
        theta_degrees = math.degrees(theta)
        angle_factor = 1 + ANGULAR_SENSITIVITY*abs(math.sin(theta))
        base_time = MOUSE_SENSITIVITY*abs(offset_x)
        holdout_time = base_time*angle_factor
        
        # for now i'm just focusing on horizontal alignment - also play around and test with the time.sleep present inside it
        if abs(offset_x)>threshold:
            # test by using PressKey(W) to fix orientation
            PressKey(W)
            time.sleep(0.0001)
            ReleaseKey(W)
            if offset_x>0:
                print(f"box which needs to be shot detected at: {target_center_x, target_center_y}. moving right.", flush=True)
                PressKey(D)
                time.sleep(0.003)
                ReleaseKey(D)
            else:
                print(f"box which needs to be shot detected at: {target_center_x, target_center_y}. moving left.", flush=True)
                PressKey(A)
                time.sleep(0.003)
                ReleaseKey(A)
                
            # it is assumed that the orientation/things are adjusted now, and we are good to shoot the person present in the red bouding box
            PressMouse(1)
            time.sleep(0.05)
            ReleaseMouse(1)

    return processed_img