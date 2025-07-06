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
PLAYER_AREA_DISTANCE = 32 # this is a variable, try tinkering with it

def process_img(original_img):
    processed_img = cv2.resize(original_img, (320, 320)) # resizing the original feed coming from game, all the
    return processed_img                                 # operations will be done on this resized/processed image

def detect_objects(model, processed_img):
    result = model.predict(processed_img, imgsz=(320, 320), iou=0.7, verbose=False, conf=0.5, max_det=5)
    return result

def draw_person_boxes(processed_img, result):
    if result is not None: # check if any object is detected
        boxes = result[0].boxes
        for box in boxes:
            if int(box.cls)==0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = (x1+x2)//2
                center_y = (y1+y2)//2
                distance = math.sqrt((center_x-IMG_CENTER_X)**2 + (center_y-IMG_CENTER_Y)**2)
                if distance<PLAYER_AREA_DISTANCE: # if the player itself is detected then ignore it
                    continue
                cv2.rectangle(processed_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
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
        processed_img = draw_person_boxes(processed_img, result) # if object (person, in this case) is detected in 
                                                                # the frame/image then draw a rectangle around it
        cv2.imshow("object_detection window", processed_img)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            sct.close()
            break

main()