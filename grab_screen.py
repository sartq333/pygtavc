# https://stackoverflow.com/questions/50963283/opencv-imshow-doesnt-need-convert-from-bgr-to-rgb
# import math
import numpy as np
import cv2
# import time
import mss
# from directkeys import PressKey, ReleaseKey, PressMouse, ReleaseMouse, W, A, S, D, MoveMouseRelative
from object_detection import load_model, detect_objects, draw_bounding_boxes, shoot

sct = mss.mss()
monitor = {"top": 200, "left": 0, "width": 800, "height": 475} # this much part of the screen i want to record

MOUSE_SENSITIVITY = 0.0001
ANGULAR_SENSITIVITY = 0.5
IMG_WIDTH = 320
IMG_HEIGHT = 320
IMG_CENTER_X = IMG_WIDTH//2
IMG_CENTER_Y = IMG_HEIGHT//2
object_number = 2

def process_img(original_img):
    processed_img = cv2.resize(original_img, (320, 320)) # resizing the original feed coming from game, all the
    return processed_img                                 # operations will be done on this resized/processed image

def main():
    cv2.namedWindow("object_detection window", cv2.WINDOW_NORMAL) 
    model = load_model()

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
            processed_img, nearest_box_idx = draw_bounding_boxes(processed_img, object_number, boxes, MOUSE_SENSITIVITY, ANGULAR_SENSITIVITY, IMG_CENTER_X, IMG_CENTER_Y) # if object (person, in this case) is detected in 
                                                                    # the frame/image then draw a rectangle around it
            processed_img = shoot(processed_img, boxes, nearest_box_idx, 160, 160)
        
        cv2.imshow("object_detection window", processed_img)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            sct.close()
            break

main()