# https://stackoverflow.com/questions/50963283/opencv-imshow-doesnt-need-convert-from-bgr-to-rgb
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import cv2
import time
import mss
from ultralytics import YOLO
from directkeys import PressKey, ReleaseKey, W, A, S, D

cv2.setUseOptimized(False)
cv2.setNumThreads(1)

sct = mss.mss()
monitor = {"top": 200, "left": 0, "width": 800, "height": 475}

# try:
#    model = YOLO("yolov8n.pt")
#    model.to("cpu")
#    model.predict(np.zeros((320, 320, 3), dtype=np.uint8))
# except Exception as e:
#    print(f"Error {e} loading up model.")

def detect_objects(model, img):
    start = time.time()
    result = model.predict(img, imgsz=(320, 320), iou=0.7, verbose=False, conf=0.5, max_det=3)
    print(f"Time taken for object detection: {time.time()-start}.")
    return result

def draw_person_boxes(img, result):
    boxes = result[0].boxes
    for box in boxes:
        if int(box.cls)==0:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return img
    
def process_img(original_img):
    # processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.resize(original_img, (320, 320))
    return processed_img

def main():
    
    print("Main function started.", flush=True)    
    cv2.namedWindow("processed window", cv2.WINDOW_NORMAL)
    print("Window for cv2 initialized.", flush=True)
    
    model = YOLO("models/yolov8n.pt")
    model.to("cpu")

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    # flag = True
    frame_count = 0

    while True:
       
       # if flag:
       #     for i in range(4):    
       #         PressKey(W)
       #         time.sleep(1)
       #         ReleaseKey(W)
       #         flag = False
        
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)[:, :, :3]
        
        print("process_img function is going to get called.", flush=True)
        processed_frame = process_img(frame)
        print("process_img function called successfully.", flush=True)

        if frame_count%3==0:
            print("detect objects before", flush=True)
            last_result = detect_objects(model, processed_frame)
            print("detect objects after", flush=True)
        
        if last_result is not None and last_result[0].boxes is not None:
            print("draw person boxes before.", flush=True)
            processed_frame =  draw_person_boxes(processed_frame, last_result)
            print("draw person boxes after.", flush=True)

        frame_count += 1
        
        print("Time taken to grab one screen frame in seconds:", time.time()-last_time)
        last_time = time.time() 
        
        cv2.imshow("processed window", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            sct.close()
            break

main()
