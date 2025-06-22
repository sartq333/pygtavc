# https://stackoverflow.com/questions/50963283/opencv-imshow-doesnt-need-convert-from-bgr-to-rgb
import numpy as np
import cv2
import time
import mss
from directkeys import PressKey, ReleaseKey, W, A, S, D

sct = mss.mss()
monitor = {"top": 100, "left": 0, "width": 800, "height": 450}

def process_img(original_img):
    processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img

def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    flag = True

    while True:
        if flag:
            for i in range(4):    
                PressKey(W)
                time.sleep(1)
                ReleaseKey(W)
                flag = False
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)[:, :, :3]
        processed_frame = process_img(frame)
        print("Time taken to grab one screen frame in seconds:", {time.time()-last_time})
        last_time = time.time() 
        cv2.imshow("processed window", processed_frame)
        # cv2.imshow("window", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            sct.close()
            break

main()
