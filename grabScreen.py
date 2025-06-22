import numpy as np
import cv2
import time
import mss

# Initialize mss
sct = mss.mss()

# Define the region to capture
monitor = {"top": 100, "left": 0, "width": 800, "height": 450}

last_time = time.time()
frame_count = 0

while True:
    # Capture the screen
    sct_img = sct.grab(monitor)
    print("Time taken to grab one screen frame in seconds:", {time.time()-last_time})
    last_time = time.time()
    # Convert to numpy array (BGR format for OpenCV)
    frame = np.array(sct_img)[:, :, :3]  # Remove alpha channel
    # frame = cv2.cvtColor(frame)
    
    # Calculate FPS every 30 frames
    frame_count += 1
    if frame_count % 30 == 0:
        current_time = time.time()
        fps = 30 / (current_time - last_time)
        print(f"FPS: {fps:.1f}")
        last_time = current_time
    
    cv2.imshow("Fast Screen Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
sct.close()
