import cv2
import time

# 1. Open the video file asset
video_reader = cv2.VideoCapture('mock_edge_source.mp4')

frame_counter = 0
processed_count = 0

print("Starting video processing loop...")

# 2. Run the frame reading loop
while video_reader.isOpened():
    success, frame = video_reader.read()
    
    # If the video fails to read or ends, break the loop
    if not success:
        print("End of video file or read error.")
        break
        
    frame_counter += 1
    
    # 3. Skip 3 out of 4 frames (Drops 60 FPS down to 15 FPS)
    if frame_counter % 4 != 0:
        continue  # Skip everything below, jump back to top of loop
        
    # 4. Resize the image matrix down to handle low CPU power (Width, Height)
    resized_frame = cv2.resize(frame, (360, 640))
    
    processed_count += 1
    print(f"Processed frame number: {frame_counter} (Total kept: {processed_count})")
    
    # 5. Stop the script cleanly after processing exactly 5 kept frames
    if processed_count >= 5:
        print("Successfully processed 5 kept frames. Exiting loop.")
        break

# 6. Clean up resources
video_reader.release()
print("Resources released cleanly.")