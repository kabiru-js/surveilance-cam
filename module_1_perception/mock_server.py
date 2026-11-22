import socket
import struct
import time
import cv2

# 1. Initialize TCP Network Socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('127.0.0.1', 6000))
server_socket.listen(1)

print("Edge Camera Server online on 127.0.0.1:6000.")

# Outer loop keeps the camera alive if the central server disconnects
while True:
    print("Waiting for central ingestion pipeline to connect...")
    connection, address = server_socket.accept()
    print(f"Central server connected from: {address}")

    # Open the local video file to simulate live camera feed
    video_reader = cv2.VideoCapture('mock_edge_source.mp4')
    frame_counter = 0

    try:
        while video_reader.isOpened():
            success, frame = video_reader.read()

            # Rewind video seamlessly to loop infinitely like a real CCTV feed
            if not success:
                video_reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_counter += 1

            # Gatekeeper: Skip 3 out of 4 frames (drops 60 FPS down to 15 FPS)
            if frame_counter % 4 != 0:
                continue

            # Resize matrix to protect dual-core CPU
            resized_frame = cv2.resize(frame, (360, 640))

            # Compress matrix into JPEG bytes
            _, encoded_buffer = cv2.imencode('.jpg', resized_frame)
            data_bytes = encoded_buffer.tobytes()

            # Pack byte length into a 4-byte binary envelope header
            payload_size = len(data_bytes)
            header = struct.pack('!I', payload_size)

            # Transmit header first, then image bytes
            connection.sendall(header)
            connection.sendall(data_bytes)

            # Enforce 15 FPS pacing
            time.sleep(1 / 15)

    except (ConnectionResetError, BrokenPipeError, socket.error):
        print("Central server disconnected. Resetting camera stream...")
    finally:
        video_reader.release()
        connection.close()
