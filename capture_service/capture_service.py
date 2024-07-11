import argparse
import time
import json
import cv2
import numpy as np
from collections import deque
import zmq
import base64

def restart_stream(cap, url):
    """Attempt to restart the RTSP stream."""
    cap.release()
    time.sleep(1)  # Wait for 1 second before restarting the stream
    new_cap = cv2.VideoCapture(url)
    if new_cap.isOpened():
        print(f"Successfully restarted RTSP stream: {url}")
    else:
        print(f"Failed to restart RTSP stream: {url}")
    return new_cap

def capture_motion(stream_configs, buffer_size=25, min_area=300, sensitivity=20, min_motion_percentage=1.0, min_motion_duration=2.0, zmq_socket=None):
    caps = [cv2.VideoCapture(stream['url']) for stream in stream_configs]
    previous_frames = [None] * len(stream_configs)
    motion_buffers = [deque(maxlen=buffer_size) for _ in stream_configs]
    motion_events = [False] * len(stream_configs)
    reference_frames = {'day': [None] * len(stream_configs), 'night': [None] * len(stream_configs)}
    motion_start_times = [None] * len(stream_configs)
    adaptive_thresholds = [None] * len(stream_configs)  # Initialize with a default threshold
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True) 

    while True:
        current_time = time.time()
        for i, cap in enumerate(caps):
            ret, frame = cap.read()

            # Error Handling and Stream Restart
            if not ret or frame is None: 
                print(f"Error: Unable to read frame from RTSP stream: {stream_configs[i]['url']}. Attempting to restart...")
                caps[i] = restart_stream(cap, stream_configs[i]['url'])  # Attempt to restart the stream
                time.sleep(2)  # Give the stream some time to recover before trying again
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            # Background Subtraction
            fgmask = fgbg.apply(gray_frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            # Adaptive Thresholding (Corrected)
            if adaptive_thresholds[i] is None:
                # Initialize with a default threshold value if it's still None
                adaptive_thresholds[i] = 127 
                _, thresholded = cv2.threshold(gray_frame, adaptive_thresholds[i], 255, cv2.THRESH_BINARY_INV)
            else:
                # Use the updated threshold from the previous frame
                _, thresholded = cv2.threshold(gray_frame, adaptive_thresholds[i], 255, cv2.THRESH_BINARY_INV)

            # Update the adaptive threshold for the next frame
            adaptive_thresholds[i] = cv2.mean(gray_frame)[0]

            thresh = cv2.bitwise_and(fgmask, thresholded)

            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate motion percentage (Corrected)
            motion_percentage = (np.count_nonzero(thresh) / (thresh.shape[0] * thresh.shape[1])) * 100

            if motion_detected and motion_percentage >= min_motion_percentage:
                if motion_start_times[i] is None:
                    motion_start_times[i] = current_time
                motion_duration = current_time - motion_start_times[i]
                if motion_duration >= min_motion_duration and not motion_events[i]:
                    motion_events[i] = True
                    print(f"Motion started in stream: {stream_configs[i]['url']}")
                    zmq_socket.send_json({'event': 'motion_start', 'stream_url': stream_configs[i]['url']})
            else:
                motion_start_times[i] = None

            if not motion_detected or motion_percentage < min_motion_percentage:
                if motion_events[i]:
                    motion_events[i] = False
                    print(f"Motion stopped in stream: {stream_configs[i]['url']}")
                    zmq_socket.send_json({'event': 'motion_stop', 'stream_url': stream_configs[i]['url']})

            if motion_detected:
                motion_buffers[i].append(frame)
                print(f"Motion detected in stream: {stream_configs[i]['url']} with {motion_percentage:.2f}% motion")

            # Overlay motion percentage on the frame
            cv2.putText(frame, f"Motion: {motion_percentage:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display the frame
            cv2.imshow(f'Stream {i+1} - {stream_configs[i]["url"]}', frame)

            # Publish motion frames to ZMQ if motion is detected
            if motion_detected:
                stream_url = stream_configs[i]['url']
                _, buffer = cv2.imencode('.jpg', frame)

                # Fix unhashable type error:
                message = {
                    "stream_url": stream_url,  # Key is now a string
                    "frame_data": base64.b64encode(buffer).decode('utf-8')  # Convert bytes to string for JSON compatibility
                }

                zmq_socket.send_json(message)  # Send as JSON

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='Path to configuration JSON file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    buffer_size = config.get('buffer_size', 25)
    min_area = config.get('min_area', 300)  # Adjusted to be more sensitive to smaller objects
    sensitivity = config.get('sensitivity', 20)  # New parameter for sensitivity
    min_motion_percentage = config.get('min_motion_percentage', 1.0)  # Minimum motion percentage to trigger event
    min_motion_duration = config.get('min_motion_duration', 2.0)  # Minimum duration of motion to trigger event

    # Set up ZMQ publisher
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUB)
    zmq_socket.bind("tcp://*:5555")

    capture_motion(config['streams'], buffer_size, min_area, sensitivity, min_motion_percentage, min_motion_duration, zmq_socket)

if __name__ == '__main__':
    main()
