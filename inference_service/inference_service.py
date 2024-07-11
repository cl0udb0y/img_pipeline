import argparse
import zmq
import cv2
import json
import base64
import numpy as np
from PIL import Image, ImageDraw
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import os
import time
from ultralytics import YOLO

def perform_inference(model_path, labels_path, model_task, confidence_threshold=0.5):
    model = YOLO(model_path, task=model_task)
    labels = read_label_file(labels_path) if labels_path else {}

    # ZMQ Setup
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics

    result_socket = context.socket(zmq.PUB)
    result_socket.bind("tcp://*:5557")

    ack_socket = context.socket(zmq.PULL)
    ack_socket.bind("tcp://*:5558")

    # Video Writing Setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    stream_writers = {} 
    output_dir = "inference_videos"
    os.makedirs(output_dir, exist_ok=True)

    while True:
        try:
            message = socket.recv_json(flags=zmq.NOBLOCK)
            stream_url = message.get('stream_url')
            frame_data = message.get('frame_data')
            if not stream_url or not frame_data:
                print("Error: Invalid message format - missing stream_url or frame_data")
                continue

            # Handle motion_start and motion_stop events
            try:
                event = json.loads(frame_data)
                if "event" in event and event["event"] == "motion_start":
                    current_video_file = os.path.join(
                        output_dir,
                        f"{stream_url.replace('/', '_')}_motion_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
                    )
                    stream_writers[stream_url] = cv2.VideoWriter(
                        current_video_file, fourcc, 20.0, (640, 360)
                    )
                    print(
                        f"Motion started in stream: {stream_url}, recording to {current_video_file}"
                    )
                elif "event" in event and event["event"] == "motion_stop":
                    if stream_url in stream_writers:
                        stream_writers[stream_url].release()
                        del stream_writers[stream_url]
                    print(f"Motion stopped in stream: {stream_url}")
                    continue
            except json.JSONDecodeError:
                pass  # Handle cases where frame_data is not a valid JSON string

            # Decode and process the frame
            try:
                frame = cv2.imdecode(np.frombuffer(base64.b64decode(frame_data), np.uint8), -1)

                # Ensure 'image' is always defined
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
                image = cv2.resize(image, (640, 360))

                # Inference using YOLOv8 (Only if image is not None)
                results = model(image) if image is not None else None

                if results:
                    if model_task == "detect":
                        for r in results:
                            boxes = r.boxes
                            if boxes: # <-- Added this condition
                                for box in boxes:
                                    # Extract coordinates, class name, label
                                    x1, y1, x2, y2 = box.xyxy[0]
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    class_id = box.cls[0]
                                    confidence = box.conf[0]

                                    if confidence > confidence_threshold:
                                        label = f"{labels.get(class_id, 'Unknown')}: {confidence:.2f}"

                                        # Draw rectangle and label on the image
                                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                        # Create the result dictionary
                                        result = {
                                            "class_id": int(class_id),
                                            "confidence": float(confidence),
                                            "box": [x1, y1, x2, y2],
                                            "stream_url": stream_url
                                        }

                                        result_socket.send_json(result)  # Send detection results
                    
                    elif model_task == "classify":
                        # Handle classification results
                        pass  # or send classification results if required

                # Convert image back to BGR for OpenCV display and saving (if image is not None)
                frame_with_detections = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) if image is not None else None

                # Save the frame to the appropriate stream's video writer
                if stream_url in stream_writers and frame_with_detections is not None:
                    stream_writers[stream_url].write(frame_with_detections)

                # Display the frame (if image is not None)
                if frame_with_detections is not None:
                    cv2.imshow(stream_url, frame_with_detections)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except Exception as e:  # Catch all exceptions to avoid crashing
                print(f"Error processing frame from {stream_url}: {e}")

        except zmq.Again:
            pass  # No message available, continue to next iteration

        # Check for acknowledgment from aggregation service
        try:
            ack = ack_socket.recv(flags=zmq.NOBLOCK)
            print("Received acknowledgment from aggregation service")
        except zmq.Again:
            pass

        time.sleep(0.1)  # Adjust sleep time as needed

    # Close all video writers before exiting
    for writer in stream_writers.values():
        writer.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', required=True,
                        help='Path to .tflite model file')
    parser.add_argument('--labels', required=False,
                        help='Path to labels file')
    parser.add_argument('--task', required=True,
                        help='Model Task: detect, segment, classify,pose or obb')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold for detected objects')
    args = parser.parse_args()

    perform_inference(args.model, args.labels, args.task, args.confidence)

if __name__ == '__main__':
    main()
