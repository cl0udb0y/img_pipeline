import zmq
import json
from collections import defaultdict
import time

def aggregate_inference_results(confidence_threshold=0.5):
    context = zmq.Context()
    zmq_socket = context.socket(zmq.SUB)
    zmq_socket.connect("tcp://localhost:5557")  # Changed port to 5557
    zmq_socket.setsockopt_string(zmq.SUBSCRIBE, '')

    # Socket for sending acknowledgments to inference service
    ack_socket = context.socket(zmq.PUSH)
    ack_socket.connect("tcp://localhost:5558")

    valid_detections = defaultdict(list)

    while True:
        try:
            result = zmq_socket.recv_json(flags=zmq.NOBLOCK)
            if result['confidence'] >= confidence_threshold:
                valid_detections[result['object_label']].append(result)
                print(f"Valid detection: {result['object_label']} with confidence {result['confidence']}")
                ack_socket.send(b"ack")  # Send acknowledgment

            # Apply further logic to determine if the objects detected are valid
            # Example: filter out objects detected for less than a certain duration
            final_detections = {}
            for label, detections in valid_detections.items():
                if len(detections) >= 5:  # Example: only consider if detected in 5 frames
                    final_detections[label] = detections

            # Save final detections to disk or send to database
            with open('final_detections.json', 'w') as f:
                json.dump(final_detections, f)

        except zmq.Again:
            pass

        time.sleep(0.1)  # Add a short sleep to prevent constant looping

def main():
    aggregate_inference_results()

if __name__ == '__main__':
    main()
