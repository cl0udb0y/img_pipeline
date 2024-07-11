from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

client = MongoClient('localhost', 27017)
db = client['object_detection']
collection = db['detections']

IMAGE_SAVE_PATH = '../dataset/images'

@app.route('/detections', methods=['GET'])
def get_detections():
    detections = list(collection.find({}, {'_id': False}))
    print(f"Retrieved {len(detections)} detections")
    return jsonify(detections)

@app.route('/image/<filename>', methods=['GET'])
def get_image(filename):
    print(f"Serving image: {filename}")
    return send_from_directory(IMAGE_SAVE_PATH, filename)

@app.route('/validate', methods=['POST'])
def validate_detection():
    data = request.json
    detection_id = data['id']
    is_valid = data['is_valid']
    result = collection.update_one({'_id': detection_id}, {'$set': {'is_valid': is_valid}})
    print(f"Validated detection: {detection_id}, result: {result.modified_count}")
    return jsonify({'status': 'success'})

@app.route('/reclassify', methods=['POST'])
def reclassify_detection():
    data = request.json
    detection_id = data['id']
    new_class_id = data['new_class_id']
    result = collection.update_one({'_id': detection_id}, {'$set': {'class_id': new_class_id}})
    print(f"Reclassified detection: {detection_id}, new class: {new_class_id}, result: {result.modified_count}")
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    print("API service started")
    app.run(debug=True, host='0.0.0.0', port=5001)
