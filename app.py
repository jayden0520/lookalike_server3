from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from flask_cors import CORS
import pickle
import os
import uuid
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_PATH = os.path.join(BASE_DIR, "celebrity_encodings.pkl")

print("[INFO] Loading celebrity embeddings from pickle...")
with open(PICKLE_PATH, "rb") as f:
    data = pickle.load(f)

celebrity_encodings = data["encodings"]
celebrity_names = data["names"]
celebrity_images_b64 = data.get("images", [])  # base64 images added in pickle

print(f"[✅] Loaded {len(celebrity_encodings)} celebrity faces")

app = Flask(__name__)
CORS(app)

@app.route('/match', methods=['POST'])
def match_celeb():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    algorithm = request.form.get('algorithm', 'hog').lower()
    if algorithm not in ['hog', 'cnn']:
        algorithm = 'hog'

    file = request.files['image']
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    file.save(temp_filename)

    try:
        image = face_recognition.load_image_file(temp_filename)
        face_locations = face_recognition.face_locations(image, model=algorithm)
        input_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)

        if not input_encodings:
            return jsonify({"error": "No face found"}), 400

        input_encoding = input_encodings[0]
        distances = face_recognition.face_distance(celebrity_encodings, input_encoding)
        best_index = np.argmin(distances)

        match_name = celebrity_names[best_index]
        similarity = round((1 - float(distances[best_index])) * 100, 2)
        match_image_b64 = celebrity_images_b64[best_index] if celebrity_images_b64 else None

        return jsonify({
            "match_name": match_name,
            "similarity": similarity,
            "celebrity_image_b64": match_image_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)