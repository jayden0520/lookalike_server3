from flask import Flask, request, jsonify, send_from_directory
import face_recognition
import os
import uuid
import imghdr
from flask_cors import CORS
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Make sure this matches your actual celebrity images folder path
CELEB_DIR = os.path.join(BASE_DIR, "celebrity_images")

if not os.path.exists(CELEB_DIR):
    raise FileNotFoundError(f"Celebrity images folder not found: {CELEB_DIR}")

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    template_folder=os.path.join(BASE_DIR, "templates")
)

CORS(app)

# --- Load celebrity encodings once ---
celebrity_encodings = []
celebrity_names = []

for filename in os.listdir(CELEB_DIR):
    path = os.path.join(CELEB_DIR, filename)

    if not imghdr.what(path):  # Skip non-images
        continue

    try:
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            celebrity_encodings.append(encodings[0])
            celebrity_names.append(os.path.splitext(filename)[0])
            print(f"[INFO] Loaded: {filename}")
    except Exception as e:
        print(f"[ERROR] Skipping {filename}: {e}")

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
        input_image = face_recognition.load_image_file(temp_filename)

        # ✅ First find faces with the chosen algorithm
        face_locations = face_recognition.face_locations(input_image, model=algorithm)
        input_encodings = face_recognition.face_encodings(input_image, known_face_locations=face_locations)

        if not input_encodings:
            return jsonify({"error": "No face found"}), 400

        input_encoding = input_encodings[0]
        distances = face_recognition.face_distance(celebrity_encodings, input_encoding)
        best_index = np.argmin(distances)
        match_name = celebrity_names[best_index]
        similarity = 1 - distances[best_index]

        # ✅ Get full image URL
        host_url = request.host_url.rstrip('/')
        image_url = f"{host_url}/static/{match_name}.jpg"

        return jsonify({
            "match_name": match_name,
            "similarity": round(similarity * 100, 2),
            "celebrity_image": image_url
        })

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.route('/static/<filename>')
def serve_static_file(filename):
    return send_from_directory(CELEB_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)