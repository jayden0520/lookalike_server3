# build_embeddings.py

import os
import face_recognition
import pickle
import imghdr

# Folder containing your celeb images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELEB_DIR = os.path.join(BASE_DIR, "static")

encodings = []
names = []

print("[INFO] building face embeddings...")

for filename in os.listdir(CELEB_DIR):
    path = os.path.join(CELEB_DIR, filename)
    if not imghdr.what(path):
        continue

    try:
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)
        if enc:
            encodings.append(enc[0])
            names.append(os.path.splitext(filename)[0])
            print(f"  + encoded {filename}")
    except Exception as e:
        print(f"  - skipped {filename}: {e}")

# write to pickle
data = {"encodings": encodings, "names": names}
with open("celebrity_encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("[✅] Encodings saved to celebrity_encodings.pkl")
