import face_recognition
import os
import pickle
import base64

CELEB_DIR = "static"  # Your folder with celebrity images
encodings = []
names = []
images_base64 = []

for filename in os.listdir(CELEB_DIR):
    path = os.path.join(CELEB_DIR, filename)
    try:
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)
        if enc:
            encodings.append(enc[0])
            names.append(os.path.splitext(filename)[0])
            # Save image as base64
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
                images_base64.append(img_b64)
            print(f"[INFO] Added {filename}")
    except Exception as e:
        print(f"[WARN] Skipped {filename}: {e}")

data = {"encodings": encodings, "names": names, "images": images_base64}

with open("celebrity_encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print(f"[✅] Pickle created with {len(encodings)} celebrities")