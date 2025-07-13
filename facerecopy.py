import cv2
import os
import numpy as np
from PIL import Image

# --- Paths ---
training_data_dir = r'C:\Users\Isddu\Documents\facerecog\training_data'
model_file = r'C:\Users\Isddu\Documents\facerecog\face_model.yml'
label_file = r'C:\Users\Isddu\Documents\facerecog\labels.npy'

# --- Face recognizer and detector ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# --- Training Phase ---
print("[INFO] Training model from images...")

label_ids = {}
faces = []
labels = []
current_id = 0

# Traverse folders and collect face data
for root, dirs, files in os.walk(training_data_dir):
    for file in files:
        if file.endswith(("jpg", "jpeg", "png")):
            path = os.path.join(root, file)
            label = os.path.basename(root).lower().replace(" ", "_")

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            img = Image.open(path).convert("L")  # Grayscale
            img_np = np.array(img, "uint8")
            faces_rects = face_cascade.detectMultiScale(img_np)

            for (x, y, w, h) in faces_rects:
                roi = img_np[y:y+h, x:x+w]
                faces.append(roi)
                labels.append(id_)

if not faces:
    print("[ERROR] No face images found in training_data/. Exiting.")
    exit()

# Train the recognizer
recognizer.train(faces, np.array(labels))
recognizer.save(model_file)
np.save(label_file, label_ids)

print("[INFO] Training complete. Model and labels saved.")
print("[INFO] Starting webcam for face recognition...")

# --- Recognition Phase ---
recognizer.read(model_file)
label_ids = np.load(label_file, allow_pickle=True).item()
reverse_labels = {v: k for k, v in label_ids.items()}

# --- Start webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)

        if conf < 70:
            name = reverse_labels.get(id_, "Unknown")

            # Optional: Format name like "jogishwar_patel" → "Jogishwar Patel"
            name = name.replace("_", " ").title()

            label = f"{name} ({100 - int(conf)}% match)"
        else:
            label = "Unknown"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam closed.")
