import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model dan label
model_path = "keras_model.h5"  # Path ke file model
labels_path = "labels.txt"  # Path ke file label

# Muat model
model = load_model(model_path)

# Muat label
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Fungsi untuk membuat prediksi
def predict(frame):
    # Resize frame ke ukuran yang diharapkan oleh model (biasanya 224x224 untuk TM)
    img_size = (224, 224)  # Pastikan ukuran sesuai dengan model Anda
    img = cv2.resize(frame, img_size)
    img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch
    img = img.astype("float32") / 255.0  # Normalisasi ke rentang [0, 1]

    # Prediksi
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]
    return labels[class_idx], confidence

# Inisialisasi kamera
camera = cv2.VideoCapture(0)  # Gunakan kamera default

if not camera.isOpened():
    print("Error: Kamera tidak dapat dibuka.")
    exit()

print("Tekan 'q' untuk keluar.")

# Loop utama
while True:
    ret, frame = camera.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Prediksi dengan model
    label, confidence = predict(frame)

    # Tampilkan hasil pada frame
    cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Teachable Machine Real-Time Detection", frame)

    # Break loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release kamera dan tutup jendela
camera.release()
cv2.destroyAllWindows()
