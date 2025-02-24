import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset dari file CSV
color_data = pd.read_csv('colors.csv')
X = color_data[['R', 'G', 'B']].values
y = color_data['ColorName'].values

# Normalisasi Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prediksi data test dan menghitung akurasi
y_pred = knn.predict(X_test)
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {overall_accuracy * 100:.2f}%")

cap = cv2.VideoCapture(0)

detected_colors = []
detected_true_labels = []

def find_nearest_color(avg_color):
    distances = np.linalg.norm(X - avg_color, axis=1)
    nearest_idx = np.argmin(distances)
    return y[nearest_idx]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ambil ukuran frame
    height, width, _ = frame.shape
    
    # Tentukan koordinat bounding box di tengah layar
    box_size = 100
    x_start, y_start = (width // 2 - box_size // 2, height // 2 - box_size // 2)
    x_end, y_end = (x_start + box_size, y_start + box_size)
    
    # Ambil warna dari area tengah
    roi = frame[y_start:y_end, x_start:x_end]
    avg_color = np.mean(roi, axis=(0, 1)).astype(int)  # Ambil warna rata-rata
    
    # Normalisasi warna
    avg_color_scaled = scaler.transform([avg_color])
    
    # Prediksi warna
    color_pred = knn.predict(avg_color_scaled)[0]
    true_color = find_nearest_color(avg_color)
    
    # Simpan prediksi dan warna asli untuk perhitungan akurasi
    detected_colors.append(color_pred)
    detected_true_labels.append(true_color)
    
    # Hitung akurasi deteksi warna secara real-time
    if len(detected_colors) > 50:
        detected_colors.pop(0)
        detected_true_labels.pop(0)
    
    color_accuracy = accuracy_score(detected_true_labels, detected_colors) * 100 if detected_colors else 0.0
    
    # Cetak akurasi ke terminal
    print(f"Deteksi Warna: {color_pred} | Warna Asli: {true_color} | Akurasi Deteksi Warna: {color_accuracy:.2f}%")
    
    # Gambar bounding box
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    
    # Tambahkan label warna pada bounding box
    cv2.putText(frame, f'{color_pred}', (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Tampilkan informasi warna dan akurasi
    cv2.putText(frame, f'Color: {color_pred}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f'True Color: {true_color}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'Accuracy: {color_accuracy:.2f}%', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
