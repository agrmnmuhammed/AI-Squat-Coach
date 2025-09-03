import cv2
import mediapipe as mp
import numpy as np

# MediaPipe ve diğer gerekli kütüphaneleri başlat
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    Üç nokta arasındaki açıyı derece cinsinden hesaplar.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))

    if angle > 180.0:
        angle = 360 - angle

    return angle

cap = cv2.VideoCapture(0)

# Squat sayacı ve durum değişkenleri
counter = 0
state = "UP" # Durum: 'UP', 'DOWN', 'PARTIAL' (kısmi)
last_feedback = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Noktaları al
        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
        left_knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h))
        left_ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h))

        right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
        right_knee = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * h))
        right_ankle = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h))

        # Diz açısını hesapla
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_angle = calculate_angle(right_hip, right_knee, right_ankle)
        avg_angle = (left_angle + right_angle) / 2

        # Geri bildirim ve sayaç mantığı
        feedback = ""
        
        # DOWN durumu
        if avg_angle < 100:
            state = "DOWN"
            if avg_angle < 70:
                feedback = "Cok fazla iniyorsun ⚠️ Dizine dikkat et"
            elif 70 <= avg_angle <= 90:
                feedback = "Derinlik iyi, Squat dogru ✅"
            else:
                feedback = "Daha asagi in ⚠️"

        # UP durumu (Tamamlanmış döngü)
        elif avg_angle > 150:
            feedback = "Yukaridasin, yeni tekrar icin hazir ol 👍"
            if state == "DOWN":
                counter += 1
                state = "UP"
        
        # Geri bildirimlerin önceliği
        if avg_angle < 160:
            left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
            shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
            current_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h

            if current_hip_y > shoulder_y + 30:
                if "✅" in feedback:
                    feedback += " ve sırtını daha dik tut!"
                elif not "⚠️" in feedback and not "👍" in feedback:
                    feedback = "Sirtini dik tutmaya calis ⚠️"

        # Ekrana metinleri yazdır
        feedback_color = (0, 0, 255)
        if "✅" in feedback:
            feedback_color = (0, 255, 0)
        elif "👍" in feedback:
            feedback_color = (0, 255, 255)

        cv2.putText(frame, f"Diz Acisi: {int(avg_angle)}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        if feedback:
            last_feedback = feedback

        cv2.putText(frame, last_feedback, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2)

        cv2.putText(frame, f"Squat Sayisi: {counter}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Hocam Squat Analizi", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()