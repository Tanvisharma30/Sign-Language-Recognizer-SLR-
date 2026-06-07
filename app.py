import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("model.pkl")

# ---------------------------
# MediaPipe Setup
# ---------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ---------------------------
# Gesture Labels
# ---------------------------
labels = ["A", "B", "C", "D", "E"]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🖐 Sign Language Recognition (A–E)")
st.write("Upload a frame or use webcam (local run recommended)")

# ---------------------------
# Gesture Guide
# ---------------------------
st.sidebar.title("📖 Gesture Guide")
st.sidebar.write("""
A → Fist  
B → Open Hand  
C → Curved Hand  
D → Point  
E → Special Gesture  
""")

# ---------------------------
# Prediction Function
# ---------------------------
def predict_hand(landmarks):
    data = np.array(landmarks).flatten().reshape(1, -1)
    pred = model.predict(data)[0]
    return pred

# ---------------------------
# Webcam (ONLY LOCAL)
# ---------------------------
run = st.checkbox("📷 Enable Webcam (Works only on local system)")

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    while run:
        success, frame = cap.read()
        if not success:
            st.error("Camera not accessible")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y])

                pred = predict_hand(landmarks)
                cv2.putText(frame, f"Prediction: {pred}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()

else:
    st.info("Enable webcam to run locally. Cloud deployment will only show UI.")