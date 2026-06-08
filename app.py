import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import pyttsx3

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🖐",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# TEXT TO SPEECH ENGINE
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# -----------------------------
# SESSION STATE
# -----------------------------
if "word" not in st.session_state:
    st.session_state.word = ""

if "last_added" not in st.session_state:
    st.session_state.last_added = ""

if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=10)

# -----------------------------
# HEADER
# -----------------------------
st.title("🖐 Real-Time Sign Language Recognition System")
st.markdown("AI-based hand gesture recognition using MediaPipe + ML")

# -----------------------------
# LAYOUT
# -----------------------------
left_col, right_col = st.columns([3, 1])

# -----------------------------
# RIGHT PANEL (CONTROLS)
# -----------------------------
with right_col:

    st.subheader("⚙ Controls")

    run = st.checkbox("Start Camera")

    if st.button("🔄 Reset Word"):
        st.session_state.word = ""
        st.session_state.last_added = ""

    if st.button("🔊 Speak Word"):

        if st.session_state.word != "":
            engine.say(st.session_state.word)
            engine.runAndWait()

    st.divider()

    st.subheader("📖 Gesture Guide")

    st.markdown("""
    **A** → ✊ Fist  

    **B** → ✋ Open Palm  

    **C** → ☝️ Index Finger  

    **D** → 👌 OK Sign  

    **E** → ✌️ Peace Sign  
    """)

# -----------------------------
# LEFT PANEL
# -----------------------------
with left_col:

    frame_placeholder = st.empty()
    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()

    st.subheader("📝 Detected Word")
    word_placeholder = st.empty()

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

while run:

    success, frame = cap.read()
    if not success:
        st.error("Camera not accessible")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    prediction = ""
    confidence = 0

    if result.multi_hand_landmarks:

        for handLms in result.multi_hand_landmarks:

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                handLms,
                mp.solutions.hands.HAND_CONNECTIONS
            )

            # -------------------------
            # NORMALIZED FEATURES
            # -------------------------
            features = []

            wrist_x = handLms.landmark[0].x
            wrist_y = handLms.landmark[0].y

            for lm in handLms.landmark:
                features.extend([
                    lm.x - wrist_x,
                    lm.y - wrist_y
                ])

            features = np.array(features).reshape(1, -1)

            pred = model.predict(features)[0]
            st.session_state.buffer.append(pred)

    # -----------------------------
    # SMOOTHING
    # -----------------------------
    if st.session_state.buffer:

        most_common = Counter(st.session_state.buffer).most_common(1)[0]

        prediction = most_common[0]
        confidence = most_common[1] / len(st.session_state.buffer)

    # -----------------------------
    # WORD BUILDING
    # -----------------------------
    if (
        confidence > 0.8 and
        prediction != "" and
        prediction != st.session_state.last_added
    ):
        st.session_state.word += prediction
        st.session_state.last_added = prediction

    # -----------------------------
    # DISPLAY UI
    # -----------------------------
    frame_placeholder.image(frame, channels="BGR", use_container_width=True)

    prediction_placeholder.success(f"Prediction: {prediction}")

    confidence_placeholder.info(f"Confidence: {confidence:.2f}")

    word_placeholder.markdown(f"# {st.session_state.word}")

cap.release()