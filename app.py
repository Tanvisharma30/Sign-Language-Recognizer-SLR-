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
    page_title="SLR System",
    page_icon="🖐",
    layout="wide"
)

# -----------------------------
# SIMPLE DARK UI CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    margin-bottom: 0px;
}

.subtitle {
    text-align: center;
    color: #aaa;
    font-size: 16px;
    margin-bottom: 20px;
}

.card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 15px;
    border: 1px solid #2c2f36;
}

.big-text {
    font-size: 40px;
    font-weight: bold;
}

.status {
    font-size: 14px;
    color: #aaa;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# TEXT TO SPEECH
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# -----------------------------
# MEDIAPIPE
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
st.markdown("<div class='title'>Sign Language Recognition</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time A–E Gesture Detection using MediaPipe + ML</div>", unsafe_allow_html=True)

# -----------------------------
# LAYOUT
# -----------------------------
left, right = st.columns([3, 1])

# -----------------------------
# RIGHT PANEL
# -----------------------------
with right:
    st.markdown("### Controls")

    run = st.checkbox("Start Camera")

    if st.button("Reset Word"):
        st.session_state.word = ""
        st.session_state.last_added = ""

    if st.button("Speak Word"):
        if st.session_state.word:
            engine.say(st.session_state.word)
            engine.runAndWait()

    st.markdown("### Gesture Guide")

    st.table({
        "Letter": ["A", "B", "C", "D", "E"],
        "Gesture": [
            "Fist",
            "Open Palm",
            "Index Finger",
            "OK Sign",
            "Peace Sign"
        ]
    })

# -----------------------------
# LEFT PANEL
# -----------------------------
with left:
    frame_placeholder = st.empty()
    prediction_placeholder = st.empty()
    confidence_bar = st.empty()
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

            features = []
            wrist_x = handLms.landmark[0].x
            wrist_y = handLms.landmark[0].y

            for lm in handLms.landmark:
                features.extend([lm.x - wrist_x, lm.y - wrist_y])

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
    # WORD BUILDING (UNCHANGED LOGIC)
    # -----------------------------
    if (
        confidence > 0.8 and
        prediction != "" and
        prediction != st.session_state.last_added
    ):
        st.session_state.word += prediction
        st.session_state.last_added = prediction

    # -----------------------------
    # UI DISPLAY (IMPROVED ONLY)
    # -----------------------------
    frame_placeholder.image(frame, channels="BGR", use_container_width=True)

    prediction_placeholder.markdown(f"""
    <div class="card">
        <div class="status">Current Prediction</div>
        <div class="big-text">{prediction}</div>
    </div>
    """, unsafe_allow_html=True)

    confidence_bar.progress(float(confidence))

    word_placeholder.markdown(f"""
    <div class="card">
        <div class="status">Detected Word</div>
        <div class="big-text">{st.session_state.word}</div>
    </div>
    """, unsafe_allow_html=True)

cap.release()