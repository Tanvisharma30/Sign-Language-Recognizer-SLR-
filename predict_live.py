import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter

# Load model
model = joblib.load("model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# For smoothing predictions
prediction_buffer = deque(maxlen=10)

# Word building variables
word = ""
last_added = ""

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    stable_prediction = ""
    confidence = 0

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

            # NORMALIZED FEATURES
            features = []

            wrist_x = handLms.landmark[0].x
            wrist_y = handLms.landmark[0].y

            for lm in handLms.landmark:
                features.extend([
                    lm.x - wrist_x,
                    lm.y - wrist_y
                ])

            features = np.array(features).reshape(1, -1)

            # Prediction
            prediction = model.predict(features)[0]
            prediction_buffer.append(prediction)

    # Smooth prediction
    if prediction_buffer:
        most_common = Counter(prediction_buffer).most_common(1)[0]
        stable_prediction = most_common[0]
        confidence = most_common[1] / len(prediction_buffer)

    # Add letter to word only if stable
    if confidence > 0.8 and stable_prediction != last_added:
        word += stable_prediction
        last_added = stable_prediction

    # Display prediction
    cv2.putText(
        frame,
        f"Prediction: {stable_prediction}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display confidence
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Display word
    cv2.putText(
        frame,
        f"Word: {word}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    # Reset word
    if key == ord('r'):
        word = ""
        last_added = ""
        print("Word reset")

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()