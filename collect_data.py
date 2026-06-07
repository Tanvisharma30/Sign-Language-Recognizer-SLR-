import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

file_name = "dataset.csv"

# Create CSV if not exists
if not os.path.exists(file_name):
    with open(file_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}"]
        header.append("label")
        writer.writerow(header)

cap = cv2.VideoCapture(0)

label = "A"
collect = False

print("Controls:")
print("S = Start/Stop collecting")
print("A/B/C/D/E = change label")
print("Q = quit")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    features = []

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

            # 🔥 NORMALIZATION START (key fix)
            wrist_x = handLms.landmark[0].x
            wrist_y = handLms.landmark[0].y

            for lm in handLms.landmark:
                features.append(lm.x - wrist_x)
                features.append(lm.y - wrist_y)

    cv2.putText(frame,
                f"Label: {label} | Collecting: {collect}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Label switching
    if key == ord('a'):
        label = "A"
    if key == ord('b'):
        label = "B"
    if key == ord('c'):
        label = "C"
    if key == ord('d'):
        label = "D"
    if key == ord('e'):
        label = "E"

    # Start/Stop collecting
    if key == ord('s'):
        collect = not collect
        print("Collecting:", collect)

    # Save data
    if collect and len(features) == 42:
        with open(file_name, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = features + [label]
            writer.writerow(row)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()