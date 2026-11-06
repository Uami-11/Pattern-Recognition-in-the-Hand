import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    # Flip camera for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks and connections
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Landmark Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()

def recognize_gesture(hand_landmarks):

    landmarks = hand_landmarks.landmark

    # Finger tip and PIP joint positions
    fingers = [
        (8, 6),    # Index
        (12, 10), # Middle
        (16, 14), # Ring
        (20, 18)  # Pinky
    ]

    extended_fingers = 0

    for tip, joint in fingers:

        if landmarks[tip].y < landmarks[joint].y:
            extended_fingers += 1

    if extended_fingers == 4:
        return "OPEN PALM"

    elif extended_fingers == 0:
        return "FIST"

    elif extended_fingers == 2:
        return "TWO FINGERS"

    else:
        return "UNKNOWN"

gesture = recognize_gesture(hand_landmarks)

cv2.putText(
    frame,
    gesture,
    (50, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)