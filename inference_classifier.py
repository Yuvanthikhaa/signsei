import pickle
import cv2
import mediapipe as mp
import numpy as np


try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: model.p not found! Train the model first.")
    exit()


try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    unique_labels = sorted(set(data_dict['labels']))  # Ensure correct order
    labels_dict = {i: label for i, label in enumerate(unique_labels)}
    print(f"🔤 Labels Loaded: {labels_dict}")
except FileNotFoundError:
    print(" Error: data.pickle not found! Create dataset first.")
    exit()


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 2
MAX_HANDS = 2
EXPECTED_FEATURE_LENGTH = NUM_LANDMARKS * FEATURES_PER_LANDMARK * MAX_HANDS  # 84

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Error: Unable to capture frame.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:MAX_HANDS]:  # Process up to 2 hands
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

    # Ensure correct feature length (84 features)
    while len(data_aux) < EXPECTED_FEATURE_LENGTH:
        data_aux.append(0.0)  # Pad missing values

    if len(data_aux) == EXPECTED_FEATURE_LENGTH:
        try:
            prediction = model.predict([np.asarray(data_aux)])[0]  # Extract first value
            print(f"Raw Prediction: {prediction}")  # Debugging
            if isinstance(prediction, str):
                predicted_character = prediction  # Directly use the string prediction
            else:
                predicted_label = int(prediction)
                predicted_character = labels_dict.get(predicted_label, "Unknown")
        except Exception as e:
            print(f" Prediction error: {e}")
            predicted_character = "Unknown"

        # Draw prediction text on frame
        cv2.putText(frame, predicted_character, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
