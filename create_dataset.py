import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

NUM_LANDMARKS = 21  # Each hand has 21 landmarks
FEATURES_PER_LANDMARK = 2  # x, y coordinates
MAX_HANDS = 2  # Store max 2 hands
EXPECTED_FEATURE_LENGTH = NUM_LANDMARKS * FEATURES_PER_LANDMARK * MAX_HANDS  # 84

if not os.path.exists(DATA_DIR):
    print(" Error: DATA_DIR not found!")
else:
    print(f"Processing dataset in {DATA_DIR}")

    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)

        # Skip non-directory files (like .gitignore)
        if not os.path.isdir(dir_path):
            print(f"Skipping non-directory file: {dir_}")
            continue

        print(f" Processing label: {dir_}")

        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)
            print(f"🖼️ Processing image: {img_path}")

            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(img_full_path)
            if img is None:
                print(f"Error loading {img_path}, skipping.")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks[:MAX_HANDS]:  # Process up to 2 hands
                    for i in range(len(hand_landmarks.landmark)):
                        x_.append(hand_landmarks.landmark[i].x)
                        y_.append(hand_landmarks.landmark[i].y)

                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                        data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                #  Ensure all samples have the same length (84 features)
                while len(data_aux) < EXPECTED_FEATURE_LENGTH:
                    data_aux.append(0.0)  # Pad missing hands with zeros

                data.append(data_aux)
                labels.append(dir_)

print(f" Collected {len(data)} samples.")

if len(data) > 0:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(" data.pickle saved successfully!")
else:
    print(" No data was collected. Check image files and preprocessing.")
