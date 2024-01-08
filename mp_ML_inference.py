import cv2
import mediapipe as mp
import joblib
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

video_path = "/Users/cheungbh/Documents/lab_dataset/class/raw_video/fight/cody_l.mp4"
ML_path = "/Users/cheungbh/Documents/lab_code/KpsActionClassification/exp/knn_model.joblib"
ML_label = "label.txt"
with open(ML_label, 'r') as file:
    ML_classes = file.readlines()
joblib_model = joblib.load(ML_path)


from config import kpts_idx
target_idx = kpts_idx

# For webcam input:

cap = cv2.VideoCapture(video_path)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    kps_record = []
    for idx in target_idx:
        kps_record.append(results.pose_landmarks.landmark[idx].x)
        kps_record.append(results.pose_landmarks.landmark[idx].y)

    predict_nums = joblib_model.predict(np.array([kps_record]))
    # predict_action = ML_classes[int(predict_num)][:-1]
    actions = [ML_classes[int(n)][:-1] for n in predict_nums]
    print(actions)

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()