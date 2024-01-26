import cv2
import mediapipe as mp
from temporal.utils import TemporalQueue
from temporal.model_1d import TemporalSequenceModel
import torch

device = "cpu"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

video_path = "/media/hkuit164/Backup/xjl/20231207_kpsVideo/det_video/daniel.mp4"

temporal_label = "/media/hkuit164/Backup/xjl/20231207_kpsVideo/temp_data/train_input/cls.txt"
with open(temporal_label, 'r') as file:
    temporal_classes = file.readlines()

n_classes = len(temporal_classes)

temporal_module = "BiLSTM"
model_path = '/media/hkuit164/Backup/TemporalClassifier/exp/mp_test/BiLSTM_lr0.001/model.pth'
kps_num = 66
hidden_dims, num_rnn_layers, attention = [64, 2, False]
temporal_model = TemporalSequenceModel(num_classes=n_classes, input_dim=kps_num, hidden_dims=hidden_dims,
                                       num_rnn_layers=num_rnn_layers, attention=attention,
                                       temporal_module=temporal_module)
temporal_model.load_state_dict(torch.load(model_path))
temporal_model.to(device)
temporal_model.eval()
kps_queue = TemporalQueue(5, 1)

target_idx = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


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
      # continue
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    kps_record = []
    for idx in range(33):
        kps_record.append(results.pose_landmarks.landmark[idx].x)
        kps_record.append(results.pose_landmarks.landmark[idx].y)

    kps_queue.update(kps_record)
    kps = kps_queue.get_data()
    # kps = torch.tensor(kps).unsqueeze(0).to(device)
    if kps is not None:
        kps = torch.tensor(kps).unsqueeze(0).to(device)
    outputs = temporal_model(kps)
    # print(Softmax(outputs))
    if outputs is not None:
        temporal_pred = outputs.data.max(1)[1]
        actions = [temporal_classes[i][:-1] for i in temporal_pred]
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
