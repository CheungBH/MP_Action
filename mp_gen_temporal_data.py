import argparse
import cv2
import mediapipe as mp
import json

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, help="Path to the video file",
                    default="/Users/cheungbh/Documents/lab_dataset/class/raw_video/fight/booker_l.mp4")
parser.add_argument("--json_path", type=str, help="Path to the output JSON file", default="test.json")
args = parser.parse_args()

video_path = args.video_path
json_path = args.json_path
cap = cv2.VideoCapture(video_path)
frame_cnt = 0
json_data = {}

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  json_data["img_height"] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  json_data["img_width"] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  while cap.isOpened():
    success, image = cap.read()
    frame_cnt += 1
    # frame_content = dict()
    # frame_content["frame_idx"] = frame_cnt
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      # sys.exit(0)
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    kps = []
    if not results.pose_landmarks:
      print(f"Cannot find pose landmarks in image {frame_cnt}")
      continue
    for i in range(len(results.pose_landmarks.landmark)):
        kps.append(results.pose_landmarks.landmark[i].x)
        kps.append(results.pose_landmarks.landmark[i].y)
    json_data["frame_" + str(frame_cnt)] = kps
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
with open(json_path, "w") as f:
    json.dump(json_data, f)
print(f"Results saved to {json_path}")