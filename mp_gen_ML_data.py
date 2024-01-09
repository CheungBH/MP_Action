import cv2
import mediapipe as mp
import numpy as np
import os
import csv

from config import kpts_idx
target_idx = kpts_idx

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def process_action(image_folder, output_folder, csv_path, info):
    IMAGE_FILES = os.listdir(image_folder)
    BG_COLOR = (192, 192, 192) # gray
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
      for idx, file in enumerate(IMAGE_FILES):
        if idx % 50 == 0:
            print(f"Processing {idx}th image: {file}")
        kps_record = []
        file_name = os.path.join(image_folder, file)
        result_file_name = os.path.join(output_folder, file)
        image = cv2.imread(file_name)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
          print(f"Cannot find pose landmarks in image {file_name}")
          continue

        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        if output_folder:
            cv2.imwrite(result_file_name, annotated_image)

        for idx in target_idx:
            kps_record.append(results.pose_landmarks.landmark[idx].x)
            kps_record.append(results.pose_landmarks.landmark[idx].y)
        # print(kps_record)
        # print(len(kps_record))
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([str(i) for i in kps_record]+info+[file_name])


if __name__ == '__main__':


    # image_folder = "/Users/cheungbh/Documents/lab_dataset/mediapipe_auto"
    # output_folder = "test"
    # csv_path = "/Users/cheungbh/Documents/lab_dataset/mediapipe_test/folder1_output/cody.csv"
    # info = ["0", "stand"]
    # process_action(image_folder, output_folder, csv_path, info)


    def get_classes(label_path, folder):
        cls_ls = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and not f.startswith(".")]
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                label_ls = [l.replace("\n", "") for l in f.readlines()]
                for label in label_ls:
                    if label not in cls_ls:
                        raise Exception(f"Label {label} not found in folder {folder}. Please check your label and folder name")
        else:
            print("No label file found. Use folder name as label and write to label.txt")
            label_ls = cls_ls
            with open("label.txt", "w") as f:
                for label in label_ls:
                    f.write(label + "\n")
        return label_ls

    image_src = "images/squat"
    output_src = "tmp/squat"
    csv_path = "squat.csv"
    assert csv_path and image_src, "Please specify image_src and csv_path"
    label_path = ""
    classes = get_classes(label_path, image_src)
    for idx, cls in enumerate(classes):
        if idx % 50 == 0:
            print(f"Processing {idx}th class: {cls}")
        info = [str(idx), cls]
        image_folder = os.path.join(image_src, cls)
        output_folder = os.path.join(output_src, cls) if output_src else ""
        process_action(image_folder, output_src, csv_path, info)


