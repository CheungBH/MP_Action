import json
import argparse
import os
import numpy as np


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def extract_info(json_data, data_txt, frame_length, label_txt, label_num, sample_interval):
    kps_values = []
    frame_list = []
    # img_height = json_data["img_height"]
    # img_width = json_data["img_width"]
    for frame_number, kps_data in json_data.items():
        if frame_number.startswith("frame_"):
            frame_index = int(frame_number.split("_")[1])
            frame_list.append(frame_index)
            kps_values.append(kps_data)

    sample_num = len(kps_values) // sample_interval - 1
    for i in range(sample_num):
        label_idx = 0
        target_kps = kps_values[i*sample_interval: (i*sample_interval) + frame_length]
        target_kps_flat = np.array(target_kps).flatten()

        with open(data_txt, "a") as writer:
            kps_str = '\t'.join(f"{k:.6f}" for k in target_kps_flat)
            label_idx += 1
            # else:
            writer.write(f"{kps_str}\n")

        with open(label_txt, "a") as lb:
            for idx in range(label_idx):
                lb.write(f"{label_num}\n")


def process_files(input_dir, data_output_dir, label_output_dir, frame_length, label_num, sample_interval):
    for file in os.listdir(input_dir):
        if file.endswith(".json"):
            json_file = os.path.join(input_dir, file)
            basename = os.path.splitext(file)[0]
            data_txt = os.path.join(data_output_dir, f"{basename}.txt")
            label_txt = os.path.join(label_output_dir, f"{basename}_label.txt")

            json_data = read_json_file(json_file)
            extract_info(json_data, data_txt, frame_length, label_txt, label_num, sample_interval)


def main(args):
    input_dir = args.input_dir
    data_output_dir = args.data_output_dir
    label_output_dir = args.label_output_dir
    label_num = args.label_num
    frame_length = args.frame_length
    sample_interval = args.sample_interval

    os.makedirs(data_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    process_files(input_dir, data_output_dir, label_output_dir, frame_length, label_num, sample_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/media/hkuit164/Backup/MP_Action', help='Path to the input directory containing JSON files')
    parser.add_argument('--data_output_dir', type=str, default='tmp', help='Path to the output directory for data TXT files')
    parser.add_argument('--label_output_dir', type=str, default='tmp', help='Path to the output directory for label TXT files')
    parser.add_argument('--label_num', type=int, default=0, help='Label number for the current action')
    parser.add_argument('--frame_length', type=int, default=5, help='Interval for saving frames')
    parser.add_argument('--sample_interval', type=int, default=4, help='Interval for saving frames')

    opt = parser.parse_args()
    main(opt)
