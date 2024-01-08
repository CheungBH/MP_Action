import os


src_folder = "/Users/cheungbh/Documents/lab_dataset/class/raw_video"
json_folder = "json_temporal"
label_path = ""


def get_classes(label_path, folder):
    cls_ls = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and not f.startswith(".")]
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            label_ls = [l.replace("\n", "") for l in f.readlines()]
            for label in label_ls:
                if label not in cls_ls:
                    raise Exception(
                        f"Label {label} not found in folder {folder}. Please check your label and folder name")
    else:
        print("No label file found. Use folder name as label and write to label_temporal.txt")
        label_ls = cls_ls
        with open("label_temporal.txt", "w") as f:
            for label in label_ls:
                f.write(label + "\n")
    return label_ls


actions = get_classes(label_path, src_folder)
actions_path = [os.path.join(src_folder, action) for action in os.listdir(src_folder)]
for action, action_path in zip(actions, actions_path):
    if action == ".DS_Store":
        continue
    videos_name = os.listdir(action_path)
    os.makedirs(os.path.join(json_folder, action), exist_ok=True)

    for video_name in videos_name:
        video_path = os.path.join(src_folder, action, video_name)
        target_json = os.path.join(json_folder, action, video_name.split(".")[0] + ".json")
        # os.makedirs()
        cmd = "python mp_gen_temporal_data.py --video_path {} --json_path {}".format(video_path, target_json)
        print(cmd)
        os.system(cmd)
