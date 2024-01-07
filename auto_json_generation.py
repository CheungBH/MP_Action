import os


src_folder = "/Users/cheungbh/Documents/lab_dataset/class/raw_video"
json_folder = "json_temporal"

actions = os.listdir(src_folder)
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
