import numpy as np
import os

selected_training_seq = ["0000", "0004"]
selected_val_seq = ["0009", "0010"]

def get_file_name(file):
    return file.split(".")[0]

# Might need to maintain different file names eg. train_0,4.txt
# Generating train.txt file with the given sequences

base_path = "../../../..//data/kitti-odometry/"

# removing existing file
train_file_path = base_path + 'ImageSets/train.txt'
if os.path.exists(train_file_path):
    os.remove(train_file_path)

for seq in selected_training_seq:

    seq_path = base_path + "training/velodyne/%s/"%(seq)
    files_list = list(map(get_file_name, os.listdir(seq_path)))

    np_arr = np.empty((len(files_list), 2), dtype="object")
    np_arr[:, 0] = seq
    np_arr[:, 1] = files_list

    with open(base_path + 'ImageSets/train.txt', 'a') as f:
        np.savetxt(f, np_arr, delimiter=" ", fmt='%s')

# Generating val.txt file with the given sequences

# removing existing file
test_file_path = base_path + 'ImageSets/val.txt'
if os.path.exists(test_file_path):
    os.remove(test_file_path)

for seq in selected_val_seq:

    seq_path = base_path + "training/velodyne/%s/"%(seq)
    files_list = list(map(get_file_name, os.listdir(seq_path)))

    np_arr = np.empty((len(files_list), 2), dtype="object")
    np_arr[:, 0] = seq
    np_arr[:, 1] = files_list

    with open(base_path + 'ImageSets/val.txt', 'a') as f:
        np.savetxt(f, np_arr, delimiter=" ", fmt='%s')