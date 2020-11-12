import numpy as np
import os

selected_training_seq = ["0000", "0004"]
selected_val_seq = ["0009", "0010"]

def get_file_name(file):
    return file.split(".")[0]

# Might need to maintain different file names eg. train_0,4.txt
# Generating train.txt file with the given sequences

# removing existing file
os.remove('../ImageSets/train.txt')

for seq in selected_training_seq:

    seq_path = "../training/velodyne/%s/"%(seq)
    files_list = list(map(get_file_name, os.listdir(seq_path)))

    np_arr = np.empty((len(files_list), 2), dtype="object")
    np_arr[:, 0] = seq
    np_arr[:, 1] = files_list

    with open('../ImageSets/train.txt', 'a') as f:
        np.savetxt(f, np_arr, delimiter=" ", fmt='%s')

# Generating val.txt file with the given sequences

# removing existing file
os.remove('../ImageSets/val.txt')

for seq in selected_val_seq:

    seq_path = "../training/velodyne/%s/"%(seq)
    files_list = list(map(get_file_name, os.listdir(seq_path)))

    np_arr = np.empty((len(files_list), 2), dtype="object")
    np_arr[:, 0] = seq
    np_arr[:, 1] = files_list

    with open('../ImageSets/val.txt', 'a') as f:
        np.savetxt(f, np_arr, delimiter=" ", fmt='%s')