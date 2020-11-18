import numpy as np
import os
import argparse
from pathlib import Path
import numpy as np


def get_file_name(file):
    return file.split(".")[0]

def generate_file_with_given_seq(base_path, data_ref, file_name, selected_seq, exp_id):
    
    # Paths for tracking Label and path to save it
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../../').resolve()
    root_split_path = ROOT_DIR / 'data' / 'kitti-odometry'/ ('training' if data_ref != 'test' else 'testing')

    # Creating directory
    Path(base_path + 'ImageSets/' + exp_id + "/").mkdir(parents=True, exist_ok=True)

    # removing existing file
    train_file_path = base_path + 'ImageSets/%s/%s.txt' % (exp_id, file_name)
    if os.path.exists(train_file_path):
        os.remove(train_file_path)

    for seq in selected_seq:

        if data_ref == "train":
            data_ref_path = "training"
        else:
            data_ref_path = "testing"

        seq_path = base_path + data_ref_path + "/velodyne/%s/"%(seq)
        files_list = list(map(get_file_name, os.listdir(seq_path)))

        np_arr = np.empty((len(files_list), 2), dtype="object")
        np_arr[:, 0] = seq
        np_arr[:, 1] = files_list

        # Removing items with no objects or just don't cares
        if data_ref != "test":
            object_count_array = np.apply_along_axis(lambda x: create_lidar_file_and_get_number_of_items(x, root_split_path), 1, np_arr)

            # Generating mask for non-zero objects
            mask_non_zero = object_count_array != 0
            print(seq, np_arr.shape, mask_non_zero.shape)
            np_arr = np_arr[mask_non_zero, :]
            print(seq, np_arr.shape, np.sum(mask_non_zero))

        with open(base_path + 'ImageSets/%s/%s.txt' % (exp_id, file_name), 'a') as f:
            np.savetxt(f, np_arr, delimiter=" ", fmt='%s')

def create_lidar_file_and_get_number_of_items(idx_info, root_split_path):

    seq = idx_info[0]
    idx = idx_info[1]

    label_file_parent = root_split_path / 'label_02' / ('%s.txt' % seq)
    label_file = root_split_path / 'label_02_splits' / ('%s_%s.txt' % (seq, idx))    

    obj_count = 0

    if label_file.exists():
        with open(label_file, 'r') as f:
            label_data = np.genfromtxt(f, dtype=str, delimiter=' ')
            
            if len(label_data.shape) == 2:
                mask = label_data[:, 0] != 'DontCare'
            elif len(label_data.shape) == 1:
                label_data = label_data.reshape((-1, 1))
                mask = label_data[:, 0] != 'DontCare'
            else:
                mask = np.array([])

            obj_count = int(np.sum(mask))
    
    else: 
        with open(label_file_parent, 'r') as f:

            lines = np.genfromtxt(f, dtype=str, delimiter=' ')
            # Need columns 2-end
            lines = lines[lines[:, 0].astype(int) == int(idx)]
            label_data = lines[:, 2:]
            
            # Creating the file
            with open(label_file, 'w+') as f_local:
                np.savetxt(f_local, np.array(label_data), delimiter=" ", fmt='%s')

            if len(label_data.shape) == 2:
                mask = label_data[:, 0] != 'DontCare'
            elif len(label_data.shape) == 1:
                label_data = label_data.reshape((-1, 1))
                mask = label_data[:, 0] != 'DontCare'
            else:
                mask = np.array([])

            obj_count = int(np.sum(mask))

    return obj_count

def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--train_seq', type=str, default='',
                        help='comma separated sequences train list')
    parser.add_argument('--val_seq', type=str, default='',
                        help='comma separated sequences val list')     
    parser.add_argument('--test_seq', type=str, default='',
                        help='comma separated sequences test list')                             
    parser.add_argument('--base_path', type=str, default='"../../../../data/kitti-odometry/"',
                        help='base path of the data directory')
    parser.add_argument('--exp_id', type=str, default='',
                        help='experiment id')                        

    args = parser.parse_args()

    return args


if __name__ == '__main__':  
    args = parse_args()

    train_sequences = args.train_seq.split(",")
    val_sequences = args.val_seq.split(",")
    test_sequences = args.test_seq.split(",")

    base_path = args.base_path
    exp_id = args.exp_id

    # # Generate train.txt
    generate_file_with_given_seq(
        base_path = base_path, 
        data_ref="train",
        file_name="train",
        selected_seq=train_sequences,
        exp_id=exp_id
    )

    # # Generate val.txt
    generate_file_with_given_seq(
        base_path = base_path, 
        data_ref="train",
        file_name="val",
        selected_seq=val_sequences,
        exp_id=exp_id
    )    

    # Generate test.txt
    generate_file_with_given_seq(
        base_path = base_path, 
        data_ref="test",
        file_name="test",
        selected_seq=val_sequences,
        exp_id=exp_id
    )        