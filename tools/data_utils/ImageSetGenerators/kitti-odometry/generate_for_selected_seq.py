import numpy as np
import os
import argparse
from pathlib import Path

def get_file_name(file):
    return file.split(".")[0]

def generate_file_with_given_seq(base_path, file_name, selected_seq, exp_id):
    
    # Creating directory
    Path(base_path + 'ImageSets/' + exp_id + "/").mkdir(parents=True, exist_ok=True)

    # removing existing file
    train_file_path = base_path + 'ImageSets/%s/%s.txt' % (exp_id, file_name)
    if os.path.exists(train_file_path):
        os.remove(train_file_path)

    for seq in selected_seq:

        seq_path = base_path + "training/velodyne/%s/"%(seq)
        files_list = list(map(get_file_name, os.listdir(seq_path)))

        np_arr = np.empty((len(files_list), 2), dtype="object")
        np_arr[:, 0] = seq
        np_arr[:, 1] = files_list

        with open(base_path + 'ImageSets/%s/%s.txt' % (exp_id, file_name), 'a') as f:
            np.savetxt(f, np_arr, delimiter=" ", fmt='%s')

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

    # Generate train.txt
    generate_file_with_given_seq(
        base_path = base_path, 
        file_name="train",
        selected_seq=train_sequences,
        exp_id=exp_id
    )

    # Generate val.txt
    generate_file_with_given_seq(
        base_path = base_path, 
        file_name="val",
        selected_seq=val_sequences,
        exp_id=exp_id
    )    

    # Generate test.txt
    generate_file_with_given_seq(
        base_path = base_path, 
        file_name="test",
        selected_seq=val_sequences,
        exp_id=exp_id
    )        