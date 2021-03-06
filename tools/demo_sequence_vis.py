import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from visual_utils import visualize_utils as V
import pickle as pkl
import os


"""
    The demo_sequence file is split into run and vis.
    run is written to run on server where we face problem with mayavi
    vis is written to run anywhere that has mayavi working and takes in the outputs generated by run
"""

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--seq_path', type=str, default='demo_data',
                        help='specify the point cloud data sequence path')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--output_dir', type=str, default='', help='The path to save predictions and output log files for tracking')
    parser.add_argument('--saved_pred', type=str, default='', help='The path to existing saved predictions and output log files for visualizing')


    args = parser.parse_args()

    return args, {}


def main():

    args, cfg = parse_config()

    # Creating output dir if it does not already exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for res_file in os.listdir(args.saved_pred):

        file_name_parts_ = res_file.split('.')

        if file_name_parts_[-1] == 'pkl':

            with open('%s/%s.pkl' % (args.saved_pred, file_name_parts_[0]), 'rb') as f:
                data_ = pkl.load(f)

            data_dict = data_["data_dict"]
    #        pred_dicts = data_["pred_dicts"]
            pred_boxes=data_["pred_boxes"]
            pred_labels=data_["pred_labels"]
            pred_scores=data_["pred_scores"]
            gt_boxes=data_["gt_boxes"]
            
            pred_dicts=list()
            d=dict()
            d["pred_boxes"] = pred_boxes
            d["pred_labels"] = pred_labels
            d["pred_scores"] = pred_scores
            
            pred_dicts.append(d)

            Rot_matrix = np.array([
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0]
            ])

            gt_boxes[:, 0:3] = (Rot_matrix @ gt_boxes[:, 0:3].T).T
            gt_boxes[:, 3:6] = gt_boxes[:, [4, 5, 3]]
            gt_boxes[:, 2] = gt_boxes[:, 2] + gt_boxes[:, 5]/2

            fig = V.draw_scenes(
                points=data_dict, 
                gt_boxes=gt_boxes,
                ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], 
                ref_labels=pred_dicts[0]['pred_labels']
            )

            mlab.show(stop=True)
            mlab.savefig("%s/%s.png"%(args.output_dir, file_name_parts_[0]))

    # logger.info('Demo done.')


if __name__ == '__main__':  
    main()