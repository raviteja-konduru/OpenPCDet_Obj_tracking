import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, box_utils, calibration_kitti
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

import pickle as pkl
import os

from visual_utils import visualize_utils as V
import pickle as pkl
import os
from pcdet.datasets.kitti.kitti_object_eval_python.kitti_common import get_label_anno as gla

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def get_calib(cfg, seq_num):

    root_split_path = Path("../data/kitti-odometry/training/calib")
    calib_file = root_split_path / ('%s.txt' % seq_num)
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the nn model for similarity dataset generation')
    parser.add_argument('--seq_path', type=str, default='demo_data',
                        help='specify the point cloud data sequence path')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--output_dir', type=str, default='../data/kitti-similarity/', help='The path to save generated data to')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():

    args, cfg = parse_config()

    curr_seq = args.seq_path.split("/")[-1]

    # Map dictionary - string labels --> int
    map_dict = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4, 'Person_sitting': 4, 'Truck': 5}
    
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.seq_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # Loading the ground truth from kitti-odometry
    gt_path = "/".join(args.seq_path.split("/")[0:-2]) + "/label_02/" + curr_seq + ".txt"
    # gt_data = np.genfromtxt(gt_path, dtype=str)[:, [0, 2, 13, 14, 15, 10, 11, 12, 16, 1]]
    gt_data = np.genfromtxt(gt_path, dtype=str)[:, [0, 2, 13, 14, 15, 12, 10, 11, 16, 1]]

    gt_data = gt_data[gt_data[:, 1] != 'DontCare', :]
    gt_labels=gt_data[:, 1].reshape((-1, 1))
    print(gt_labels.shape)
    gt_labels=np.apply_along_axis(lambda x : -2 if map_dict.get(x[0]) is None else map_dict.get(x[0]), 1, gt_labels)
    # -2 ==> not don't care or any other object we have in the map
    print(gt_labels.dtype, gt_labels.shape)
    print(gt_labels[0:10])

    # Adding the object tracking id as the last column
    gt_data = gt_data[:, [0, 2, 3, 4, 5, 6, 7, 8, 9]]

    # Converting to floats
    gt_data = gt_data.astype(np.float)

    # Set IOU threshold
    IOU_THRESH = 0.5

    with torch.no_grad():

        for idx, data_dict in enumerate(demo_dataset):

            logger.info(f'Visualized sample index: \t{idx + 1}')

            # shape: Num_objs * 8
            mask = gt_data[:, 0] == idx
            relevant_gt_boxes = gt_data[mask][:, 1:-1]
            relevant_gt_ids = gt_data[mask][:, -1]
            relevant_gt_labels = gt_labels[mask]

            data_dict = demo_dataset.collate_batch([data_dict])

            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # Creating output dir if it does not already exist
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            pred_scores=pred_dicts[0]["pred_scores"].cpu().detach().numpy()
            gt_scores=np.ones_like(relevant_gt_labels)
            
            pred_labels=pred_dicts[0]["pred_labels"].cpu().detach().numpy()
            
            print("pred_labels", pred_labels)
            print("relevant_gt_labels", relevant_gt_labels)

            # assert(pred_labels.shape == relevant_gt_labels.shape)
            assert(pred_labels.dtype == relevant_gt_labels.dtype)

            # To transform relevant_gt_boxes to lidar coordinates
            calib = get_calib(cfg, curr_seq)
            relevant_gt_boxes_trnsfm = box_utils.boxes3d_kitti_camera_to_lidar(relevant_gt_boxes, calib)

            data_ = {
                "data_dict": data_dict['points'][:, 1:].cpu().detach().numpy(),
                "pred_boxes": pred_dicts[0]["pred_boxes"].cpu().detach().numpy(),
                "pred_labels": pred_dicts[0]["pred_labels"].cpu().detach().numpy(),
                "pred_scores": pred_dicts[0]["pred_scores"].cpu().detach().numpy(),
                "gt_boxes": relevant_gt_boxes_trnsfm,
                "gt_labels": relevant_gt_labels,
                "gt_scores": gt_scores,
                "gt_ids": relevant_gt_ids
                # "pooled_features": pred_dicts[0]["pooled_features"].cpu().detach().numpy()
            }
            pooled_features = pred_dicts[0]["pooled_features"].cpu().detach().numpy()

            print('data_["pred_boxes"].shape', data_["pred_boxes"].shape)
            print('data_["pred_labels"].shape', data_["pred_labels"].shape)
            print('data_["pred_scores"].shape', data_["pred_scores"].shape)

            # print("pooled_features.shape", data_["pooled_features"].shape)
            # print("pooled_features.dtype", data_["pooled_features"].dtype)

            # # double for loop
            # matched_detections_ind = []
            # for i in range(data_["gt_boxes"].shape[0]):
            #     curr_gt_box = data_["gt_boxes"][i, :]
            #     scores = []

            #     for j in range(data_["pred_boxes"].shape[0]):
            #         # compute IoU between ith ground truth box, and jth predicted box.
            #         # pooled_features[j] is the jth boxes pooled features to save if iou > IOU_THRESH
            #         # seq_num/obj_id/frame.npy -> ../data/kitti-similarity/{curr_seq}/{gt_boxes[-1]}/{idx}.npy
            #         curr_prediction = data_["pred_boxes"][j, :]
            #         curr_iou_score = "TODO" #TODO

            # IoU Scores Computation
            device='cuda' if torch.cuda.is_available() else 'cpu'
            print("device", device)
            gt_boxes_t = torch.FloatTensor(data_["gt_boxes"]).to(device)
            pred_boxes_t = torch.FloatTensor(data_["pred_boxes"]).to(device)
            """
                gt_boxes_t - shape: (N, 7)
                pred_boxes_t - shape: (M, 7)
                iou_scores - shape: (N, M), N=num gt objects, m=num of predicted objects
            """
            iou_scores = boxes_iou3d_gpu(gt_boxes_t, pred_boxes_t)

            """
                idea: 
                    We expect that there won't be more than one match for a gt box with iou > 0.5 or threshold
                Implementation: 
                    We match only one box to a gt through argmax
                    => each gt box cannot have more than one match     
            """
            matched_indices = []
            sorted_inds = torch.argsort(iou_scores, dim=1, descending=True)
            for i in range(iou_scores.shape[0]):
                for j in range(iou_scores.shape[1]):
                    
                    curr_pred_ind = sorted_inds[i, j].item()

                    if curr_pred_ind in matched_indices:
                        break

                    curr_iou = iou_scores[i, curr_pred_ind].item()

                    if curr_iou < IOU_THRESH:
                        break
                    
                    if data_["pred_labels"][curr_pred_ind] != relevant_gt_labels[i]:
                        break

                    matched_indices.append(curr_pred_ind)
                    # Save pooled features to ../data/kitti-similarity/{curr_seq}/{gt_boxes[-1]}/{idx}.npy
                    save_dir = '../data/kitti-similarity/{}/{}/'.format(curr_seq, int(relevant_gt_ids[i]))
                    save_path = '{}/{}.npy'.format(save_dir, idx)

                    curr_feature = pooled_features[curr_pred_ind]
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    np.save(save_path, curr_feature)
                    
                    # Saving label of object
                    label_save_path = '{}/label.txt'.format(save_dir)
                    if not os.path.exists(label_save_path):
                        f = open(label_save_path, "a")
                        f.write(str(relevant_gt_labels[i]))
                        f.close()


            print(iou_scores.shape)
            print(iou_scores.dtype)
            print(type(iou_scores))
            print(iou_scores)

            # with open('%s/curr_pickle_%s.pkl' % (args.output_dir, str(idx)), 'wb+') as f:
            #     pkl.dump(data_, f) 

            # # Writing to text file in kitti format for tracking step
            # frame_data = np.zeros((data_["pred_labels"].shape[0], 15))
            # frame_data[:, 0] = idx # Frame ID
            # frame_data[:, 1] = data_["pred_labels"] # Labels
            # frame_data[:, 2:6] = 0 # 2d bounding boxes
            # frame_data[:, 6] = data_["pred_scores"] # 2d bounding boxes
            # frame_data[:, 7:10]= data_["pred_boxes"][:, 3:6]
            # frame_data[:, 10:13]= data_["pred_boxes"][:, 0:3]
            # frame_data[:, 13]= data_["pred_boxes"][:, -1]
            # frame_data[:, 14]= 0 # Alpha

            # with open('%s/%s.csv' % (args.output_dir, curr_seq), 'a') as f:
            #     np.savetxt(f, frame_data, delimiter=",")

if __name__ == '__main__':  
    main()
