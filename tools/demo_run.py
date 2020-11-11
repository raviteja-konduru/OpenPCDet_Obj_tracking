import argparse
import glob
from pathlib import Path

# import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
# from visual_utils import visualize_utils as V
# from xvfbwrapper import Xvfb
import pickle as pkl


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


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--saved_pred', type=str, default='', help='The path to saved predictions')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():

    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    if args.saved_pred == "":
        
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()  

        with torch.no_grad():
            for idx, data_dict in enumerate(demo_dataset):
                logger.info(f'Visualized sample index: \t{idx + 1}')
                data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)

            with open('../saved_pred/curr_pickle.pkl', 'wb+') as f:
                data_ = {
                    "data_dict": data_dict['points'][:, 1:].cpu().detach().numpy(),
                    "gt_labels": pred_dicts.cpu().detach().numpy()
                    "pred_boxes": pred_dicts[0]["pred_boxes"].cpu().detach().numpy(),
                    "pred_labels": pred_dicts[0]["pred_labels"].cpu().detach().numpy(),
                    "pred_scores": pred_dicts[0]["pred_scores"].cpu().detach().numpy()		
                }
                pkl.dump(data_, f) 

    else:

        with open('../saved_pred/curr_pickle.pkl', 'rb') as f:
            data_ = pkl.load(f)

        data_dict = data_["data_dict"]
#        pred_dicts = data_["pred_dicts"]
        pred_boxes=data_["pred_boxes"]
        pred_labels=data_["pred_labels"]
        pred_scores=data_["pred_scores"]
        
        pred_dicts=list()
        d=dict()
        d["pred_boxes"] = pred_boxes
        d["pred_labels"] = pred_labels
        d["pred_scores"] = pred_scores
        
        pred_dicts.append(d)

        # vdisplay = Xvfb(width=1920, height=1080)
        # vdisplay.start()  
        # V.draw_scenes(
        #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
        #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
        # )
        # vdisplay.stop()

        # mlab.show(stop=True)
        # mlab.savefig("./test_eg.png")

    logger.info('Demo done.')


if __name__ == '__main__':  
    main()
