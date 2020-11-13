# Install dependencies
pwd
cd ../../
./install/install.sh
cd tools/scripts/

# Setup Experiment
export EXP_ID='seq_train_0_seq_val_10_seq_test_19'

# Generate train and test split accordingly
# python ../data_utils/ImageSetGenerators/kitti-odometry/generate_for_selected_seq.py \
# --train_seq 0012 \
# --val_seq 0012 \
# --test_seq 0012 \
# --base_path ../../data/kitti-odometry/ \
# --exp_id $EXP_ID

# # Generate pickle files with this given data

# python -m pcdet.datasets.kitti.kitti_odometry_dataset \
# create_kitti_seq_infos \
# ../../tools/cfgs/dataset_configs/kitti_odometry_dataset.yaml \
# $EXP_ID

# Train the network using the processed data created above and save the weights
# 
#     Train with multiple GPUs or multiple machines
#     ------------
#     > sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
#     or
#     > sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

#     Train with a single GPU:
#     ------------
#     python train.py --cfg_file ${CONFIG_FILE}

cd ../
pwd
python train.py --cfg_file cfgs/kitti_models/pv_rcnn_ours.yaml # Making it background
# python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml


# Eval Script using the above weights


# Demo Vis script