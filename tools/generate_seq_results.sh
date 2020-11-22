export EXP_ID="seq_train_15,0,3,6,9,12_seq_val_17,18,19_seq_test_1,2,4,5,13,14"
for SEQ_NUM in 0004
    do

    echo "Starting $SEQ_NUM computation"
    python demo_sequence_run.py --ckpt ../output_wk6_res/kitti_models/pv_rcnn_ours/default/ckpt/$EXP_ID/checkpoint_epoch_70.pth --cfg_file cfgs/kitti_models/pv_rcnn_ours.yaml --seq_path ../data/kitti-odometry/training/velodyne/$SEQ_NUM --output_dir ../output_wk6_res/demos_pkl/pv_rcnn_ours/training/$SEQ_NUM
    echo "Completed $SEQ_NUM computation"

    done