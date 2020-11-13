for SEQ_NUM in 0001 0002
    do

    echo "Starting $SEQ_NUM computation"
    python demo_sequence.py --ckpt ../checkpoints/online/pv_rcnn_8369.pth --cfg_file cfgs/kitti_models/pv_rcnn.yaml --seq_path ../data/kitti-odometry/training/velodyne/$SEQ_NUM --output_dir ../output/pv_rcnn/training/$SEQ_NUM
    echo "Completed $SEQ_NUM computation"

    done

# for SEQ_NUM in 0000 0001 0002 0003 0004 0005 0006 0007 0008 0009 0010 0011 0012 0013 0014 0015 0016 0017 0018 0019 0020
#     do

#     echo "Starting $SEQ_NUM computation"
#     python demo_sequence.py --ckpt ../checkpoints/online/pointrcnn_7870.pth --cfg_file cfgs/kitti_models/pointrcnn.yaml --seq_path ../data/kitti-odometry/training/velodyne/$SEQ_NUM --output_dir ../output/pointrcnn/training/$SEQ_NUM
#     echo "Completed $SEQ_NUM computation"

#     done

# python demo_sequence.py --ckpt ../checkpoints/online/pv_rcnn_8369.pth --cfg_file cfgs/kitti_models/pv_rcnn.yaml --seq_path ../data/kitti-odometry/training/velodyne/0000 --output_dir ../output/pv_rcnn/training/0000
