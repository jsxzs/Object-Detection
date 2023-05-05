# test bbox AP
python tools/test.py \
        ./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
        ./checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
        --eval bbox

# test inference speed
python -m torch.distributed.launch --nproc_per_node=1 \
        tools/benchmark.py \
        ./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
        ./checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
        --launcher pytorch