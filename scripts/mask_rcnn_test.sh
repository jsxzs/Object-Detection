# test bbox and segm AP
python ./tools/test.py \
        ./configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
        ./checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
        --work-dir ./work_dir \
        --eval bbox segm

# test inference speed
python -m torch.distributed.launch --nproc_per_node=1 \
        tools/benchmark.py \
        ./configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
        ./checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
        --launcher pytorch