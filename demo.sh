python demo.py  \
    --config-file configs/fsod/finetune_vovnet.yaml \
    --input directory/*.png \
    --output results \
    --opts MODEL.WEIGHTS ./output/fsod/finetune_dir/vovnet_10shot——/model_final.pth

    #--opts CUDA_VISIBLE_DEVICES=1 --num-gpus 1