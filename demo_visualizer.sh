python demo_visualizer.py  \
    --config-file configs/fsod/finetune_R_50_C4_1x.yaml \
    --input directory/*.png \
    --output results \
    --opts MODEL.WEIGHTS ./output/fsod/finetune_dir/model_final.pth 

    #--opts CUDA_VISIBLE_DEVICES=1 --num-gpus 1