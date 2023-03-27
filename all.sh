
#finetune_R_50_C4_1x
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=3 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x.yaml 2>&1 | tee log/finetune_R_50_C4_1x.txt
# CUDA_VISIBLE_DEVICES=3 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/r50_lightcenternet_sm+rg/model_final.pth 2>&1 | tee log/finetune_R_50_C4_1x.txt



#cen + few shot
# #vovnet
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_vovnet.yaml 2>&1 | tee log/fsod_finetune_vovnet_cen_train_25shot.txt
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_vovnet.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/vovnet_25shot/model_final.pth 2>&1 | tee log/fsod_finetune_stone_vovnet_25_test_log.txt

rm support_dir/support_feature.pkl
CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
	--config-file configs/fsod/finetune_vovnet.yaml 2>&1 | tee log/fsod_finetune_vovnet_cen_train_25shot.txt
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_vovnet.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/vovnet_25shot/model_final.pth 2>&1 | tee log/fsod_finetune_stone_vovnet_25_test_log.txt

#dla
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=3 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_dla.yaml 2>&1 | tee log/fsod_finetune_dla_cen_train_log2.txt
# CUDA_VISIBLE_DEVICES=3 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_dla.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/dla/model_final.pth 2>&1 | tee log/fsod_finetune_stone_dla_test_log.txt





# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x.yaml 2>&1 | tee log/fsod_finetune_coco_R50_train_log.txt

# CUDA_VISIBLE_DEVICES=2 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/cen_fsod1/model_final.pth 2>&1 | tee log/fsod_finetune_stone_R50_test_log.txt
# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_coco_R50_test_log.txt
# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/R_50_C4_1x/model_0002399.pth 2>&1 | tee log/fsod_finetune_coco_R50_test_log.txt

#base
#train
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=1,2 python3 fsod_train_net.py --num-gpus 2  \
# 	--config-file configs/fsod/R_18_C4_1x.yaml 2>&1 | tee log/fsod_train_R_18_log.txt

# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=2 python3 fsod_train_net.py --num-gpus 1  \
# 	--config-file configs/fsod/R_34_C4_1x.yaml 2>&1 | tee log/fsod_train_R_34_change3_log.txt

# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=1,2 python3 fsod_train_net.py --num-gpus 2  \
# 	--config-file configs/fsod/R_50_C4_1x.yaml 2>&1 | tee log/fsod_train_R_50_log.txt

# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1  \
# 	--config-file configs/fsod/mobilenet_small_1x.yaml 2>&1 | tee log/fsod_train_mobile_small_log.txt

#val
# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/R_34_C4_1x.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_34_C4_1x/model_final.pth 2>&1 | tee log/fsod_stone_r34_base_test_log.txt

# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/mobilenet_small_1x.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/mobilenet_small_1x/model_final.pth 2>&1 | tee log/fsod_stone_mobilenet_small_base_test_log.txt
###############################################################################
#fine-tune
#coco
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_coco_R_34_C4_1x.yaml 2>&1 | tee log/fsod_finetune_train_coco_log.txt
#train
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_18_C4_1x.yaml 2>&1 | tee log/fsod_finetune_stone_R18_train_log.txt

# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_34_C4_1x.yaml 2>&1 | tee log/fsod_finetune_stone_R34_train_log.txt

# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x.yaml 2>&1 | tee log/fsod_finetune_stone_R50_train_log.txt

# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_mobilenet_small_1x.yaml 2>&1 | tee log/fsod_finetune_stone_mobilenet_small_1x_train_log.txt
#val
# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_18_C4_1x.yaml \
#  	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/R_18_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_stone_R18_test_log.txt

# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_34_C4_1x.yaml \
#  	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/R_34_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_stone_R34_test_log.txt

# CUDA_VISIBLE_DEVICES=2 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_stone_R50_test_log.txt

#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
#	--config-file configs/fsod/finetune_R_50_C4_1x.yaml \
#	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log.txt

# CUDA_VISIBLE_DEVICES=1 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_mobilenet_small_1x.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/mobilenet_small_1x/model_final.pth 2>&1 | tee log/fsod_finetune_mobilenet_small_test_log.txt