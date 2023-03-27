# Faster-OreFSDet

Offical codes for "Faster OreFSDet: A Lightweight and Effective Few-shot Object Detector for Ore Images"

# Faster-OreFSDet

**Faster-OreFSDet** is based on [**FewX**](https://github.com/fanq15/FewX) ( an open source toolbox on top of Detectron2 for data-limited instance-level recognition tasks, e.g.) 

## Faster-OreFsdet and baseline on ore dataset
<table >
    <tr align="center">
        <th rowspan="2">Method</th>
        <th colspan="2">5-shot</th>
        <th colspan="2">15-shot</th>
        <th colspan="2">25-shot</th>
	     <th colspan="2">25-shot</th>
    </tr>
    <tr align="center">
        <td>AP</td>
        <td>AP75</td>
        <td>AP</td>
        <td>AP75</td>
        <td>AP</td>
        <td>AP75</td>
	     <td>AP75</td>
    </tr>
    <tr align="center">
        <td>Attentionrpn(baseline)</td>
        <td>25.1</td>
        <td>27.0</td>
        <td>29.2</td>
        <td>34.5</td>
        <td>30.8</td>
        <td>37.0</td>
	     <td>37.0</td>
    </tr>
    <tr align="center">
        <td>Faster-OreFSDet</td>
        <td>48.5</td>
        <td>57.6</td>
        <td>52.1</td>
        <td>62.5</td>
        <th>54.1</th>
        <th>64.7</th>
	    <th>64.7</th>
    </tr>
	
</table>


The model can be obtained from here <a href="https://drive.google.com/file/d/1iv5aXgT7ExHfuZi6lIzdMb-pCB-IduSK/view?usp=share_link"> model</a>&nbsp;\.



## Step 1: Installation
You only need to install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). We recommend the Pre-Built Detectron2 (Linux only) version with pytorch 1.7. I use the Pre-Built Detectron2 with CUDA 10.1 and pytorch 1.7 and you can run this code to install it.

```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
```

## Step 2: Prepare dataset
- Prepare for ore dataset, you can get from [here](https://drive.google.com/file/d/1eYkPHgDWULHind802P4tvy9l7lIQrpqk/view?usp=share_link). The ore dataset has been handled under few-shot setting, you only need to add it to dataset.


## Step 3: Training and Evaluation

Run `sh all.sh` in the root dir. 

### Training
change all.sh
```
rm support_dir/support_feature.pkl
CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
	--config-file configs/fsod/finetune_R_50_C4_1x.yaml 2>&1 | tee log/fsod_finetune_stone_R50_train_log_5shot.txt
```
Then, run the following
```
sh all.sh
```
### Evaluation
change the all.sh
```
CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
	--config-file configs/fsod/finetune_R_50_C4_1x.yaml \
	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/R_50_C4_1x_stone_5shot/model_final.pth 2>&1 | tee log/fsod_finetune_stone_R50_test_log_5shot.txt
```
just run the following
```
sh all.sh
```
## Visualize the results
```
python demo.py  \
    --config-file configs/fsod/finetune_R_50_C4_1x.yaml \
    --input directory/*.png \
    --output results \
    --opts MODEL.WEIGHTS ./output/fsod/finetune_dir/R_50_C4_1x/model_final.pth
```

This repo is developed based on [FewX](https://github.com/fanq15/FewX) and [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). Thanks for their wonderful codebases.



