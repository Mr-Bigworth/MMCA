# Visual Grounding with Multi-modal Conditional Adaptation (MMCA)
[Visual Grounding with Multi-modal Conditional Adaptation](https://openreview.net/forum?id=wYiRt6gIgc&referrer=%5Bthe%20profile%20of%20Ruilin%20Yao%5D(%2Fprofile%3Fid%3D~Ruilin_Yao1)), ACMMM (Oral), 2024.

by Ruilin Yao, Shengwu Xiong, Yichen Zhao, Yi Rong*

**Update on 2024/9/7: We have submitted a basic version of MMCA based on Transvg, welcome to use and provide feedback!**

**Update on 2024/7/31: This paper has been accepted by the ACM Multimedia 2024 (Oral). Our code will be released soon!**

### Getting Started

Please refer to [GETTING_STARGTED.md](docs/GETTING_STARTED.md) to learn how to prepare the datasets and pretrained checkpoints.

### Training and Evaluation

1.  Training
    ```
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --output_dir outputs/referit_r50 --epochs 90 --lr_drop 60
    ```

    We recommend to set --max_query_len 40 for RefCOCOg, and --max_query_len 20 for other datasets. 
    
    We recommend to set --epochs 180 (--lr_drop 120 acoordingly) for RefCOCO+, and --epochs 90 (--lr_drop 60 acoordingly) for other datasets. 

2.  Evaluation
    ```
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset referit --max_query_len 20 --eval_set test --eval_model ./outputs/referit_r50/best_checkpoint.pth --output_dir ./outputs/referit_r50
    ```

### Acknowledge
This codebase is partially based on [ReSC](https://github.com/zyang-ur/ReSC), [DETR](https://github.com/facebookresearch/detr) and [TransVG](https://github.com/djiajunustc/TransVG/tree/main).
