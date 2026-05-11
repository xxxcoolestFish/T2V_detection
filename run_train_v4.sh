#!/bin/bash
cd /root/autodl-tmp
rm -f msc_checkpoints/latest.pth
/root/miniconda3/bin/python -u msc_train.py     --data_root /root/autodl-tmp     --epochs_s1 5 --epochs_s2 20     --batch_size 64     --no_resume     > msc_train_v4.log 2>&1
