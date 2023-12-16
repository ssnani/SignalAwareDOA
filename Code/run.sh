#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
srun python SignalAwareDOA_train.py --num_nodes=$1 \
                  --num_gpu_per_node=$2 \
                  --train \
                  --dataset_file=../dataset_file_circular_motion_snr_-5_t60_0.2_noisy_reverb.txt \
                  --val_dataset_file=../val_dataset_file_circular_motion_snr_-5_t60_0.2_noisy_reverb.txt \
                  --diffuse_files_path=/fs/scratch/PAS0774/Shanmukh/Databases/Timit/train_spk_signals \
                  --batch_size=16 \
                  --max_n_epochs=100 \
                  --num_workers=5 \
                  --ckpt_dir=/fs/scratch/PAA0005/Shanmukh/Habets_SignalAware_Doa/Experiments_bce_lr_1e-4/ \
                  --exp_name=EndtoEnd_Training_DOA_CCE \
                  --resume_model=last.ckpt \
                  --array_job \
                  --input_train_filename=$3 \
                  #--fast_dev_run
                  
                  
                  
                  

