results_DIR=./al_transformers/output

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_squad_L2.py \
  --thresh 0.1 \
  --num_round 2 \
  --flag_sub 0 \
  --init 1000 \
  --query 6000 \
  --query_learnrate 1.1 \
  --al_method beyond \
  --version 7 \
  --do_lower_case \
  --al_incremental \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_active \
  --overwrite_output_dir \
  --lambda_parameter 1\
  --per_gpu_train_batch_size 6 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --dataset squad \
  --output_dir $results_DIR
