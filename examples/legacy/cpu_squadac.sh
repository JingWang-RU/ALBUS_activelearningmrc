#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #zhiheng-huang/bert-base-uncased-embedding-relative-key-query \
DATA_DIR=/al_transformers/dataset/squad
OUTPUT_DIR=/al_transformers/examples/legacy/output/
TRAIN_FILE=train-v1.1.json
PREDICT_FILE=dev-v1.1.json
python3 /home/.../al_transformers/examples/legacy/run_squad_L2.py \
    --model_name_or_path bert-base-uncased \
    --data_dir $DATA_DIR \
    --train_file $DATA_DIR/$TRAIN_FILE \
    --predict_file $DATA_DIR/$PREDICT_FILE \
    --output_dir $OUTPUT_DIR \
    --model_type bert \
    --do_train \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --per_gpu_eval_batch_size=60 \
    --per_gpu_train_batch_size=6 \
    --thresh 0.1 \
    --do_eval \
    --flag_sub 0 \
    --init 1000 \
    --query 2000 \
    --query_learnrate 1.1 \
    --al_method beyond \
    --version 1 \
    --do_lower_case \
    --al_incremental \
    --overwrite_output_dir \
    --lambda_parameter 1.0 \
    --do_active \
    --overwrite_output_dir \
    --doc_stride 128 \
    --n_gpu 4 \
    --save_steps 3000 


