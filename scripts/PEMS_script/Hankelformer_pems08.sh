export CUDA_VISIBLE_DEVICES=0

model_name=Hankelformer

# 更新 run.py 和 dataset 的路径

run_path="../../run.py"
root_path="../../dataset/PEMS"






python -u $run_path \
    --task_name long_term_forecast_contra\
    --is_training 1 \
    --root_path $root_path/ \
    --data_path PEMS08.npz \
    --model_id PEMS08_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 4 \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --batch_size 64 \
    --use_norm 0 \
    --window_size 4 \
    --contrastive_weight 0.05 \
    --learning_rate 0.001 \
    --train_epochs 20\








