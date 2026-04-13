export CUDA_VISIBLE_DEVICES=0

model_name=Hankelformer

# 更新 run.py 和 dataset 的路径

run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/PEMS"


python -u $run_path \
    --task_name long_term_forecast_contra\
    --is_training 1 \
    --root_path $root_path/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --batch_size 30 \
    --use_norm 0 \
    --window_size 1 \
    --contrastive_weight 0.03 \
    --learning_rate 0.001








