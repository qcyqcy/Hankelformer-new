export CUDA_VISIBLE_DEVICES=0

model_name=Hankelformer

# 更新 run.py 和 dataset 的路径

run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/PEMS"




# window_size:1, contrastive_weight:0.05, learning_rate:0.0005, pred_len:96  
# mse:0.10796036571264267, mae:0.22124363481998444
python -u $run_path \
    --task_name long_term_forecast_contra\
    --is_training 1 \
    --root_path $root_path/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 4 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 1024 \
    --itr 1 \
    --batch_size 32 \
    --use_norm 0 \
    --window_size 1 \
    --contrastive_weight 0.05 \
    --learning_rate 0.0005








