export CUDA_VISIBLE_DEVICES=0

model_name=Hankelformer

# 更新 run.py 和 dataset 的路径
run_path="../../run.py"
root_path="../../dataset/PEMS"





# # window_size:1, contrastive_weight:0.03, learning_rate:0.001, pred_len:96  
# # mse:0.13822849094867706, mae:0.2526562213897705
# python -u $run_path \
#     --task_name long_term_forecast_contra\
#     --is_training 1 \
#     --root_path $root_path/ \
#     --data_path PEMS03.npz \
#     --model_id PEMS03_96_96 \
#     --model $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --e_layers 4 \
#     --enc_in 358 \
#     --dec_in 358 \
#     --c_out 358 \
#     --des 'Exp' \
#     --d_model 512 \
#     --d_ff 512 \
#     --itr 1 \
#     --batch_size 64 \
#     --use_norm 0 \
#     --window_size 1 \
#     --contrastive_weight 0.03 \
#     --learning_rate 0.001





# # window_size:1, contrastive_weight:0.03, learning_rate:0.001, pred_len:96  
# # mse:0.13822849094867706, mae:0.2526562213897705
# python -u $run_path \
#     --task_name long_term_forecast_contra\
#     --is_training 1 \
#     --root_path $root_path/ \
#     --data_path PEMS03.npz \
#     --model_id PEMS03_96_96 \
#     --model $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --e_layers 4 \
#     --enc_in 358 \
#     --dec_in 358 \
#     --c_out 358 \
#     --des 'Exp' \
#     --d_model 512 \
#     --d_ff 512 \
#     --itr 1 \
#     --batch_size 64 \
#     --use_norm 0 \
#     --window_size 1 \
#     --contrastive_weight 0.03 \
#     --learning_rate 0.001





# window_size:1, contrastive_weight:0.03, learning_rate:0.001, pred_len:96  
# mse:0.13822849094867706, mae:0.2526562213897705
python -u $run_path \
    --task_name long_term_forecast_contra\
    --is_training 1 \
    --root_path $root_path/ \
    --data_path PEMS03.npz \
    --model_id PEMS03_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 4 \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --batch_size 64 \
    --use_norm 0 \
    --window_size 1 \
    --contrastive_weight 0.03 \
    --learning_rate 0.001

















# window_size:1, contrastive_weight:0.03, learning_rate:0.001, pred_len:96  
# mse:0.13822849094867706, mae:0.2526562213897705
# python -u $run_path \
#     --task_name long_term_forecast_contra\
#     --is_training 1 \
#     --root_path $root_path/ \
#     --data_path PEMS03.npz \
#     --model_id PEMS03_96_96 \
#     --model $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --e_layers 4 \
#     --enc_in 358 \
#     --dec_in 358 \
#     --c_out 358 \
#     --des 'Exp' \
#     --d_model 512 \
#     --d_ff 512 \
#     --itr 1 \
#     --batch_size 64 \
#     --use_norm 0 \
#     --window_size 1 \
#     --contrastive_weight 0.03 \
#     --learning_rate 0.001