export CUDA_VISIBLE_DEVICES=0

model_name=Hankelformer

run_path="../../run.py"
root_path="../../dataset/Bosphorus_Wind"

# enc_in/dec_in/c_out = 94 (站点数量)

# 12
python -u $run_path \
    --task_name long_term_forecast_contra\
    --is_training 1 \
    --root_path $root_path/ \
    --data_path bosphorus_weather.csv \
    --model_id Bosphorus_Wind_96_12 \
    --model $model_name \
    --data Bosphorus_Wind \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 2 \
    --enc_in 94 \
    --dec_in 94 \
    --c_out 94 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 64 \
    --itr 1 \
    --train_epochs 20 \
    --window_size 1 \
    --contrastive_weight 0.00005 \
    --learning_rate 0.0001


# 24
python -u $run_path \
    --task_name long_term_forecast_contra\
    --is_training 1 \
    --root_path $root_path/ \
    --data_path bosphorus_weather.csv \
    --model_id Bosphorus_Wind_96_24 \
    --model $model_name \
    --data Bosphorus_Wind \
    --features M \
    --seq_len 96 \
    --pred_len 24 \
    --e_layers 2 \
    --enc_in 94 \
    --dec_in 94 \
    --c_out 94 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 64 \
    --itr 1 \
    --train_epochs 20 \
    --window_size 1 \
    --contrastive_weight 0.00005 \
    --learning_rate 0.0001


# 48
python -u $run_path \
    --task_name long_term_forecast_contra\
    --is_training 1 \
    --root_path $root_path/ \
    --data_path bosphorus_weather.csv \
    --model_id Bosphorus_Wind_96_48 \
    --model $model_name \
    --data Bosphorus_Wind \
    --features M \
    --seq_len 96 \
    --pred_len 48 \
    --e_layers 2 \
    --enc_in 94 \
    --dec_in 94 \
    --c_out 94 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 64 \
    --itr 1 \
    --train_epochs 20 \
    --window_size 1 \
    --contrastive_weight 0.00005 \
    --learning_rate 0.0001


# 96
python -u $run_path \
    --task_name long_term_forecast_contra\
    --is_training 1 \
    --root_path $root_path/ \
    --data_path bosphorus_weather.csv \
    --model_id Bosphorus_Wind_96_96 \
    --model $model_name \
    --data Bosphorus_Wind \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 94 \
    --dec_in 94 \
    --c_out 94 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 64 \
    --itr 1 \
    --train_epochs 20 \
    --window_size 1 \
    --contrastive_weight 0.00005 \
    --learning_rate 0.0001
