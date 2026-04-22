export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

run_path="../../run.py"
root_path="../../dataset/Bosphorus_Wind"

down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=3
batch_size=16

# enc_in/c_out = 94 (站点数量)

# 12
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path\
  --data_path  bosphorus_weather.csv\
  --model_id Bosphorus_Wind_96_12 \
  --model $model_name \
  --data Bosphorus_Wind \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 94 \
  --c_out 94 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 64 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


# 24
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path\
  --data_path bosphorus_weather.csv \
  --model_id Bosphorus_Wind_96_24 \
  --model $model_name \
  --data Bosphorus_Wind \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 94 \
  --c_out 94 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 64 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


# 48
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path\
  --data_path bosphorus_weather.csv \
  --model_id Bosphorus_Wind_96_48 \
  --model $model_name \
  --data Bosphorus_Wind \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 94 \
  --c_out 94 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 64 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


# 96
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path\
  --data_path bosphorus_weather.csv \
  --model_id Bosphorus_Wind_96_96 \
  --model $model_name \
  --data Bosphorus_Wind \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 94 \
  --c_out 94 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 64 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
