
export CUDA_VISIBLE_DEVICES=0


model_name=TimeMixer


# 更新 run.py 和 dataset 的路径
run_path="../../run.py"
root_path="../../dataset/L90"



down_sampling_layers=2
down_sampling_window=2
learning_rate=0.01
d_model=8
d_ff=4
train_epochs=20
patience=3

# # 无噪声
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path  $root_path/\
#   --data_path coupled_lorenz_clean.csv \
#   --model_id coupled_lorenz_clean_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 0 \
#   --pred_len 96 \
#   --e_layers 1 \
#   --enc_in 90 \
#   --c_out 90 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 64 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window




# 0.1噪声
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path coupled_lorenz_noise_0.1.csv \
  --model_id coupled_lorenz_noise_0.1_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 90 \
  --c_out 90 \
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


# 0.2噪声
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path coupled_lorenz_noise_0.2.csv \
  --model_id coupled_lorenz_noise_0.2_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 90 \
  --c_out 90 \
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



# 0.3噪声
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path coupled_lorenz_noise_0.3.csv \
  --model_id coupled_lorenz_noise_0.3_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 90 \
  --c_out 90 \
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


# 0.4噪声
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path coupled_lorenz_noise_0.4.csv \
  --model_id coupled_lorenz_noise_0.4_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 90 \
  --c_out 90 \
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


# 0.5噪声
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path coupled_lorenz_noise_0.5.csv \
  --model_id coupled_lorenz_noise_0.5_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 90 \
  --c_out 90 \
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



# 0.6噪声
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path coupled_lorenz_noise_0.6.csv \
  --model_id coupled_lorenz_noise_0.6_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 90 \
  --c_out 90 \
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


# 0.7噪声
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path coupled_lorenz_noise_0.7.csv \
  --model_id coupled_lorenz_noise_0.7_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 90 \
  --c_out 90 \
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



# 0.8噪声
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path coupled_lorenz_noise_0.8.csv \
  --model_id coupled_lorenz_noise_0.8_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 90 \
  --c_out 90 \
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



# 0.9噪声
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path coupled_lorenz_noise_0.9.csv \
  --model_id coupled_lorenz_noise_0.9_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 90 \
  --c_out 90 \
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



# 1.0噪声
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path coupled_lorenz_noise_1.0.csv \
  --model_id coupled_lorenz_noise_1.0_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 90 \
  --c_out 90 \
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