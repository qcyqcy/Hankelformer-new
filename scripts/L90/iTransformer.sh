
export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

# 更新 run.py 和 dataset 的路径
run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/L90"

# # 无噪声
# python -u $run_path \
#   --task_name long_term_forecast\
#   --is_training 1 \
#   --root_path $root_path/ \
#   --data_path coupled_lorenz_clean.csv \
#   --model_id coupled_lorenz_clean_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 90 \
#   --dec_in 90 \
#   --c_out 90\
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 64\
#   --itr 1 \
#   --train_epochs 20



# 0.1噪声
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.1.csv \
  --model_id coupled_lorenz_noise_0.1_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20


# 0.2噪声
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.2.csv \
  --model_id coupled_lorenz_noise_0.2_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20


# 0.3噪声
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.3.csv \
  --model_id coupled_lorenz_noise_0.3_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20


# 0.4噪声
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.4.csv \
  --model_id coupled_lorenz_noise_0.4_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20


# 0.5噪声
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.5.csv \
  --model_id coupled_lorenz_noise_0.5_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20

# 0.6噪声
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.6.csv \
  --model_id coupled_lorenz_noise_0.6_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20

# 0.7噪声
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.7.csv \
  --model_id coupled_lorenz_noise_0.7_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20


# 0.8噪声
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.8.csv \
  --model_id coupled_lorenz_noise_0.8_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20

# 0.9噪声
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.9.csv \
  --model_id coupled_lorenz_noise_0.9_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20

# 1.0噪声
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_1.0.csv \
  --model_id coupled_lorenz_noise_1.0_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20