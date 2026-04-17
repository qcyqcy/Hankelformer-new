
export CUDA_VISIBLE_DEVICES=0

model_name=Hankelformer

# 更新 run.py 和 dataset 的路径
run_path="../../run.py"
root_path="../../dataset/L90"


# # 无噪声

# python -u $run_path \
#   --task_name long_term_forecast_contra\
#   --is_training 1 \
#   --root_path $root_path/ \
#   --data_path coupled_lorenz_clean.csv \
#   --model_id coupled_lorenz_clean_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 4 \
#   --enc_in 90 \
#   --dec_in 90 \
#   --c_out 90\
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 128\
#   --itr 1 \
#   --window_size 3 \
#   --contrastive_weight 0.375 \
#   --learning_rate 0.0014 \
#   --train_epochs 20




# 0.1噪声

python -u $run_path \
  --task_name long_term_forecast_contra\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.1.csv \
  --model_id coupled_lorenz_noise_0.1_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 128\
  --itr 1 \
  --window_size 3 \
  --contrastive_weight 0.375 \
  --learning_rate 0.0014 \
  --train_epochs 20

# 0.2噪声

python -u $run_path \
  --task_name long_term_forecast_contra\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.2.csv \
  --model_id coupled_lorenz_noise_0.2_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 128\
  --itr 1 \
  --window_size 3 \
  --contrastive_weight 0.375 \
  --learning_rate 0.0014 \
  --train_epochs 20


  # 0.3噪声

python -u $run_path \
  --task_name long_term_forecast_contra\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.3.csv \
  --model_id coupled_lorenz_noise_0.3_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 128\
  --itr 1 \
  --window_size 3 \
  --contrastive_weight 0.375 \
  --learning_rate 0.0014 \
  --train_epochs 20



  # 0.4噪声

python -u $run_path \
  --task_name long_term_forecast_contra\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.4.csv \
  --model_id coupled_lorenz_noise_0.4_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 128\
  --itr 1 \
  --window_size 3 \
  --contrastive_weight 0.375 \
  --learning_rate 0.0014 \
  --train_epochs 20


  # 0.5噪声

python -u $run_path \
  --task_name long_term_forecast_contra\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.5.csv \
  --model_id coupled_lorenz_noise_0.5_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 128\
  --itr 1 \
  --window_size 3 \
  --contrastive_weight 0.375 \
  --learning_rate 0.0014 \
  --train_epochs 20


  # 0.6噪声

python -u $run_path \
  --task_name long_term_forecast_contra\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.6.csv \
  --model_id coupled_lorenz_noise_0.6_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 128\
  --itr 1 \
  --window_size 3 \
  --contrastive_weight 0.375 \
  --learning_rate 0.0014 \
  --train_epochs 20



  # 0.7噪声

python -u $run_path \
  --task_name long_term_forecast_contra\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.7.csv \
  --model_id coupled_lorenz_noise_0.7_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 128\
  --itr 1 \
  --window_size 3 \
  --contrastive_weight 0.375 \
  --learning_rate 0.0014 \
  --train_epochs 20



  # 0.8噪声

python -u $run_path \
  --task_name long_term_forecast_contra\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.8.csv \
  --model_id coupled_lorenz_noise_0.8_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 128\
  --itr 1 \
  --window_size 3 \
  --contrastive_weight 0.375 \
  --learning_rate 0.0014 \
  --train_epochs 20


  # 0.9噪声

python -u $run_path \
  --task_name long_term_forecast_contra\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_0.9.csv \
  --model_id coupled_lorenz_noise_0.9_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 128\
  --itr 1 \
  --window_size 3 \
  --contrastive_weight 0.375 \
  --learning_rate 0.0014 \
  --train_epochs 20


  # 1.0声

python -u $run_path \
  --task_name long_term_forecast_contra\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path coupled_lorenz_noise_1.0.csv \
  --model_id coupled_lorenz_noise_1.0_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 128\
  --itr 1 \
  --window_size 3 \
  --contrastive_weight 0.375 \
  --learning_rate 0.0014 \
  --train_epochs 20































# # 粗略搜索

# # # 建议的搜索范围
# # window_sizes=(1 2 4 8 16)  
# # contrastive_weights=(0.1 0.2 0.3 0.5 0.8)  
# # learning_rates=(0.0001 0.0002 0.0005 0.001 0.002)

# # 精搜
# # window_sizes=(1 2 4 6 8 10 12 14 16)  
# # contrastive_weights=(0.25 0.28 0.3 0.32 0.35 0.38 0.4)  
# # learning_rates=(0.0008 0.0009 0.00095 0.001 0.00105 0.0011 0.0012)

# window_sizes=(3)
# contrastive_weights=(0.375)  # 0.375附近极小范围
# learning_rates=(0.0014)  # 锁定最优lr


# for window_size in "${window_sizes[@]}"; do
#   for contrastive_weight in "${contrastive_weights[@]}"; do
#     for learning_rate in "${learning_rates[@]}"; do
#       python -u $run_path \
#         --task_name long_term_forecast_contra\
#         --is_training 1 \
#         --root_path $root_path/ \
#         --data_path coupled_lorenz_clean.csv \
#         --model_id coupled_lorenz_clean_96_96 \
#         --model $model_name \
#         --data custom \
#         --features M \
#         --seq_len 96 \
#         --pred_len 96 \
#         --e_layers 4 \
#         --enc_in 90 \
#         --dec_in 90 \
#         --c_out 90\
#         --des 'Exp' \
#         --d_model 512 \
#         --d_ff 512 \
#         --batch_size 128\
#         --itr 1 \
#         --window_size $window_size \
#         --contrastive_weight $contrastive_weight \
#         --learning_rate $learning_rate \
#         --train_epochs 20
#     done
#   done
# done
