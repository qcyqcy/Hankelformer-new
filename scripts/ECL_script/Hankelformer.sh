
export CUDA_VISIBLE_DEVICES=0

model_name=Hankelformer

# 更新 run.py 和 dataset 的路径
run_path="../../run.py"
root_path="../../dataset/ECL/"

# # 96-96 bingo
# # window_size:18, contrastive_weight:0.03, learning_rate:0.0005, pred_len:96  
# # mse:0.14092953503131866, mae:0.2373410314321518

python -u $run_path \
  --is_training 1 \
  --task_name long_term_forecast_contra\
  --root_path $root_path/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --batch_size 16 \
  --window_size 18 \
  --contrastive_weight 0.03 \
  --learning_rate 0.0005 \
  --train_epochs 20

# # # 192 bingo
# # # window_size:24, contrastive_weight:0.01, learning_rate:0.0005, pred_len:192  
# # # mse:0.15737943351268768, mae:0.25243180990219116
# python -u $run_path \
#   --is_training 1 \
#   --root_path $root_path/ \
#   --task_name long_term_forecast_contra\
#   --data_path electricity.csv \
#   --model_id ECL_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 192 \
#   --e_layers 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1 \
#   --batch_size 16 \
#   --window_size 24 \
#   --contrastive_weight 0.01 \
#   --learning_rate 0.0005 \
#   --train_epochs 20








# # # 336
# # # window_size:32, contrastive_weight:0.1, learning_rate:0.0005, pred_len:336  
# # # mse:0.16860854625701904, mae:0.26777157187461853

# python -u $run_path \
#   --is_training 1 \
#   --task_name long_term_forecast_contra\
#   --root_path $root_path/ \
#   --data_path electricity.csv \
#   --model_id ECL_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 336 \
#   --e_layers 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1 \
#   --batch_size 16 \
#   --window_size 32 \
#   --contrastive_weight 0.1 \
#   --learning_rate 0.0005 \
#   --train_epochs 20



# # # 720

# # # window_size:32, contrastive_weight:0.1, learning_rate:0.0005, pred_len:720  
# # # mse:0.19916197657585144, mae:0.2955571413040161
# python -u $run_path \
#   --is_training 1 \
#   --root_path $root_path/ \
#   --task_name long_term_forecast_contra\
#   --data_path electricity.csv \
#   --model_id ECL_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 720 \
#   --e_layers 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1 \
#   --batch_size 16 \
#   --window_size 32 \
#   --contrastive_weight 0.1 \
#   --learning_rate 0.0005 \
#   --train_epochs 20







# # 超参数范围
# window_sizes=(20 22 24)
# # contrastive_weights=(0.01  0.03  0.05  0.07  0.1)
# contrastive_weights=(0.005)
# learning_rates=(0.0005)



# # 循环遍历所有超参数组合
# for window_size in "${window_sizes[@]}"; do
#   for contrastive_weight in "${contrastive_weights[@]}"; do
#     for learning_rate in "${learning_rates[@]}"; do
#       python -u $run_path \
#         --task_name long_term_forecast_contra\
#         --is_training 1 \
#         --root_path $root_path/ \
#         --data_path electricity.csv \
#         --model_id ECL_96_192 \
#         --model $model_name \
#         --data custom \
#         --features M \
#         --seq_len 96 \
#         --pred_len 192 \
#         --e_layers 3 \
#         --enc_in 321 \
#         --dec_in 321 \
#         --c_out 321 \
#         --des 'Exp' \
#         --d_model 512 \
#         --d_ff 512 \
#         --itr 1 \
#         --batch_size 16 \
#         --train_epochs 20 \
#         --window_size $window_size \
#         --contrastive_weight $contrastive_weight \
#         --learning_rate $learning_rate
#     done
#   done
# done


# # # 循环遍历所有超参数组合
# for window_size in "${window_sizes[@]}"; do
#   for contrastive_weight in "${contrastive_weights[@]}"; do
#     for learning_rate in "${learning_rates[@]}"; do
#       python -u $run_path \
#         --task_name long_term_forecast_contra\
#         --is_training 1 \
#         --root_path $root_path/ \
#         --data_path electricity.csv \
#         --model_id ECL_96_336 \
#         --model $model_name \
#         --data custom \
#         --features M \
#         --seq_len 96 \
#         --pred_len 336 \
#         --e_layers 3 \
#         --enc_in 321 \
#         --dec_in 321 \
#         --c_out 321 \
#         --des 'Exp' \
#         --d_model 512 \
#         --d_ff 512 \
#         --itr 1 \
#         --batch_size 16 \
#         --window_size $window_size \
#         --contrastive_weight $contrastive_weight \
#         --learning_rate $learning_rate
#     done
#   done
# done



# # 循环遍历所有超参数组合
# for window_size in "${window_sizes[@]}"; do
#   for contrastive_weight in "${contrastive_weights[@]}"; do
#     for learning_rate in "${learning_rates[@]}"; do
#       python -u $run_path \
#         --task_name long_term_forecast_contra\
#         --is_training 1 \
#         --root_path $root_path/ \
#         --data_path electricity.csv \
#         --model_id ECL_96_720 \
#         --model $model_name \
#         --data custom \
#         --features M \
#         --seq_len 96 \
#         --pred_len 720 \
#         --e_layers 3 \
#         --enc_in 321 \
#         --dec_in 321 \
#         --c_out 321 \
#         --des 'Exp' \
#         --d_model 512 \
#         --d_ff 512 \
#         --itr 1 \
#         --batch_size 16 \
#         --window_size $window_size \
#         --contrastive_weight $contrastive_weight \
#         --learning_rate $learning_rate
#     done
#   done
# done