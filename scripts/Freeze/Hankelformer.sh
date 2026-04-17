
export CUDA_VISIBLE_DEVICES=1

model_name=Hankelformer

# 更新 run.py 和 dataset 的路径
run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/Texas_Freeze"



# # 12
# # window_size:15, contrastive_weight:0.005, learning_rate:0.0005, pred_len:12  
# # mse:0.10465674102306366, mae:0.21241395175457
# python -u $run_path \
#   --task_name long_term_forecast_contra\
#   --is_training 1 \
#   --root_path $root_path/ \
#   --data_path Texas_Freeze.csv \
#   --model_id Texas_96_12 \
#   --model $model_name \
#   --data Texas_Freeze \
#   --features M \
#   --seq_len 96 \
#   --pred_len 12 \
#   --e_layers 2 \
#   --enc_in 121 \
#   --dec_in 121 \
#   --c_out 121\
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 64\
#   --itr 1 \
#   --window_size 15 \
#   --contrastive_weight 0.005 \
#   --learning_rate 0.0005


 

# # # 24
# # # window_size:5, contrastive_weight:0.005, learning_rate:0.0005, pred_len:24 

# python -u $run_path \
#   --task_name long_term_forecast_contra\
#   --is_training 1 \
#   --root_path $root_path/ \
#   --data_path Texas_Freeze.csv \
#   --model_id Texas_96_24 \
#   --model $model_name \
#   --data Texas_Freeze \
#   --features M \
#   --seq_len 96 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --enc_in 121 \
#   --dec_in 121 \
#   --c_out 121\
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 64\
#   --itr 1 \
#   --window_size 5 \
#   --contrastive_weight 0.005 \
#   --learning_rate 0.0005





# 48 重新调
# window_size:10, contrastive_weight:0.01, learning_rate:0.0003, pred_len:48  
# mse:0.3465724587440491, mae:0.43119242787361145
python -u $run_path \
  --task_name long_term_forecast_contra\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path Texas_Freeze.csv \
  --model_id Texas_96_48 \
  --model $model_name \
  --data Texas_Freeze \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 121 \
  --dec_in 121 \
  --c_out 121\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --window_size 10 \
  --contrastive_weight 0.01 \
  --learning_rate 0.0003



# # 96 重新调
# # 
# # window_size:20, contrastive_weight:0.001, learning_rate:0.0003, pred_len:96  
# # mse:0.6282870769500732, mae:0.6193678379058838
# python -u $run_path \
#   --task_name long_term_forecast_contra\
#   --is_training 1 \
#   --root_path $root_path/ \
#   --data_path Texas_Freeze.csv \
#   --model_id Texas_96_96 \
#   --model $model_name \
#   --data Texas_Freeze \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 121 \
#   --dec_in 121 \
#   --c_out 121\
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 64\
#   --itr 1 \
#   --window_size 20 \
#   --contrastive_weight 0.001 \
#   --learning_rate 0.0003




# # # # # 超参数范围
# window_sizes=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24) 
# contrastive_weights=(0.001 0.005 0.01)
# learning_rates=(0.00005 0.0001 0.0003 0.0005)



# # # # 循环遍历所有超参数组合
# for window_size in "${window_sizes[@]}"; do
#   for contrastive_weight in "${contrastive_weights[@]}"; do
#     for learning_rate in "${learning_rates[@]}"; do
#       python -u $run_path \
#         --task_name long_term_forecast_contra\
#         --is_training 1 \
#         --root_path $root_path/ \
#         --data_path Texas_Freeze.csv \
#         --model_id Texas_96_96 \
#         --model $model_name \
#         --data Texas_Freeze \
#         --features M \
#         --seq_len 96 \
#         --pred_len 96 \
#         --e_layers 2 \
#         --enc_in 121 \
#         --dec_in 121 \
#         --c_out 121\
#         --des 'Exp' \
#         --d_model 512 \
#         --d_ff 512 \
#         --batch_size 64\
#         --itr 1 \
#         --window_size $window_size \
#         --contrastive_weight $contrastive_weight \
#         --learning_rate $learning_rate
#     done
#   done
# done
