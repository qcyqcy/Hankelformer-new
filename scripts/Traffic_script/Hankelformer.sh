
# export CUDA_VISIBLE_DEVICES=0

# model_name=Hankelformer

# # 更新 run.py 和 dataset 的路径
run_path="../../run.py"
root_path="../../dataset/traffic"


# # # 96-96 
# window_size:3, contrastive_weight:0.1, learning_rate:0.001, pred_len:96  
# mse:0.39555981755256653, mae:0.27423495054244995
# window_size:13, contrastive_weight:0.05, learning_rate:0.001, pred_len:96  
# mse:0.3931780159473419, mae:0.27368029952049255
# # python -u $run_path \
# #   --is_training 1 \
# #   --task_name long_term_forecast_contra\
# #   --root_path $root_path/ \
# #   --data_path traffic.csv \
# #   --model_id traffic_96_96 \
# #   --model $model_name \
# #   --data custom \
# #   --features M \
# #   --seq_len 96 \
# #   --pred_len 96 \
# #   --e_layers 4 \
# #   --enc_in 862 \
# #   --dec_in 862 \
# #   --c_out 862 \
# #   --des 'Exp' \
# #   --d_model 512 \
# #   --d_ff 512 \
# #   --itr 1 \
# #   --batch_size 16 \
# #   --window_size 6 \
# #   --contrastive_weight 0.01 \
# #   --learning_rate 0.001 \
# #   --train_epochs 30 \
# #   --patience 10 \




# # # # # 192 bingo
# # # 老版本  learning_rate 0.001  patience 10   train_epochs 30  window_size 11  contrastive_weight 0.01
# # python -u $run_path \
# #   --is_training 1 \
# #   --task_name long_term_forecast_contra\
# #   --root_path $root_path/ \
# #   --data_path traffic.csv \
# #   --model_id traffic_96_192 \
# #   --model $model_name \
# #   --data custom \
# #   --features M \
# #   --seq_len 96 \
# #   --pred_len 192 \
# #   --e_layers 4 \
# #   --enc_in 862 \
# #   --dec_in 862 \
# #   --c_out 862 \
# #   --des 'Exp' \
# #   --d_model 512 \
# #   --d_ff 512 \
# #   --itr 1 \
# #   --batch_size 16 \
# #   --window_size 11 \
# #   --contrastive_weight 0.01 \
# #   --learning_rate 0.001 \
# #   --train_epochs 30 \
# #   --patience 10 \







# # # # # 336
# # # 老版本  learning_rate 0.001  patience 10   train_epochs 30  window_size 8  contrastive_weight 0.01

# # python -u $run_path \
# #   --is_training 1 \
# #   --task_name long_term_forecast_contra\
# #   --root_path $root_path/ \
# #   --data_path traffic.csv \
# #   --model_id traffic_96_336 \
# #   --model $model_name \
# #   --data custom \
# #   --features M \
# #   --seq_len 96 \
# #   --pred_len 336 \
# #   --e_layers 4 \
# #   --enc_in 862 \
# #   --dec_in 862 \
# #   --c_out 862 \
# #   --des 'Exp' \
# #   --d_model 512 \
# #   --d_ff 512 \
# #   --itr 1 \
# #   --batch_size 16 \
# #   --window_size 8 \
# #   --contrastive_weight 0.01 \
# #   --learning_rate 0.001 \
# #   --train_epochs 30 \
# #   --patience 10 \



# # # # # 720
# # # 老版本  learning_rate 0.001  patience 10   train_epochs 30  window_size 11  contrastive_weight 0.01

# # python -u $run_path \
# #   --is_training 1 \
# #   --task_name long_term_forecast_contra\
# #   --root_path $root_path/ \
# #   --data_path traffic.csv \
# #   --model_id traffic_96_720 \
# #   --model $model_name \
# #   --data custom \
# #   --features M \
# #   --seq_len 96 \
# #   --pred_len 720 \
# #   --e_layers 4 \
# #   --enc_in 862 \
# #   --dec_in 862 \
# #   --c_out 862 \
# #   --des 'Exp' \
# #   --d_model 512 \
# #   --d_ff 512 \
# #   --itr 1 \
# #   --batch_size 16 \
# #   --window_size 11 \
# #   --contrastive_weight 0.01 \
# #   --learning_rate 0.001 \
# #   --train_epochs 30 \
# #   --patience 10 \





# # # # # # 超参数范围
# window_sizes=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24) 
# contrastive_weights=( 0.03 0.05 0.07 )
# learning_rates=(0.001)



# # # # 循环遍历所有超参数组合
# for window_size in "${window_sizes[@]}"; do
#   for contrastive_weight in "${contrastive_weights[@]}"; do
#     for learning_rate in "${learning_rates[@]}"; do
#       python -u $run_path \
#         --task_name long_term_forecast_contra\
#         --is_training 1 \
#         --root_path $root_path/ \
#         --data_path traffic.csv \
#         --model_id traffic_96_96 \
#         --model $model_name \
#         --data custom \
#         --features M \
#         --seq_len 96 \
#         --pred_len 96 \
#         --e_layers 4 \
#         --enc_in 862 \
#         --dec_in 862 \
#         --c_out 862\
#         --des 'Exp' \
#         --d_model 512 \
#         --d_ff 512 \
#         --batch_size 16\
#         --itr 1 \
#         --train_epochs 30 \
#         --patience 3 \
#         --window_size $window_size \
#         --contrastive_weight $contrastive_weight \
#         --learning_rate $learning_rate
#     done
#   done
# done



#!/bin/bash

model_name=Hankelformer
run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/traffic"

window_sizes=(12) 
contrastive_weights=(0.07)
learning_rates=(0.0008 0.001)

# 最大并行任务数（根据GPU数量）
MAX_PARALLEL=1
current_jobs=0
gpu_id=0

# # 192

# for window_size in "${window_sizes[@]}"; do
#   for contrastive_weight in "${contrastive_weights[@]}"; do
#     for learning_rate in "${learning_rates[@]}"; do
      
#       # 等待如果已达到最大并行数
#       while [ $current_jobs -ge $MAX_PARALLEL ]; do
#         wait -n  # 等待任意一个后台任务完成
#         current_jobs=$((current_jobs - 1))
#       done
      
#       # 启动新任务
#       (
#         export CUDA_VISIBLE_DEVICES=$gpu_id
#         echo "Starting: window_size=$window_size, contrastive_weight=$contrastive_weight, GPU=$gpu_id"
        
#         python -u $run_path \
#           --task_name long_term_forecast_contra\
#           --is_training 1 \
#           --root_path $root_path/ \
#           --data_path traffic.csv \
#           --model_id traffic_96_192 \
#           --model $model_name \
#           --data custom \
#           --features M \
#           --seq_len 96 \
#           --pred_len 192 \
#           --e_layers 4 \
#           --enc_in 862 \
#           --dec_in 862 \
#           --c_out 862\
#           --des 'Exp' \
#           --d_model 512 \
#           --d_ff 512 \
#           --batch_size 16\
#           --itr 1 \
#           --train_epochs 10 \
#           --patience 3 \
#           --window_size $window_size \
#           --contrastive_weight $contrastive_weight \
#           --learning_rate $learning_rate
        
#         echo "Finished: window_size=$window_size, contrastive_weight=$contrastive_weight, GPU=$gpu_id"
#       ) &
      
#       current_jobs=$((current_jobs + 1))
#       gpu_id=$((gpu_id + 1))
      
#       # 重置GPU ID（轮换使用）
#       if [ $gpu_id -eq 2 ]; then
#         gpu_id=0
#       fi
      
#     done
#   done
# done



# # 336


# for window_size in "${window_sizes[@]}"; do
#   for contrastive_weight in "${contrastive_weights[@]}"; do
#     for learning_rate in "${learning_rates[@]}"; do
      
#       # 等待如果已达到最大并行数
#       while [ $current_jobs -ge $MAX_PARALLEL ]; do
#         wait -n  # 等待任意一个后台任务完成
#         current_jobs=$((current_jobs - 1))
#       done
      
#       # 启动新任务
#       (
#         export CUDA_VISIBLE_DEVICES=$gpu_id
#         echo "Starting: window_size=$window_size, contrastive_weight=$contrastive_weight, GPU=$gpu_id"
        
#         python -u $run_path \
#           --task_name long_term_forecast_contra\
#           --is_training 1 \
#           --root_path $root_path/ \
#           --data_path traffic.csv \
#           --model_id traffic_96_336 \
#           --model $model_name \
#           --data custom \
#           --features M \
#           --seq_len 96 \
#           --pred_len 336 \
#           --e_layers 4 \
#           --enc_in 862 \
#           --dec_in 862 \
#           --c_out 862\
#           --des 'Exp' \
#           --d_model 512 \
#           --d_ff 512 \
#           --batch_size 16\
#           --itr 1 \
#           --train_epochs 10 \
#           --patience 3 \
#           --window_size $window_size \
#           --contrastive_weight $contrastive_weight \
#           --learning_rate $learning_rate
        
#         echo "Finished: window_size=$window_size, contrastive_weight=$contrastive_weight, GPU=$gpu_id"
#       ) &
      
#       current_jobs=$((current_jobs + 1))
#       gpu_id=$((gpu_id + 1))
      
#       # 重置GPU ID（轮换使用）
#       if [ $gpu_id -eq 2 ]; then
#         gpu_id=0
#       fi
      
#     done
#   done
# done


# 720

for window_size in "${window_sizes[@]}"; do
  for contrastive_weight in "${contrastive_weights[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      
      # 等待如果已达到最大并行数
      while [ $current_jobs -ge $MAX_PARALLEL ]; do
        wait -n  # 等待任意一个后台任务完成
        current_jobs=$((current_jobs - 1))
      done
      
      # 启动新任务
      (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        echo "Starting: window_size=$window_size, contrastive_weight=$contrastive_weight, GPU=$gpu_id"
        
        python -u $run_path \
          --task_name long_term_forecast_contra\
          --is_training 1 \
          --root_path $root_path/ \
          --data_path traffic.csv \
          --model_id traffic_96_720 \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 96 \
          --pred_len 720 \
          --e_layers 4 \
          --enc_in 862 \
          --dec_in 862 \
          --c_out 862\
          --des 'Exp' \
          --d_model 512 \
          --d_ff 512 \
          --batch_size 16\
          --itr 1 \
          --train_epochs 30 \
          --patience 10 \
          --window_size $window_size \
          --contrastive_weight $contrastive_weight \
          --learning_rate $learning_rate
        
        echo "Finished: window_size=$window_size, contrastive_weight=$contrastive_weight, GPU=$gpu_id"
      ) &
      
      current_jobs=$((current_jobs + 1))
      gpu_id=$((gpu_id + 1))
      
      # 重置GPU ID（轮换使用）
      if [ $gpu_id -eq 2 ]; then
        gpu_id=0
      fi
      
    done
  done
done



# 等待所有后台任务完成
wait
echo "All hyperparameter combinations completed!"