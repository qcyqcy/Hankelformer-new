
export CUDA_VISIBLE_DEVICES=0

model_name=Hankelformer

# 更新 run.py 和 dataset 的路径
run_path="../../run.py"
root_path="../../dataset/Antarctic_heat"

# # 12


python -u $run_path \
        --task_name long_term_forecast_contra\
        --is_training 1 \
        --root_path $root_path/ \
        --data_path antarctic_weather.csv \
        --model_id Antarctic_heat_96_12 \
        --model $model_name \
        --data Antarctic_Heat \
        --features M \
        --seq_len 96 \
        --pred_len 12 \
        --e_layers 2 \
        --enc_in 22 \
        --dec_in 22 \
        --c_out 22\
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --batch_size 64\
        --itr 1 \
        --window_size 2 \
        --contrastive_weight 0.00005 \
        --learning_rate 0.0012

 












# # # # # 超参数范围
# window_sizes=(1 3 6 9 12 15 18 21 24)  
# contrastive_weights=(0.0001 0.001 0.01)  # 减少到3个关键值，跨度更大
# learning_rates=(0.0001 0.0005 0.001)  # 减少到3个常用值


# 验证学习率

# window_sizes=(21)
# contrastive_weights=(0.01)
# learning_rates=(0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008)


# for window_size in "${window_sizes[@]}"; do
#   for contrastive_weight in "${contrastive_weights[@]}"; do
#     for learning_rate in "${learning_rates[@]}"; do
#       python -u $run_path \
#         --task_name long_term_forecast_contra\
#         --is_training 1 \
#         --root_path $root_path/ \
#         --data_path Northwest_Heatwave.csv \
#         --model_id Northwest_Heatwave_96_48 \
#         --model $model_name \
#         --data Northwest_Heatwave \
#         --features M \
#         --seq_len 96 \
#         --pred_len 48 \
#         --e_layers 2 \
#         --enc_in 70 \
#         --dec_in 70 \
#         --c_out 70\
#         --des 'Exp' \
#         --d_model 512 \
#         --d_ff 512 \
#         --batch_size 64\
#         --itr 1 \
#         --window_size $window_size \
#         --contrastive_weight $contrastive_weight \
#         --learning_rate $learning_rate \
#         --train_epochs 20
#     done
#   done
# done





# ## 原版
# for window_size in "${window_sizes[@]}"; do
#   for contrastive_weight in "${contrastive_weights[@]}"; do
#     for learning_rate in "${learning_rates[@]}"; do
#       python -u $run_path \
#         --task_name long_term_forecast_contra\
#         --is_training 1 \
#         --root_path $root_path/ \
#         --data_path Northwest_Heatwave.csv \
#         --model_id Northwest_Heatwave_96_12 \
#         --model $model_name \
#         --data Northwest_Heatwave \
#         --features M \
#         --seq_len 96 \
#         --pred_len 12 \
#         --e_layers 2 \
#         --enc_in 70 \
#         --dec_in 70 \
#         --c_out 70\
#         --des 'Exp' \
#         --d_model 512 \
#         --d_ff 512 \
#         --batch_size 64\
#         --itr 1 \
#         --window_size $window_size \
#         --contrastive_weight $contrastive_weight \
#         --learning_rate $learning_rate \
#         --train_epochs 20
#     done
#   done
# done

