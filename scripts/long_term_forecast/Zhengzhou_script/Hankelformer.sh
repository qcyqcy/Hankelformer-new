
export CUDA_VISIBLE_DEVICES=1

model_name=Hankelformer

# 更新 run.py 和 dataset 的路径
run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/ZZ-precipitation"


# # # # 超参数范围
window_sizes=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24) 
contrastive_weights=(0.0001 0.0005 0.001  0.005 0.01 0.05)
learning_rates=(0.00005 0.0001 0.0003 0.0005 0.001)


for window_size in "${window_sizes[@]}"; do
  for contrastive_weight in "${contrastive_weights[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      python -u $run_path \
        --task_name long_term_forecast_contra\
        --is_training 1 \
        --root_path $root_path/ \
        --data_path zhengzhou_hourly_weather_precipitation_2016_2021_complete.csv \
        --model_id zhengzhou_96_12 \
        --model $model_name \
        --data Zhengzhou \
        --features M \
        --seq_len 96 \
        --pred_len 12 \
        --e_layers 2 \
        --enc_in 23 \
        --dec_in 23 \
        --c_out 23\
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --batch_size 128\
        --itr 1 \
        --window_size $window_size \
        --contrastive_weight $contrastive_weight \
        --learning_rate $learning_rate \
        --train_epochs 20
    done
  done
done



