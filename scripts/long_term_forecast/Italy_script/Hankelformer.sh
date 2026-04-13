
export CUDA_VISIBLE_DEVICES=1

model_name=Hankelformer

# 更新 run.py 和 dataset 的路径
run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/italy_HEATWAVE"

# 用并行跑

# window_sizes=(1 2 3 4 5 6 7 8 9 10 11 12)
# contrastive_weights=(0.01 0.05 0.1 0.3 0.5 0.7)
# learning_rates=(0.0001 0.0005 0.001 0.005 0.01)

# 粗略搜索
window_sizes=(1)
contrastive_weights=(0.0002499391163583297)
learning_rates=(0.0004873574217409738)

# [I 2025-06-17 23:30:49,387] Trial 57 finished with value: 0.13994011282920837 and parameters: {'window_size': 1, 
# 'contrastive_weight': 0.0001692551076173882, 
# 'learning_rate': 0.00047936853398266747, 'e_layers': 4, 'd_model': 512, 'd_ff': 512}. Best is trial 57 with value: 0.13994011282920837.

# 0.1397317349910736 and parameters: {'window_size': 1, 'contrastive_weight': 0.0002499391163583297, 
# 'learning_rate': 0.0004873574217409738, 'e_layers': 4, 'd_model': 512, 'd_ff': 512}.



for window_size in "${window_sizes[@]}"; do
  for contrastive_weight in "${contrastive_weights[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      
      python -u $run_path \
        --task_name long_term_forecast_contra\
        --is_training 1 \
        --root_path $root_path/ \
        --data_path italy_temperature_data.csv \
        --model_id italy_Heatwave_96_48 \
        --model $model_name \
        --data custom_weather \
        --features M \
        --seq_len 96 \
        --pred_len 48 \
        --e_layers 4 \
        --enc_in 214 \
        --dec_in 214 \
        --c_out 214\
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --batch_size 128\
        --itr 1 \
        --train_epochs 20 \
        --window_size $window_size \
        --contrastive_weight $contrastive_weight \
        --learning_rate $learning_rate

    done
  done
done


