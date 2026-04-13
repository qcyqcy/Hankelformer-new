
export CUDA_VISIBLE_DEVICES=0

model_name=Hankelformer

# 更新 run.py 和 dataset 的路径
run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/ETT-small"


# 超参数范围
window_sizes=(2 3 4 5 6 7 8 9 10)
contrastive_weights=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)
learning_rates=(0.00001 0.00002 0.00003 0.00004 0.00005 0.00006 0.00007 0.00008 0.00009 0.0001 0.00015 0.0002)

# 循环遍历所有超参数组合
for window_size in "${window_sizes[@]}"; do
  for contrastive_weight in "${contrastive_weights[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      python -u $run_path \
        --task_name long_term_forecast_contra\
        --is_training 1 \
        --root_path $root_path/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_720 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --pred_len 720 \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --window_size $window_size \
        --contrastive_weight $contrastive_weight \
        --learning_rate $learning_rate
    done
  done
done
