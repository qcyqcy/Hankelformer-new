#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=Hankelformer
run_path="../../run.py"
root_path="../../dataset/Antarctic_heat"

# 初筛参数范围（尽量大）
window_sizes=(1 2 3 4 5 6 7 8)
contrastive_weights=(0.00001 0.00003 0.00005 0.00007 0.0001 0.0003 0.0005 0.001)
learning_rates=(0.00005 0.00008 0.0001 0.00015 0.0002 0.0003 0.0005)

# 固定参数
seq_len=96
pred_len=12
e_layers=2
enc_in=22
dec_in=22
c_out=22
d_model=512
d_ff=512
batch_size=64
itr=1
train_epochs=15

echo "=========================================="
echo "Antarctic_Heat 初筛 (pred_len=$pred_len)"
echo "=========================================="

for window_size in "${window_sizes[@]}"; do
  for contrastive_weight in "${contrastive_weights[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      echo "Running: ws=$window_size, cw=$contrastive_weight, lr=$learning_rate"

      python -u $run_path \
        --task_name long_term_forecast_contra \
        --is_training 1 \
        --root_path $root_path/ \
        --data_path antarctic_weather.csv \
        --model_id Antarctic_heat_${seq_len}_${pred_len}_ws${window_size}_cw${contrastive_weight}_lr${learning_rate} \
        --model $model_name \
        --data Antarctic_Heat \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers $e_layers \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --des 'Exp' \
        --d_model $d_model \
        --d_ff $d_ff \
        --batch_size $batch_size \
        --itr $itr \
        --window_size $window_size \
        --contrastive_weight $contrastive_weight \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs

      echo "Done: ws=$window_size, cw=$contrastive_weight, lr=$learning_rate"
      echo ""
    done
  done
done

echo "=========================================="
echo "All experiments completed! (pred_len=$pred_len)"
echo "=========================================="