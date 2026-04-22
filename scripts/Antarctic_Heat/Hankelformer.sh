
export CUDA_VISIBLE_DEVICES=0

model_name=Hankelformer

# 更新 run.py 和 dataset 的路径
run_path="../../run.py"
root_path="../../dataset/Antarctic_heat"

# # 12


# python -u $run_path \
#         --task_name long_term_forecast_contra\
#         --is_training 1 \
#         --root_path $root_path/ \
#         --data_path antarctic_weather.csv \
#         --model_id Antarctic_heat_96_12 \
#         --model $model_name \
#         --data Antarctic_Heat \
#         --features M \
#         --seq_len 96 \
#         --pred_len 12 \
#         --e_layers 2 \
#         --enc_in 22 \
#         --dec_in 22 \
#         --c_out 22\
#         --des 'Exp' \
#         --d_model 512 \
#         --d_ff 512 \
#         --batch_size 64\
#         --itr 1 \
#         --window_size 2 \
#         --contrastive_weight 0.00005 \
#         --learning_rate 0.0012



# ==================== Antarctic_Heat 调参版本 ====================
# 需要调优的参数范围
window_sizes=(1 2 3 6 9 12 15 18 21 24)
contrastive_weights=(0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01)
learning_rates=(0.0001 0.0002 0.0005 0.001 0.0012 0.002)

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
train_epochs=20

for window_size in "${window_sizes[@]}"; do
  for contrastive_weight in "${contrastive_weights[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      echo "=========================================="
      echo "Running experiment:"
      echo "  window_size=$window_size"
      echo "  contrastive_weight=$contrastive_weight"
      echo "  learning_rate=$learning_rate"
      echo "=========================================="

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

      echo "Experiment completed!"
      echo ""
    done
  done
done

echo "All experiments completed!"
