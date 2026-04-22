
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

# 日志目录
log_dir="../../logs/Antarctic_Heat"
if [ ! -d "$log_dir" ]; then
  mkdir -p "$log_dir"
fi

# 主日志文件
main_log="$log_dir/training_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Main log file: $main_log"

for window_size in "${window_sizes[@]}"; do
  for contrastive_weight in "${contrastive_weights[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      # 为每个实验生成唯一的日志文件名
      exp_log="$log_dir/ws${window_size}_cw${contrastive_weight}_lr${learning_rate}.log"

      echo "==========================================" | tee -a "$main_log"
      echo "Running experiment:" | tee -a "$main_log"
      echo "  window_size=$window_size" | tee -a "$main_log"
      echo "  contrastive_weight=$contrastive_weight" | tee -a "$main_log"
      echo "  learning_rate=$learning_rate" | tee -a "$main_log"
      echo "  log file: $exp_log" | tee -a "$main_log"
      echo "==========================================" | tee -a "$main_log"

      # 执行训练，同时输出到终端和日志文件
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
        --train_epochs $train_epochs 2>&1 | tee "$exp_log"

      echo "Experiment completed!" | tee -a "$main_log"
      echo "" | tee -a "$main_log"
    done
  done
done

echo "==========================================" | tee -a "$main_log"
echo "All experiments completed!" | tee -a "$main_log"
echo "Main log file: $main_log" | tee -a "$main_log"
echo "Individual logs: $log_dir/*.log" | tee -a "$main_log"
