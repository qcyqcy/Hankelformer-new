
export CUDA_VISIBLE_DEVICES=0


model_name=TimeMixer


# 更新 run.py 和 dataset 的路径
run_path="../../run.py"
root_path="../../dataset/Antarctic_heat"



down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=3
batch_size=16

# 12
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path antarctic_weather.csv \
  --model_id Antarctic_heat_96_12 \
  --model $model_name \
  --data Antarctic_Heat \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 22 \
  --c_out 22 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 64 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


