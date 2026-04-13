export CUDA_VISIBLE_DEVICES=1

model_name=TimeMixer



run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/PEMS"


e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=64
train_epochs=20
patience=3

# # 12
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path PEMS03.npz \
#   --model_id PEMS03_96_12 \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --label_len 0 \
#   --pred_len 12 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window

# # 24
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path PEMS03.npz \
#   --model_id PEMS03_96_24 \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --label_len 0 \
#   --pred_len 24 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window



# # 48
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path PEMS03.npz \
#   --model_id PEMS03_96_48 \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --label_len 0 \
#   --pred_len 48 \
#   --e_layers $e_layers \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window



# 96
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_96 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window