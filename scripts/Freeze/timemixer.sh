
export CUDA_VISIBLE_DEVICES=1


model_name=TimeMixer


# 更新 run.py 和 dataset 的路径
run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/Texas_Freeze"



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
  --data_path Texas_Freeze.csv \
  --model_id Texas_96_12 \
  --model $model_name \
  --data Texas_Freeze \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 121 \
  --c_out 121 \
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




# 24
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path Texas_Freeze.csv \
  --model_id Texas_96_24 \
  --model $model_name \
  --data Texas_Freeze \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 121 \
  --c_out 121 \
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



# 48
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path Texas_Freeze.csv \
  --model_id Texas_96_48 \
  --model $model_name \
  --data Texas_Freeze \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 121 \
  --c_out 121 \
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


# 48
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path/\
  --data_path Texas_Freeze.csv \
  --model_id Texas_96_96 \
  --model $model_name \
  --data Texas_Freeze \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 121 \
  --c_out 121 \
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
