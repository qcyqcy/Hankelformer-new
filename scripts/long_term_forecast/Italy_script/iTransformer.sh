
export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

# 更新 run.py 和 dataset 的路径
run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/italy_HEATWAVE"

# # 12
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path italy_temperature_data.csv \
  --model_id italy_96_12 \
  --model $model_name \
  --data Italy \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 214 \
  --dec_in 214 \
  --c_out 214\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \


# # 24
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path italy_temperature_data.csv \
  --model_id italy_96_24 \
  --model $model_name \
  --data Italy \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 214 \
  --dec_in 214 \
  --c_out 214\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \



# # 48
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path italy_temperature_data.csv \
  --model_id italy_96_48 \
  --model $model_name \
  --data Italy \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 214 \
  --dec_in 214 \
  --c_out 214\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \


# # 96
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path italy_temperature_data.csv \
  --model_id italy_96_96 \
  --model $model_name \
  --data Italy \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 214 \
  --dec_in 214 \
  --c_out 214\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \