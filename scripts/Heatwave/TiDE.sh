
export CUDA_VISIBLE_DEVICES=0

model_name=TiDE

# 更新 run.py 和 dataset 的路径
run_path="../../run.py"
root_path="../../dataset/Northwest_Heatwave"


# 12
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path Northwest_Heatwave.csv \
  --model_id Northwest_Heatwave_96_12 \
  --model $model_name \
  --data Northwest_Heatwave \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 70 \
  --dec_in 70 \
  --c_out 70 \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.3 \
  --batch_size 64 \
  --learning_rate 0.1 \



# 24
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path Northwest_Heatwave.csv \
  --model_id Northwest_Heatwave_96_24 \
  --model $model_name \
  --data Northwest_Heatwave \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 70 \
  --dec_in 70 \
  --c_out 70 \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.3 \
  --batch_size 64 \
  --learning_rate 0.1 \



# 48
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path Northwest_Heatwave.csv \
  --model_id Northwest_Heatwave_96_48 \
  --model $model_name \
  --data Northwest_Heatwave \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 70 \
  --dec_in 70 \
  --c_out 70 \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.3 \
  --batch_size 64 \
  --learning_rate 0.1 \

# 96
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path Northwest_Heatwave.csv \
  --model_id Northwest_Heatwave_96_96 \
  --model $model_name \
  --data Northwest_Heatwave \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 70 \
  --dec_in 70 \
  --c_out 70 \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.3 \
  --batch_size 64 \
  --learning_rate 0.1 \