
export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

# 更新 run.py 和 dataset 的路径
run_path="../../run.py"
root_path="../../dataset/Northwest_Heatwave"
# # 12
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path Northwest_Heatwave.csv \
  --model_id Northwest_Heatwave_96_12 \
  --model $model_name \
  --data Northwest_Heatwave \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 70 \
  --dec_in 70 \
  --c_out 70\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20



# 24
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path Northwest_Heatwave.csv \
  --model_id Northwest_Heatwave_96_24 \
  --model $model_name \
  --data Northwest_Heatwave \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 70 \
  --dec_in 70 \
  --c_out 70\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20


# 48
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path Northwest_Heatwave.csv \
  --model_id Northwest_Heatwave_96_48 \
  --model $model_name \
  --data Northwest_Heatwave \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 70 \
  --dec_in 70 \
  --c_out 70\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20

# #96

python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path Northwest_Heatwave.csv \
  --model_id Northwest_Heatwave_96_96 \
  --model $model_name \
  --data Northwest_Heatwave \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 70 \
  --dec_in 70 \
  --c_out 70\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20

