
export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# 更新 run.py 和 dataset 的路径
run_path="../../run.py"
root_path="../../dataset/Antarctic_heat"
# # 12
python -u $run_path \
  --task_name long_term_forecast\
  --is_training 1 \
  --root_path $root_path/ \
  --data_path antarctic_weather.csv \
  --model_id Antarctic_heat_96_12 \
  --model $model_name \
  --data Antarctic_Heat \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 22 \
  --dec_in 22 \
  --c_out 22\
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64\
  --itr 1 \
  --train_epochs 20


