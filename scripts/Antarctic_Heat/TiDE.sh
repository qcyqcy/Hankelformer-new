
export CUDA_VISIBLE_DEVICES=0

model_name=TiDE

# 更新 run.py 和 dataset 的路径
run_path="../../run.py"
root_path="../../dataset/Antarctic_heat"


# 12
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path antarctic_weather.csv \
  --model_id Antarctic_heat_96_12 \
  --model $model_name \
  --data Antarctic_Heat \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 22 \
  --dec_in 22 \
  --c_out 22\
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.3 \
  --batch_size 64 \
  --learning_rate 0.1 \


