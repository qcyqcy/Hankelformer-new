export CUDA_VISIBLE_DEVICES=0

model_name=TiDE

run_path="../../run.py"
root_path="../../dataset/ECL/"


python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.3 \
  --batch_size 16 \
  --learning_rate 0.1 \
  --patience 5 \
  --train_epochs 2 \


