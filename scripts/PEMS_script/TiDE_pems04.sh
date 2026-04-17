export CUDA_VISIBLE_DEVICES=0

model_name=TiDE

run_path="../../run.py"
root_path="../../dataset/PEMS"



python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_96 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --d_model 64 \
  --d_ff 64 \
  --dropout 0.3 \
  --batch_size 16 \
  --learning_rate 0.1 \
  --patience 3 \
  --train_epochs 10 \


