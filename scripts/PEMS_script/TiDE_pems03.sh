export CUDA_VISIBLE_DEVICES=0

model_name=TiDE

run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/PEMS"


python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_96 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --d_model 32 \
  --d_ff 32 \
  --dropout 0.3 \
  --batch_size 32 \
  --learning_rate 0.1 \
  --patience 3 \
  --train_epochs 5 \


