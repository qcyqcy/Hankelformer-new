export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

run_path="../../run.py"
root_path="../../dataset/PEMS"



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
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --d_model 256 \
  --d_ff 256 \
  --top_k 5 \
  --des 'Exp' \
  --batch_size 64 \
  --itr 1

