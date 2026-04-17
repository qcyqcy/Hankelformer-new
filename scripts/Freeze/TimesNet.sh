export CUDA_VISIBLE_DEVICES=1

model_name=TimesNet

run_path="../../run.py"
root_path="../../dataset/Texas_Freeze"


# # 12
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path Texas_Freeze.csv \
#   --model_id Texas_96_12 \
#   --model $model_name \
#   --data Texas_Freeze \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 12 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 2 \
#   --enc_in 121 \
#   --dec_in 121 \
#   --c_out 121 \
#   --d_model 256 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 3 


# # 24
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path Texas_Freeze.csv \
#   --model_id Texas_96_24 \
#   --model $model_name \
#   --data Texas_Freeze \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 2 \
#   --enc_in 121 \
#   --dec_in 121 \
#   --c_out 121 \
#   --d_model 256 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 3 



#   # 48
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path Texas_Freeze.csv \
#   --model_id Texas_96_48 \
#   --model $model_name \
#   --data Texas_Freeze \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 48 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 2 \
#   --enc_in 121 \
#   --dec_in 121 \
#   --c_out 121 \
#   --d_model 256 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 3 



#   # 96
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path Texas_Freeze.csv \
#   --model_id Texas_96_96 \
#   --model $model_name \
#   --data Texas_Freeze \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 2 \
#   --enc_in 121 \
#   --dec_in 121 \
#   --c_out 121 \
#   --d_model 256 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 3 




  # 48
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path Texas_Freeze.csv \
  --model_id Texas_96_48 \
  --model $model_name \
  --data Texas_Freeze \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 2 \
  --enc_in 121 \
  --dec_in 121 \
  --c_out 121 \
  --d_model 64 \
  --d_ff 16 \
  --des 'Exp' \
  --itr 1 \
  --top_k 2 

