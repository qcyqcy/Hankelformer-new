export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

run_path="../../run.py"
root_path="../../dataset/L90"


# # # 无噪声
python -u $run_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path coupled_lorenz_clean.csv \
  --model_id coupled_lorenz_clean_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 90 \
  --dec_in 90 \
  --c_out 90 \
  --d_model 256 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 


# # # # 0.1噪声
# # python -u $run_path \
# #   --task_name long_term_forecast \
# #   --is_training 1 \
# #   --root_path $root_path \
# #   --data_path coupled_lorenz_noise_0.1.csv \
# #   --model_id coupled_lorenz_noise_0.1_96_96 \
# #   --model $model_name \
# #   --data custom \
# #   --features M \
# #   --seq_len 96 \
# #   --label_len 48 \
# #   --pred_len 96 \
# #   --e_layers 2 \
# #   --d_layers 1 \
# #   --factor 3 \
# #   --enc_in 90 \
# #   --dec_in 90 \
# #   --c_out 90 \
# #   --d_model 256 \
# #   --d_ff 32 \
# #   --des 'Exp' \
# #   --itr 1 \
# #   --top_k 5 


# # # # 0.2噪声
# # python -u $run_path \
# #   --task_name long_term_forecast \
# #   --is_training 1 \
# #   --root_path $root_path \
# #   --data_path coupled_lorenz_noise_0.2.csv \
# #   --model_id coupled_lorenz_noise_0.2_96_96 \
# #   --model $model_name \
# #   --data custom \
# #   --features M \
# #   --seq_len 96 \
# #   --label_len 48 \
# #   --pred_len 96 \
# #   --e_layers 2 \
# #   --d_layers 1 \
# #   --factor 3 \
# #   --enc_in 90 \
# #   --dec_in 90 \
# #   --c_out 90 \
# #   --d_model 256 \
# #   --d_ff 32 \
# #   --des 'Exp' \
# #   --itr 1 \
# #   --top_k 5 



#   # # 0.3噪声
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path coupled_lorenz_noise_0.3.csv \
#   --model_id coupled_lorenz_noise_0.3_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 90 \
#   --dec_in 90 \
#   --c_out 90 \
#   --d_model 256 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 




#   # # 0.4噪声
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path coupled_lorenz_noise_0.4.csv \
#   --model_id coupled_lorenz_noise_0.4_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 90 \
#   --dec_in 90 \
#   --c_out 90 \
#   --d_model 256 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 



#   # # 0.5噪声
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path coupled_lorenz_noise_0.5.csv \
#   --model_id coupled_lorenz_noise_0.5_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 90 \
#   --dec_in 90 \
#   --c_out 90 \
#   --d_model 256 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 



#   # # 0.6噪声
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path coupled_lorenz_noise_0.6.csv \
#   --model_id coupled_lorenz_noise_0.6_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 90 \
#   --dec_in 90 \
#   --c_out 90 \
#   --d_model 256 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 



#   # # 0.7噪声
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path coupled_lorenz_noise_0.7.csv \
#   --model_id coupled_lorenz_noise_0.7_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 90 \
#   --dec_in 90 \
#   --c_out 90 \
#   --d_model 256 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 



#   # # 0.8噪声
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path coupled_lorenz_noise_0.8.csv \
#   --model_id coupled_lorenz_noise_0.8_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 90 \
#   --dec_in 90 \
#   --c_out 90 \
#   --d_model 256 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 



#   # # 0.9噪声
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path coupled_lorenz_noise_0.9.csv \
#   --model_id coupled_lorenz_noise_0.9_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 90 \
#   --dec_in 90 \
#   --c_out 90 \
#   --d_model 256 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 



#   # # 1.0噪声
# python -u $run_path \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path coupled_lorenz_noise_1.0.csv \
#   --model_id coupled_lorenz_noise_1.0_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 90 \
#   --dec_in 90 \
#   --c_out 90 \
#   --d_model 256 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 