export CUDA_VISIBLE_DEVICES=0


run_path="../../run.py"
root_path="../../dataset/ECL/"



# # Hankelformer_base
# python -u $run_path \
#   --is_training 1 \
#   --task_name long_term_forecast\
#   --root_path $root_path/ \
#   --data_path electricity.csv \
#   --model_id ECL_96_96 \
#   --model Hankelformer_base \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1 \
#   --batch_size 16 \
#   --learning_rate 0.0005 \
#   --train_epochs 20


# # Hankelformer_base
# python -u $run_path \
#   --is_training 1 \
#   --task_name long_term_forecast_contra\
#   --root_path $root_path/ \
#   --data_path electricity.csv \
#   --model_id ECL_96_96 \
#   --model Hankelformer_without_hankel \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1 \
#   --batch_size 16 \
#   --contrastive_weight 0.03 \
#   --learning_rate 0.0005 \
#   --train_epochs 20


# Hankelformer_without_Contrastive
python -u $run_path \
  --is_training 1 \
  --task_name long_term_forecast_contra\
  --root_path $root_path/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model Hankelformer_without_Contrastive \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --batch_size 16 \
  --window_size 18 \
  --contrastive_weight 0 \
  --learning_rate 0.0005 \
  --train_epochs 20