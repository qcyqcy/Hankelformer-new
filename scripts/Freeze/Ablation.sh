export CUDA_VISIBLE_DEVICES=0


run_path="../../run.py"
root_path="../../dataset/Texas_Freeze"


# Hankelformer_base
python -u $run_path \
    --task_name long_term_forecast\
    --is_training 1 \
    --root_path $root_path/ \
    --data_path Texas_Freeze.csv \
    --model_id Texas_96_48 \
    --model Hankelformer_base \
    --data Texas_Freeze \
    --features M \
    --seq_len 96 \
    --pred_len 48 \
    --e_layers 2 \
    --enc_in 121 \
    --dec_in 121 \
    --c_out 121 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --batch_size 64 \
    --learning_rate 0.0003





# Hankelformer_without_hankel
python -u $run_path \
    --task_name long_term_forecast_contra\
    --is_training 1 \
    --root_path $root_path/ \
    --data_path Texas_Freeze.csv \
    --model_id Texas_96_48 \
    --model Hankelformer_without_hankel \
    --data Texas_Freeze \
    --features M \
    --seq_len 96 \
    --pred_len 48 \
    --e_layers 2 \
    --enc_in 121\
    --dec_in 121 \
    --c_out 121 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --batch_size 64 \
    --window_size 10 \
    --contrastive_weight 0.01 \
    --learning_rate 0.0003



# # Hankelformer_without_Contrastive
python -u $run_path \
    --task_name long_term_forecast_contra\
    --is_training 1 \
    --root_path $root_path/ \
    --data_path Texas_Freeze.csv \
    --model_id Texas_96_48 \
    --model Hankelformer_without_Contrastive \
    --data Texas_Freeze \
    --features M \
    --seq_len 96 \
    --pred_len 48 \
    --e_layers 2 \
    --enc_in 121\
    --dec_in 121 \
    --c_out 121 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --batch_size 64 \
    --window_size 10 \
    --contrastive_weight 0 \
    --learning_rate 0.0003

