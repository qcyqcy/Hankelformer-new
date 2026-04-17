# 官方

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/traffic"

# Datasets and prediction lengths
dataset=custom
seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
learning_rates=(0.003243031 0.002875673 0.002429664 0.004192695)
batches=(8 8 8 8)
wavelets=(db5 db3 sym2 coif4)
levels=(1 1 1 1)
tfactors=(5 3 7 5)
dfactors=(5 7 8 5)
epochs=(20 20 20 20)
dropouts=(0.0 0.05 0.1 0.05)
embedding_dropouts=(0.05 0.0 0.0 0.1)
patch_lens=(12 12 12 12)
strides=(6 6 6 6)
lradjs=(type3 type3 type3 type3)
d_models=(64 64 64 64)
patiences=(10 10 10 10)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	python -u $run_path \
    --is_training 1\
		--model $model_name \
    --model_id traffic_${pred_lens[$i]} \
    --root_path $root_path\
    --data_path traffic.csv \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--tfactor ${tfactors[$i]} \
		--dfactor ${dfactors[$i]} \
		--wavelet ${wavelets[$i]} \
		--level ${levels[$i]} \
		--patch_len ${patch_lens[$i]} \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
		--stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--embedding_dropout ${embedding_dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp 
done














# 自己写的
# export CUDA_VISIBLE_DEVICES=1

# run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
# root_path="/share/home/qinchengyang/Time-Series-Library/dataset/traffic"

# # Model name
# model_name=WPMixer



# # Model params below need to be set in WPMixer.py Line 15, instead of this script
# wavelets=(db3 db3 bior3.1 db3)
# levels=(1 1 1 1)
# tfactors=(3 3 7 7)
# dfactors=(5 5 7 3)
# strides=(8 8 8 8)

# # # 96
# # python -u $run_path \
# #   --is_training 1 \
# #   --root_path $root_path \
# #   --data_path traffic.csv \
# #   --model_id traffic_96_96 \
# #   --model $model_name \
# #   --task_name long_term_forecast \
# #   --data custom \
# #   --features M \
# #   --seq_len 96 \
# #   --pred_len 96 \
# #   --label_len 0 \
# #   --d_model 32 \
# #   --patch_len 16 \
# #   --enc_in 862 \
# #   --dec_in 862 \
# #   --c_out 862 \
# #   --dropout 0.05 \
# #   --batch_size 16 \
# #   --lradj type3 \
# #   --use_amp \
# #   --train_epochs 20 \
# #   --learning_rate 0.001 \


# # 96
# python -u $run_path \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path traffic.csv \
#   --model_id traffic_96_192 \
#   --model $model_name \
#   --task_name long_term_forecast \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 192 \
#   --label_len 0 \
#   --d_model 32 \
#   --patch_len 16 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --dropout 0.05 \
#   --batch_size 16 \
#   --lradj type3 \
#   --use_amp \
#   --train_epochs 20 \
#   --learning_rate 0.001 \


# # 96
# python -u $run_path \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path traffic.csv \
#   --model_id traffic_96_336 \
#   --model $model_name \
#   --task_name long_term_forecast \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 336 \
#   --label_len 0 \
#   --d_model 32 \
#   --patch_len 16 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --dropout 0.05 \
#   --batch_size 16 \
#   --lradj type3 \
#   --use_amp \
#   --train_epochs 20 \
#   --learning_rate 0.001 \


# # 96
# python -u $run_path \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path traffic.csv \
#   --model_id traffic_96_720 \
#   --model $model_name \
#   --task_name long_term_forecast \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 720 \
#   --label_len 0 \
#   --d_model 32 \
#   --patch_len 16 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --dropout 0.05 \
#   --batch_size 16 \
#   --lradj type3 \
#   --use_amp \
#   --train_epochs 20 \
#   --learning_rate 0.001 \