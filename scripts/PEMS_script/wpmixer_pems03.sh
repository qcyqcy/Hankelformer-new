
# 官方

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=1

# Model name
model_name=WPMixer

run_path="../../run.py"
root_path="../../dataset/PEMS"


# # Datasets and prediction lengths
# dataset=PEMS
# seq_lens=(96 96 96 96)
# pred_lens=(12 24 48 96)
# learning_rates=(0.003243031 0.002875673 0.002429664 0.004192695)
# batches=(8 8 8 8)
# wavelets=(db5 db3 sym2 coif4)
# levels=(1 1 1 1)
# tfactors=(5 3 7 5)
# dfactors=(5 7 8 5)
# epochs=(30 30 30 30)
# dropouts=(0.0 0.05 0.1 0.05)
# embedding_dropouts=(0.05 0.0 0.0 0.1)
# patch_lens=(12 12 12 12)
# strides=(6 6 6 6)
# lradjs=(type3 type3 type3 type3)
# d_models=(64 64 64 64)
# patiences=(3 3 3 3)


# Datasets and prediction lengths
dataset=PEMS
seq_lens=(96)
pred_lens=(96)
learning_rates=(0.004192695)
batches=(8)
wavelets=(coif4)
levels=(1)
tfactors=(5)
dfactors=(5)
epochs=(30)
dropouts=(0.05)
embedding_dropouts=(0.1)
patch_lens=(12)
strides=(6)
lradjs=(type3)
d_models=(64)
patiences=(3)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	python -u $run_path \
    --is_training 1\
		--model $model_name \
    --model_id pems03_${pred_lens[$i]} \
    --root_path $root_path\
    --data_path PEMS03.npz  \
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
		--enc_in 358 \
		--dec_in 358 \
		--c_out 358 \
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







# # 变量赋值不能有空格
# run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
# root_path="/share/home/qinchengyang/Time-Series-Library/dataset/PEMS"

# # Set the GPU to use
# export CUDA_VISIBLE_DEVICES=1

# # Model name
# model_name=WPMixer


# # 12
# python -u $run_path \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path PEMS03.npz \
#   --model_id PEMS03_96_12 \
#   --model $model_name \
#   --task_name long_term_forecast \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 12 \
#   --label_len 0 \
#   --d_model 256 \
#   --patch_len 16 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --dropout 0.1 \
#   --batch_size 64 \
#   --lradj type3 \
#   --use_amp \
#   --train_epochs 20


# #24
# python -u $run_path \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path PEMS03.npz \
#   --model_id PEMS03_96_24 \
#   --model $model_name \
#   --task_name long_term_forecast \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 24 \
#   --label_len 0 \
#   --d_model 256 \
#   --patch_len 16 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --dropout 0.1 \
#   --batch_size 64 \
#   --lradj type3 \
#   --use_amp \
#   --train_epochs 20


# #48
# python -u $run_path \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path PEMS03.npz \
#   --model_id PEMS03_96_48 \
#   --model $model_name \
#   --task_name long_term_forecast \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 48 \
#   --label_len 0 \
#   --d_model 256 \
#   --patch_len 16 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --dropout 0.1 \
#   --batch_size 64 \
#   --lradj type3 \
#   --use_amp \
#   --train_epochs 20


# #96
# python -u $run_path \
#   --is_training 1 \
#   --root_path $root_path \
#   --data_path PEMS03.npz \
#   --model_id PEMS03_96_96 \
#   --model $model_name \
#   --task_name long_term_forecast \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --label_len 0 \
#   --d_model 256 \
#   --patch_len 16 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --dropout 0.1 \
#   --batch_size 64 \
#   --lradj type3 \
#   --use_amp \
#   --train_epochs 20