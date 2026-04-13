# 官方的
# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/electricity"



dataset=custom
seq_lens=(96)
pred_lens=(96)
learning_rates=(0.000872897)
batches=(32)
wavelets=(coif5)
levels=(1)
tfactors=(7)
dfactors=(7)
epochs=(20)
dropouts=(0.0)
embedding_dropouts=(0.05)
patch_lens=(16)
strides=(8)
lradjs=(type3)
d_models=(256)
patiences=(10)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	python -u $run_path \
    --is_training 1 \
	--model $model_name \
    --model_id ECL_96_${pred_lens[$i]} \
    --root_path $root_path\
    --data_path electricity.csv \
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
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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















# Datasets and prediction lengths
# dataset=custom
# seq_lens=(96 96 96 96)
# pred_lens=(96 192 336 720)
# learning_rates=(0.000872897 0.00130044 0.001079003 0.001215337)
# batches=(32 32 32 32)
# wavelets=(coif5 sym3 sym3 sym3)
# levels=(1 1 1 1)
# tfactors=(7 7 7 7)
# dfactors=(7 7 7 7)
# epochs=(20 20 20 20)
# dropouts=(0.0 0.0 0.0 0.0)
# embedding_dropouts=(0.05 0.1 0.1 0.1)
# patch_lens=(16 16 16 16)
# strides=(8 8 8 8)
# lradjs=(type3 type3 type3 type3)
# d_models=(256 256 256 256)
# patiences=(10 10 10 10)


# # Loop over datasets and prediction lengths
# for i in "${!pred_lens[@]}"; do
# 	python -u $run_path \
#     --is_training 1 \
# 		--model $model_name \
#     --model_id ele_${pred_lens[$i]} \
#     --root_path $root_path\
#     --data_path electricity.csv \
# 		--task_name long_term_forecast \
# 		--data $dataset \
# 		--seq_len ${seq_lens[$i]} \
# 		--pred_len ${pred_lens[$i]} \
# 		--d_model ${d_models[$i]} \
# 		--tfactor ${tfactors[$i]} \
# 		--dfactor ${dfactors[$i]} \
# 		--wavelet ${wavelets[$i]} \
# 		--level ${levels[$i]} \
# 		--patch_len ${patch_lens[$i]} \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
# 		--stride ${strides[$i]} \
# 		--batch_size ${batches[$i]} \
# 		--learning_rate ${learning_rates[$i]} \
# 		--lradj ${lradjs[$i]} \
# 		--dropout ${dropouts[$i]} \
# 		--embedding_dropout ${embedding_dropouts[$i]} \
# 		--patience ${patiences[$i]} \
# 		--train_epochs ${epochs[$i]} \
# 		--use_amp 
# done



