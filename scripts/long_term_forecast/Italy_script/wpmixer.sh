# 官方weather


# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer
run_path="/share/home/qinchengyang/Time-Series-Library/run.py"
root_path="/share/home/qinchengyang/Time-Series-Library/dataset/italy_HEATWAVE"


# Datasets and prediction lengths
dataset=Italy
seq_lens=(96 96 96 96)
pred_lens=(12 24 48 96)
learning_rates=(0.006434312 0.003129047 0.004176281 0.001950107)
batches=(128 128 128 128)
wavelets=(sym3 coif4 coif4 sym2)
levels=(1 1 1 1)
tfactors=(3 5 7 7)
dfactors=(5 3 7 5)
epochs=(30 30 30 30)
dropouts=(0.2 0.4 0.4 0.4)
embedding_dropouts=(0.0 0.2 0.1 0.0)
patch_lens=(16 16 16 16)
strides=(8 8 8 8)
lradjs=(type3 type3 type3 type3)
d_models=(128 256 128 128)
patiences=(5 5 5 5)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	python -u $run_path \
		--is_training 1\
		--model $model_name \
		--model_id italy_96_${pred_lens[$i]} \
		--root_path $root_path\
		--data_path italy_temperature_data.csv  \
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
		--enc_in 214 \
		--dec_in 214 \
		--c_out 214 \
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

