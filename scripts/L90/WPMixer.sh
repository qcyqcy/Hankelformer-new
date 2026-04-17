export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer
run_path="../../run.py"
root_path="../../dataset/L90"
# # Datasets and prediction lengths
# dataset=custom
# # seq_lens=(96 96 96 96)
# # pred_lens=(12 24 48 96)
# seq_lens=(96)
# pred_lens=(96)
# learning_rates=(0.01)
# batches=(128)
# wavelets=(sym3)
# levels=(1)
# tfactors=(2)
# dfactors=(1)
# epochs=(30)
# dropouts=(0.4)
# embedding_dropouts=(0.1)
# patch_lens=(4)
# strides=(2)
# lradjs=(type3)
# d_models=(64)
# patiences=(5)


# # 无噪声
# python -u $run_path \
# 	--is_training 1\
# 	--model $model_name \
# 	--model_id coupled_lorenz_clean_96_96 \
# 	--root_path $root_path\
# 	--data_path coupled_lorenz_clean.csv \
# 	--task_name long_term_forecast \
# 	--data custom \
# 	--seq_len 96 \
# 	--pred_len 96 \
# 	--d_model 64 \
# 	--tfactor 2 \
# 	--dfactor 1 \
# 	--wavelet sym3 \
# 	--level 1 \
# 	--patch_len 4 \
# 	--enc_in 90 \
# 	--dec_in 90 \
# 	--c_out 90 \
# 	--stride 2 \
# 	--batch_size 128 \
# 	--learning_rate 0.01 \
# 	--lradj type3 \
# 	--dropout 0.4 \
# 	--embedding_dropout 0.1 \
# 	--patience 5 \
# 	--train_epochs 30 \
# 	--use_amp 



# 0.1噪声
python -u $run_path \
	--is_training 1\
	--model $model_name \
	--model_id coupled_lorenz_noise_0.1_96_96 \
	--root_path $root_path\
	--data_path coupled_lorenz_noise_0.1.csv \
	--task_name long_term_forecast \
	--data custom \
	--seq_len 96 \
	--pred_len 96 \
	--d_model 64 \
	--tfactor 2 \
	--dfactor 1 \
	--wavelet sym3 \
	--level 1 \
	--patch_len 4 \
	--enc_in 90 \
	--dec_in 90 \
	--c_out 90 \
	--stride 2 \
	--batch_size 128 \
	--learning_rate 0.01 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.1 \
	--patience 5 \
	--train_epochs 30 \
	--use_amp 


# 0.2噪声
python -u $run_path \
	--is_training 1\
	--model $model_name \
	--model_id coupled_lorenz_noise_0.2_96_96 \
	--root_path $root_path\
	--data_path coupled_lorenz_noise_0.2.csv \
	--task_name long_term_forecast \
	--data custom \
	--seq_len 96 \
	--pred_len 96 \
	--d_model 64 \
	--tfactor 2 \
	--dfactor 1 \
	--wavelet sym3 \
	--level 1 \
	--patch_len 4 \
	--enc_in 90 \
	--dec_in 90 \
	--c_out 90 \
	--stride 2 \
	--batch_size 128 \
	--learning_rate 0.01 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.1 \
	--patience 5 \
	--train_epochs 30 \
	--use_amp 



	# 0.3噪声
python -u $run_path \
	--is_training 1\
	--model $model_name \
	--model_id coupled_lorenz_noise_0.3_96_96 \
	--root_path $root_path\
	--data_path coupled_lorenz_noise_0.3.csv \
	--task_name long_term_forecast \
	--data custom \
	--seq_len 96 \
	--pred_len 96 \
	--d_model 64 \
	--tfactor 2 \
	--dfactor 1 \
	--wavelet sym3 \
	--level 1 \
	--patch_len 4 \
	--enc_in 90 \
	--dec_in 90 \
	--c_out 90 \
	--stride 2 \
	--batch_size 128 \
	--learning_rate 0.01 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.1 \
	--patience 5 \
	--train_epochs 30 \
	--use_amp 



	# 0.4噪声
python -u $run_path \
	--is_training 1\
	--model $model_name \
	--model_id coupled_lorenz_noise_0.4_96_96 \
	--root_path $root_path\
	--data_path coupled_lorenz_noise_0.4.csv \
	--task_name long_term_forecast \
	--data custom \
	--seq_len 96 \
	--pred_len 96 \
	--d_model 64 \
	--tfactor 2 \
	--dfactor 1 \
	--wavelet sym3 \
	--level 1 \
	--patch_len 4 \
	--enc_in 90 \
	--dec_in 90 \
	--c_out 90 \
	--stride 2 \
	--batch_size 128 \
	--learning_rate 0.01 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.1 \
	--patience 5 \
	--train_epochs 30 \
	--use_amp 



	# 0.5噪声
python -u $run_path \
	--is_training 1\
	--model $model_name \
	--model_id coupled_lorenz_noise_0.5_96_96 \
	--root_path $root_path\
	--data_path coupled_lorenz_noise_0.5.csv \
	--task_name long_term_forecast \
	--data custom \
	--seq_len 96 \
	--pred_len 96 \
	--d_model 64 \
	--tfactor 2 \
	--dfactor 1 \
	--wavelet sym3 \
	--level 1 \
	--patch_len 4 \
	--enc_in 90 \
	--dec_in 90 \
	--c_out 90 \
	--stride 2 \
	--batch_size 128 \
	--learning_rate 0.01 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.1 \
	--patience 5 \
	--train_epochs 30 \
	--use_amp 



	# 0.6噪声
python -u $run_path \
	--is_training 1\
	--model $model_name \
	--model_id coupled_lorenz_noise_0.6_96_96 \
	--root_path $root_path\
	--data_path coupled_lorenz_noise_0.6.csv \
	--task_name long_term_forecast \
	--data custom \
	--seq_len 96 \
	--pred_len 96 \
	--d_model 64 \
	--tfactor 2 \
	--dfactor 1 \
	--wavelet sym3 \
	--level 1 \
	--patch_len 4 \
	--enc_in 90 \
	--dec_in 90 \
	--c_out 90 \
	--stride 2 \
	--batch_size 128 \
	--learning_rate 0.01 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.1 \
	--patience 5 \
	--train_epochs 30 \
	--use_amp 



	# 0.7噪声
python -u $run_path \
	--is_training 1\
	--model $model_name \
	--model_id coupled_lorenz_noise_0.7_96_96 \
	--root_path $root_path\
	--data_path coupled_lorenz_noise_0.7.csv \
	--task_name long_term_forecast \
	--data custom \
	--seq_len 96 \
	--pred_len 96 \
	--d_model 64 \
	--tfactor 2 \
	--dfactor 1 \
	--wavelet sym3 \
	--level 1 \
	--patch_len 4 \
	--enc_in 90 \
	--dec_in 90 \
	--c_out 90 \
	--stride 2 \
	--batch_size 128 \
	--learning_rate 0.01 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.1 \
	--patience 5 \
	--train_epochs 30 \
	--use_amp 



	# 0.8噪声
python -u $run_path \
	--is_training 1\
	--model $model_name \
	--model_id coupled_lorenz_noise_0.8_96_96 \
	--root_path $root_path\
	--data_path coupled_lorenz_noise_0.8.csv \
	--task_name long_term_forecast \
	--data custom \
	--seq_len 96 \
	--pred_len 96 \
	--d_model 64 \
	--tfactor 2 \
	--dfactor 1 \
	--wavelet sym3 \
	--level 1 \
	--patch_len 4 \
	--enc_in 90 \
	--dec_in 90 \
	--c_out 90 \
	--stride 2 \
	--batch_size 128 \
	--learning_rate 0.01 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.1 \
	--patience 5 \
	--train_epochs 30 \
	--use_amp 



	# 0.9噪声
python -u $run_path \
	--is_training 1\
	--model $model_name \
	--model_id coupled_lorenz_noise_0.9_96_96 \
	--root_path $root_path\
	--data_path coupled_lorenz_noise_0.9.csv \
	--task_name long_term_forecast \
	--data custom \
	--seq_len 96 \
	--pred_len 96 \
	--d_model 64 \
	--tfactor 2 \
	--dfactor 1 \
	--wavelet sym3 \
	--level 1 \
	--patch_len 4 \
	--enc_in 90 \
	--dec_in 90 \
	--c_out 90 \
	--stride 2 \
	--batch_size 128 \
	--learning_rate 0.01 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.1 \
	--patience 5 \
	--train_epochs 30 \
	--use_amp 


# 1.0噪声
python -u $run_path \
	--is_training 1\
	--model $model_name \
	--model_id coupled_lorenz_noise_1.0_96_96 \
	--root_path $root_path\
	--data_path coupled_lorenz_noise_1.0.csv \
	--task_name long_term_forecast \
	--data custom \
	--seq_len 96 \
	--pred_len 96 \
	--d_model 64 \
	--tfactor 2 \
	--dfactor 1 \
	--wavelet sym3 \
	--level 1 \
	--patch_len 4 \
	--enc_in 90 \
	--dec_in 90 \
	--c_out 90 \
	--stride 2 \
	--batch_size 128 \
	--learning_rate 0.01 \
	--lradj type3 \
	--dropout 0.4 \
	--embedding_dropout 0.1 \
	--patience 5 \
	--train_epochs 30 \
	--use_amp 