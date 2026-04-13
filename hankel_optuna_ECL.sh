#!/bin/bash
# hankel_optuna_integrated.sh

export CUDA_VISIBLE_DEVICES=0

# ===== 只需要修改这里的配置 =====
DATASET="ECL"  # 数据集名称
MODEL_NAME="Hankelformer"  # 模型名称
RUN_PATH="./run.py"  # run.py的路径
N_TRIALS=50  # 每个预测长度的试验次数

# ===== 超参数搜索配置 - 选择范围或候选值 =====

# 方式2: 使用候选值搜索（按你的要求设置）
USE_RANGE_SEARCH=false

# 候选值搜索配置
window_sizes=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48)
contrastive_weights=(0.01 0.03 0.05 0.07 0.09 0.1)
learning_rates=(0.0005)

# 其他参数（仍使用候选值）
e_layers_list=(3)
d_models=(512)
d_ffs=(512)

# 选择要调优的预测长度
pred_lens=(96 192 336 720)

# 训练轮数
TRAIN_EPOCHS=20
# ===============================

# 数据集配置
declare -A DATASET_CONFIGS

# ECL配置
DATASET_CONFIGS["ECL,root_path"]="./dataset/electricity"
DATASET_CONFIGS["ECL,data_path"]="electricity.csv"
DATASET_CONFIGS["ECL,data"]="custom"
DATASET_CONFIGS["ECL,enc_in"]="321"
DATASET_CONFIGS["ECL,target"]="OT"

# Traffic配置
DATASET_CONFIGS["Traffic,root_path"]="./dataset/traffic"
DATASET_CONFIGS["Traffic,data_path"]="traffic.csv"
DATASET_CONFIGS["Traffic,data"]="custom"
DATASET_CONFIGS["Traffic,enc_in"]="862"
DATASET_CONFIGS["Traffic,target"]="OT"

# PEMS配置
DATASET_CONFIGS["PEMS03,root_path"]="./dataset/PEMS"
DATASET_CONFIGS["PEMS03,data_path"]="PEMS03.npz"
DATASET_CONFIGS["PEMS03,data"]="PEMS"
DATASET_CONFIGS["PEMS03,enc_in"]="358"
DATASET_CONFIGS["PEMS03,target"]="OT"

DATASET_CONFIGS["PEMS04,root_path"]="./dataset/PEMS"
DATASET_CONFIGS["PEMS04,data_path"]="PEMS04.npz"
DATASET_CONFIGS["PEMS04,data"]="PEMS"
DATASET_CONFIGS["PEMS04,enc_in"]="307"
DATASET_CONFIGS["PEMS04,target"]="OT"

DATASET_CONFIGS["PEMS07,root_path"]="./dataset/PEMS"
DATASET_CONFIGS["PEMS07,data_path"]="PEMS07.npz"
DATASET_CONFIGS["PEMS07,data"]="PEMS"
DATASET_CONFIGS["PEMS07,enc_in"]="883"
DATASET_CONFIGS["PEMS07,target"]="OT"

DATASET_CONFIGS["PEMS08,root_path"]="./dataset/PEMS"
DATASET_CONFIGS["PEMS08,data_path"]="PEMS08.npz"
DATASET_CONFIGS["PEMS08,data"]="PEMS"
DATASET_CONFIGS["PEMS08,enc_in"]="170"
DATASET_CONFIGS["PEMS08,target"]="OT"

# Texas_Freeze配置
DATASET_CONFIGS["Texas_Freeze,root_path"]="./dataset/Texas_Freeze"
DATASET_CONFIGS["Texas_Freeze,data_path"]="Texas_Freeze.csv"
DATASET_CONFIGS["Texas_Freeze,data"]="Texas_Freeze"
DATASET_CONFIGS["Texas_Freeze,enc_in"]="121"
DATASET_CONFIGS["Texas_Freeze,target"]="OT"

# Northwest_Heatwave配置
DATASET_CONFIGS["Northwest_Heatwave,root_path"]="./dataset/Northwest_Heatwave"
DATASET_CONFIGS["Northwest_Heatwave,data_path"]="Northwest_Heatwave.csv"
DATASET_CONFIGS["Northwest_Heatwave,data"]="Northwest_Heatwave"
DATASET_CONFIGS["Northwest_Heatwave,enc_in"]="70"
DATASET_CONFIGS["Northwest_Heatwave,target"]="OT"

# 获取数据集配置
root_path=${DATASET_CONFIGS["$DATASET,root_path"]}
data_path=${DATASET_CONFIGS["$DATASET,data_path"]}
data_name=${DATASET_CONFIGS["$DATASET,data"]}
enc_in=${DATASET_CONFIGS["$DATASET,enc_in"]}
target=${DATASET_CONFIGS["$DATASET,target"]}

# 检查数据集是否存在
if [ -z "$root_path" ]; then
    echo "Error: Unknown dataset '$DATASET'"
    echo "Available datasets: ECL, Traffic, PEMS03, PEMS04, PEMS07, PEMS08, Texas_Freeze, Northwest_Heatwave"
    exit 1
fi

echo "========================================"
echo "HankelFormer Optuna Integration"
echo "========================================"
echo "Dataset: $DATASET"
echo "Model: $MODEL_NAME"
echo "Data path: $root_path/$data_path"
echo "Features: $enc_in"
echo "Pred_lens: ${pred_lens[@]}"
echo "Trials per pred_len: $N_TRIALS"
echo ""

if [ "$USE_RANGE_SEARCH" = true ]; then
    echo "Search mode: RANGE SEARCH (智能连续搜索)"
    echo "  window_size: [$WINDOW_SIZE_MIN, $WINDOW_SIZE_MAX]"
    echo "  contrastive_weight: [$CONTRASTIVE_WEIGHT_MIN, $CONTRASTIVE_WEIGHT_MAX] (log scale)"
    echo "  learning_rate: [$LEARNING_RATE_MIN, $LEARNING_RATE_MAX] (log scale)"
else
    echo "Search mode: CANDIDATE VALUES (候选值搜索)"
    echo "  window_sizes: ${window_sizes[@]}"
    echo "  contrastive_weights: ${contrastive_weights[@]}"
    echo "  learning_rates: ${learning_rates[@]}"
fi

echo "  e_layers: ${e_layers_list[@]}"
echo "  d_models: ${d_models[@]}"
echo "  d_ffs: ${d_ffs[@]}"
echo "========================================"

# 创建配置文件
config_file="hankel_${DATASET}_config.json"

# 根据搜索模式生成不同的配置
if [ "$USE_RANGE_SEARCH" = true ]; then
    # 范围搜索配置
    cat > $config_file << EOF
{
    "dataset": "$DATASET",
    "model_name": "$MODEL_NAME",
    "run_path": "$RUN_PATH",
    "root_path": "$root_path/",
    "data_path": "$data_path",
    "data_name": "$data_name",
    "enc_in": "$enc_in",
    "target": "$target",
    "pred_lens": [$(IFS=,; echo "${pred_lens[*]}")],
    "n_trials": $N_TRIALS,
    "train_epochs": "$TRAIN_EPOCHS",
    "search_space": {
        "window_size_range": [$WINDOW_SIZE_MIN, $WINDOW_SIZE_MAX],
        "contrastive_weight_range": [$CONTRASTIVE_WEIGHT_MIN, $CONTRASTIVE_WEIGHT_MAX],
        "learning_rate_range": [$LEARNING_RATE_MIN, $LEARNING_RATE_MAX],
        "e_layers": [$(IFS=,; echo "${e_layers_list[*]}")],
        "d_models": [$(IFS=,; echo "${d_models[*]}")],
        "d_ffs": [$(IFS=,; echo "${d_ffs[*]}")]
    }
}
EOF
else
    # 候选值搜索配置
    cat > $config_file << EOF
{
    "dataset": "$DATASET",
    "model_name": "$MODEL_NAME",
    "run_path": "$RUN_PATH",
    "root_path": "$root_path/",
    "data_path": "$data_path",
    "data_name": "$data_name",
    "enc_in": "$enc_in",
    "target": "$target",
    "pred_lens": [$(IFS=,; echo "${pred_lens[*]}")],
    "n_trials": $N_TRIALS,
    "train_epochs": "$TRAIN_EPOCHS",
    "search_space": {
        "window_sizes": [$(IFS=,; echo "${window_sizes[*]}")],
        "contrastive_weights": [$(IFS=,; echo "${contrastive_weights[*]}")],
        "learning_rates": [$(IFS=,; echo "${learning_rates[*]}")],
        "e_layers": [$(IFS=,; echo "${e_layers_list[*]}")],
        "d_models": [$(IFS=,; echo "${d_models[*]}")],
        "d_ffs": [$(IFS=,; echo "${d_ffs[*]}")]
    }
}
EOF
fi

echo "Configuration saved to: $config_file"

read -p "Start Optuna optimization? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# 运行Optuna优化
echo "Starting Optuna optimization..."
if [ "$USE_RANGE_SEARCH" = true ]; then
    echo "Using intelligent RANGE search!"
else
    echo "Using CANDIDATE VALUE search!"
fi

python optuna_runner_ECL.py $config_file

echo ""
echo "========================================"
echo "Optuna optimization completed!"
echo "Results saved as:"
echo "  - optuna_${DATASET}_pl*_results.csv"
echo "  - optuna_${DATASET}_summary.json"
echo "  - hankel_${DATASET}_pl*.db (Optuna database)"
echo "========================================"

# 清理配置文件（可选）
# rm $config_file