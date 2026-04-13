#!/bin/bash
# hankel_optuna_italy.sh

export CUDA_VISIBLE_DEVICES=1

# ===== 只需要修改这里的配置 =====
DATASET="Italy"  # 数据集名称
MODEL_NAME="Hankelformer"  # 模型名称
RUN_PATH="/share/home/qinchengyang/Time-Series-Library/run.py"  # run.py的路径
N_TRIALS=150  # 每个预测长度的试验次数

# ===== 超参数搜索配置 - 混合搜索模式 =====

# 混合搜索模式：部分参数用范围，部分参数用候选值
USE_MIXED_SEARCH=true

# 候选值搜索配置
window_sizes=(1 2 3 4 5 6 7 8 9 10 11 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48)

# 范围搜索配置（基于你的候选值范围）
CONTRASTIVE_WEIGHT_MIN=0.0001
CONTRASTIVE_WEIGHT_MAX=0.05
LEARNING_RATE_MIN=0.00005
LEARNING_RATE_MAX=0.001

# 其他参数（仍使用候选值）
e_layers_list=(2 3 4)
d_models=(512)
d_ffs=(512)

# 选择要调优的预测长度
pred_lens=(48)

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


# Italy配置（新增）
DATASET_CONFIGS["Italy,root_path"]="/share/home/qinchengyang/Time-Series-Library/dataset/italy_HEATWAVE"
DATASET_CONFIGS["Italy,data_path"]="italy_temperature_data.csv"
DATASET_CONFIGS["Italy,data"]="custom_weather"
DATASET_CONFIGS["Italy,enc_in"]="214"
DATASET_CONFIGS["Italy,target"]="OT"


# 获取数据集配置
root_path=${DATASET_CONFIGS["$DATASET,root_path"]}
data_path=${DATASET_CONFIGS["$DATASET,data_path"]}
data_name=${DATASET_CONFIGS["$DATASET,data"]}
enc_in=${DATASET_CONFIGS["$DATASET,enc_in"]}
target=${DATASET_CONFIGS["$DATASET,target"]}

# 检查数据集是否存在
if [ -z "$root_path" ]; then
    echo "Error: Unknown dataset '$DATASET'"
    echo "Available datasets: ECL, Traffic, Italy"
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

if [ "$USE_MIXED_SEARCH" = true ]; then
    echo "Search mode: MIXED SEARCH (混合搜索)"
    echo "  window_sizes (candidates): ${window_sizes[@]}"
    echo "  contrastive_weight (range): [$CONTRASTIVE_WEIGHT_MIN, $CONTRASTIVE_WEIGHT_MAX] (log scale)"
    echo "  learning_rate (range): [$LEARNING_RATE_MIN, $LEARNING_RATE_MAX] (log scale)"
    echo "  e_layers (candidates): ${e_layers_list[@]}"
    echo "  d_models (candidates): ${d_models[@]}"
    echo "  d_ffs (candidates): ${d_ffs[@]}"
else
    echo "Search mode: CANDIDATE VALUES (候选值搜索)"
    echo "  window_sizes: ${window_sizes[@]}"
    echo "  e_layers: ${e_layers_list[@]}"
    echo "  d_models: ${d_models[@]}"
    echo "  d_ffs: ${d_ffs[@]}"
fi

echo "========================================"

# 创建配置文件
config_file="hankel_${DATASET}_config.json"

# 生成混合搜索配置
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
        "contrastive_weight_range": [$CONTRASTIVE_WEIGHT_MIN, $CONTRASTIVE_WEIGHT_MAX],
        "learning_rate_range": [$LEARNING_RATE_MIN, $LEARNING_RATE_MAX],
        "e_layers": [$(IFS=,; echo "${e_layers_list[*]}")],
        "d_models": [$(IFS=,; echo "${d_models[*]}")],
        "d_ffs": [$(IFS=,; echo "${d_ffs[*]}")]
    }
}
EOF

echo "Configuration saved to: $config_file"

read -p "Start Optuna optimization? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# 运行Optuna优化
echo "Starting Optuna optimization..."
echo "Using MIXED search mode!"
echo "  - window_sizes: candidate values"
echo "  - contrastive_weight: range search (log scale)"
echo "  - learning_rate: range search (log scale)" 
echo "  - e_layers, d_models, d_ffs: candidate values"

python optuna_runner_italy.py $config_file

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