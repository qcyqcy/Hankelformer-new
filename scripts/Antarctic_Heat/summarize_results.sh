#!/bin/bash

# 汇总 Antarctic_Heat 所有实验结果

log_dir="../../logs/Antarctic_Heat"
output_file="$log_dir/summary_results.txt"

echo "========================================" > "$output_file"
echo "Antarctic_Heat 调参结果汇总" >> "$output_file"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$output_file"
echo "========================================" >> "$output_file"
echo "" >> "$output_file"

# 表头
echo "格式: window_size | contrastive_weight | learning_rate | MSE | MAE | best_epoch" >> "$output_file"
echo "--------------------------------------------------------------------------------" >> "$output_file"

# 遍历所有 .log 文件
if [ -d "$log_dir" ]; then
    for log_file in "$log_dir"/*.log; do
        if [ -f "$log_file" ]; then
            filename=$(basename "$log_file" .log)

            # 从文件名提取参数
            # 格式: ws${window_size}_cw${contrastive_weight}_lr${learning_rate}
            ws=$(echo "$filename" | sed -n 's/.*ws\([^_]*\).*/\1/p')
            cw=$(echo "$filename" | sed -n 's/.*cw\([^_]*\).*/\1/p')
            lr=$(echo "$filename" | sed -n 's/.*lr\([^_]*\).*/\1/p')

            # 从日志文件中提取最后的 MSE 和 MAE (或 best 指标)
            # 尝试从最后一行的 metrics 中提取
            last_line=$(tail -20 "$log_file" 2>/dev/null | grep -E "mse|mae" | tail -1)

            if [ -n "$last_line" ]; then
                mse=$(echo "$last_line" | grep -oiE 'mse[[:space:]]*:[[:space:]]*[0-9.]+' | grep -oE '[0-9.]+' | tail -1)
                mae=$(echo "$last_line" | grep -oiE 'mae[[:space:]]*:[[:space:]]*[0-9.]+' | grep -oE '[0-9.]+' | tail -1)
            else
                mse="N/A"
                mae="N/A"
            fi

            # 尝试提取 best epoch
            best_line=$(grep -i "best" "$log_file" 2>/dev/null | tail -1)
            if [ -n "$best_line" ]; then
                best_epoch=$(echo "$best_line" | grep -oE 'epoch[= ]*[0-9]+' | grep -oE '[0-9]+' | tail -1)
            else
                best_epoch="N/A"
            fi

            echo "$ws | $cw | $lr | $mse | $mae | $best_epoch" >> "$output_file"
        fi
    done

    # 按 MSE 排序
    echo "" >> "$output_file"
    echo "========================================" >> "$output_file"
    echo "按 MSE 从低到高排序 (前10名)" >> "$output_file"
    echo "========================================" >> "$output_file"
    grep -E "^[0-9]" "$output_file" | sort -t'|' -k4 -n | head -10 >> "$output_file"

    echo "" >> "$output_file"
    echo "汇总完成！结果保存在: $output_file"
    echo "汇总完成！结果保存在: $output_file"

else
    echo "错误: 目录 $log_dir 不存在"
    exit 1
fi