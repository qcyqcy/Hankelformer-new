# optuna_runner.py (修复MKL问题版本)
import optuna
import subprocess
import sys
import os
import re
import json
import tempfile
import random
import numpy as np


# ===== 设置全局种子 =====
def set_global_seed(seed=2023):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# 在导入后立即设置种子
set_global_seed(2023)
print("Global seed set to 2023")
# ========================


def run_experiment(params, config):
    """运行单次实验"""
    cmd = [
        "python", "-u", config['run_path'],
        "--task_name", "long_term_forecast_contra",
        "--is_training", "1",
        "--root_path", config['root_path'],
        "--data_path", config['data_path'],
        "--model_id", f"{config['dataset']}_{params['pred_len']}_trial_{params.get('trial_id', 'test')}",
        "--model", config['model_name'],
        "--data", config['data_name'],
        "--features", "M",
        "--target", config['target'],
        "--seq_len", "96",
        "--label_len", "48", 
        "--pred_len", str(params['pred_len']),
        "--e_layers", str(params['e_layers']),
        "--enc_in", config['enc_in'],
        "--dec_in", config['enc_in'],
        "--c_out", config['enc_in'],
        "--des", "Exp",
        "--d_model", str(params['d_model']),
        "--d_ff", str(params['d_ff']),
        "--batch_size", "16",
        "--itr", "1",
        "--window_size", str(params['window_size']),
        "--contrastive_weight", str(params['contrastive_weight']),
        "--learning_rate", str(params['learning_rate']),
        "--train_epochs", config.get('train_epochs', '10'),
        "--patience", "3",
        "--use_norm", "1"
    ]
    
    print(f"Running trial {params.get('trial_id', 'test')} with params: ws={params['window_size']}, cw={params['contrastive_weight']:.4f}, lr={params['learning_rate']:.6f}")
    
    # 创建临时文件保存输出
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as temp_file:
        temp_filename = temp_file.name
    
    try:
        # 设置环境变量解决MKL冲突问题
        env = os.environ.copy()
        env['MKL_SERVICE_FORCE_INTEL'] = '1'
        env['MKL_THREADING_LAYER'] = 'GNU'
        env['OMP_NUM_THREADS'] = '1'
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        # 使用文件重定向而不是capture_output
        with open(temp_filename, 'w') as f:
            result = subprocess.run(
                cmd, 
                stdout=f, 
                stderr=subprocess.STDOUT,  # 将stderr也重定向到stdout
                timeout=7200,  # 2小时超时
                env=env  # 使用修改后的环境变量
            )
        
        # 读取输出文件
        with open(temp_filename, 'r') as f:
            output = f.read()
        
        print(f"Training completed with return code: {result.returncode}")
        
        # 如果返回码不是0，说明训练失败
        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            print("Last 10 lines of output:")
            output_lines = output.split('\n')
            for line in output_lines[-10:]:
                if line.strip():
                    print(f"  {line}")
            return float('inf')
        
        # 解析输出获取MSE和MAE
        output_lines = output.split('\n')
        mse, mae = None, None
        
        # 查找包含mse和mae的行
        for line in output_lines:
            if 'mse:' in line and 'mae:' in line:
                mse_match = re.search(r'mse:([0-9.]+)', line)
                mae_match = re.search(r'mae:([0-9.]+)', line)
                if mse_match and mae_match:
                    mse = float(mse_match.group(1))
                    mae = float(mae_match.group(1))
                    break
        
        if mse is None:
            print(f"Failed to parse results. Checking last 20 lines of output:")
            for line in output_lines[-20:]:
                if line.strip():  # 只打印非空行
                    print(f"  {line}")
            
            # 查找任何包含数字的可能的结果行
            print("Searching for any lines with 'mse' or 'mae':")
            for line in output_lines:
                if 'mse' in line.lower() or 'mae' in line.lower():
                    print(f"  Found: {line}")
            
            return float('inf')
            
        print(f"Trial {params.get('trial_id', 'test')}: MSE={mse:.6f}, MAE={mae:.6f}")
        return mse
        
    except subprocess.TimeoutExpired:
        print(f"Trial {params.get('trial_id', 'test')} timed out after 2 hours")
        return float('inf')
    except Exception as e:
        print(f"Trial {params.get('trial_id', 'test')} failed: {e}")
        return float('inf')
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_filename)
        except:
            pass

def objective(trial, config, pred_len):
    """Optuna目标函数"""
    search_space = config['search_space']
    
    params = {
        'trial_id': trial.number,
        'pred_len': pred_len,
    }
    
    # ===== 支持范围和候选值两种方式 =====
    
    # window_size: 支持范围 [min, max] 或候选值列表
    if 'window_size_range' in search_space:
        params['window_size'] = trial.suggest_int(
            'window_size', 
            search_space['window_size_range'][0], 
            search_space['window_size_range'][1]
        )
    else:
        params['window_size'] = trial.suggest_categorical('window_size', search_space['window_sizes'])
    
    # contrastive_weight: 支持范围 [min, max] 或候选值列表
    if 'contrastive_weight_range' in search_space:
        params['contrastive_weight'] = trial.suggest_float(
            'contrastive_weight',
            search_space['contrastive_weight_range'][0],
            search_space['contrastive_weight_range'][1],
            log=True  # 对数搜索，适合权重参数
        )
    else:
        params['contrastive_weight'] = trial.suggest_categorical('contrastive_weight', search_space['contrastive_weights'])
    
    # learning_rate: 支持范围 [min, max] 或候选值列表
    if 'learning_rate_range' in search_space:
        params['learning_rate'] = trial.suggest_float(
            'learning_rate',
            search_space['learning_rate_range'][0],
            search_space['learning_rate_range'][1],
            log=True  # 对数搜索，适合学习率
        )
    else:
        params['learning_rate'] = trial.suggest_categorical('learning_rate', search_space['learning_rates'])
    
    # 其他参数保持候选值方式
    params['e_layers'] = trial.suggest_categorical('e_layers', search_space['e_layers'])
    params['d_model'] = trial.suggest_categorical('d_model', search_space['d_models'])
    params['d_ff'] = trial.suggest_categorical('d_ff', search_space['d_ffs'])
    
    mse = run_experiment(params, config)
    return mse

def main():
    if len(sys.argv) != 2:
        print("Usage: python optuna_runner.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # 读取配置
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    dataset = config['dataset']
    pred_lens = config['pred_lens']
    n_trials = config['n_trials']
    
    print(f"Starting Optuna optimization for {dataset}")
    print(f"Prediction lengths: {pred_lens}")
    print(f"Trials per pred_len: {n_trials}")
    
    results = {}
    
    # 对每个预测长度进行优化
    for pred_len in pred_lens:
        print(f"\n{'='*50}")
        print(f"Optimizing pred_len = {pred_len}")
        print(f"{'='*50}")
        
        study_name = f"hankel_{dataset}_pl{pred_len}"
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=f'sqlite:///{study_name}.db',
            load_if_exists=True
        )
        
        study.optimize(
            lambda trial: objective(trial, config, pred_len),
            n_trials=n_trials
        )
        
        # 保存结果
        if study.best_trial is not None and study.best_trial.value != float('inf'):
            best_params = study.best_trial.params
            best_value = study.best_trial.value
            
            results[pred_len] = {
                'best_value': best_value,
                'best_params': best_params
            }
            
            print(f"\nBest result for pred_len {pred_len}:")
            print(f"  MSE: {best_value:.6f}")
            print(f"  Params: {best_params}")
        else:
            print(f"No successful trials for pred_len {pred_len}")
        
        # 保存详细结果
        import pandas as pd
        df = study.trials_dataframe()
        df.to_csv(f'optuna_{dataset}_pl{pred_len}_results.csv', index=False)
    
    # 保存汇总
    with open(f'optuna_{dataset}_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*50}")
    for pred_len, result in results.items():
        if result:
            print(f"pred_len {pred_len}: MSE = {result['best_value']:.6f}")
        else:
            print(f"pred_len {pred_len}: No successful trials")

if __name__ == '__main__':
    main()