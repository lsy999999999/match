import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob  # 用于查找匹配模式的文件
import os

# --- 1. 配置区 ---
# 设置你的基础路径
base_path_en = '/home/lsy/match/bahavior_simul/0618_gpt4_turbo_random'
base_path_zh = '/home/lsy/match/bahavior_simul/0619_gpt4_turbo_Chinese'
num_groups = 50

# 设置Matplotlib以正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_context("talk") # 使用更大的字体和更粗的线条，适合报告和展示

# --- 2. 数据加载函数 ---
def load_all_group_data(base_path, lang_prefix, num_groups):
    """加载所有组的CSV和JSON数据"""
    all_dfs = []
    all_matchings = []
    
    for i in range(1, num_groups + 1):
        # 构造文件名，注意中文json文件名的特殊性
        if lang_prefix == 'zh':
            csv_path = os.path.join(base_path, f'0619_gpt4_turbo_random_group{i}.csv')
            json_path = os.path.join(base_path, f'0619_gpt4_turbo_random_group_Chinese{i}.json')
        else:
            csv_path = os.path.join(base_path, f'0618_gpt4_turbo_random_group{i}.csv')
            json_path = os.path.join(base_path, f'0618_gpt4_turbo_random_group{i}.json')

        # 加载CSV
        try:
            df = pd.read_csv(csv_path, header=None)
            # 根据你的案例，倒数第4列是被求偶者(target)，第3列是求偶者(proposer)，第2列是现任(current_partner)，最后1列是结果(result)
            num_cols = len(df.columns)
            df.rename(columns={
                num_cols-4: 'target',
                num_cols-3: 'proposer',
                num_cols-2: 'current_partner',
                num_cols-1: 'result'
            }, inplace=True)
            df['group'] = i  # 添加组号，方便后续分析
            all_dfs.append(df)
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"警告: 无法加载或文件为空 {csv_path} - {e}")

        # 加载JSON
        try:
            with open(json_path, 'r') as f:
                matching_data = json.load(f)
                all_matchings.append(matching_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"警告: 无法加载或文件格式错误 {json_path} - {e}")

    # 合并所有组的DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df, all_matchings

# 加载英文和中文数据
df_en, matchings_en = load_all_group_data(base_path_en, 'en', num_groups)
df_zh, matchings_zh = load_all_group_data(base_path_zh, 'zh', num_groups)

print(f"成功加载英文数据: {len(df_en)}行决策日志, {len(matchings_en)}组最终配对结果。")
print(f"成功加载中文数据: {len(df_zh)}行决策日志, {len(matchings_zh)}组最终配对结果。")