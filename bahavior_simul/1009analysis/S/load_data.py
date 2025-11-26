# 文件名: load_data.py

import pandas as pd
import json
import os
import re
import csv
from config import config

def load_all_group_data_for_model(model_key):
    """
    根据模型键加载CSV决策日志数据。支持多 run 结构。
    """
    if model_key not in config:
        print(f"错误: 模型键 '{model_key}' 在 config.py 中未找到。")
        return pd.DataFrame(), []

    model_config = config[model_key]
    base_path = model_config.get("base_path", "")
    csv_template = model_config.get("csv_template", "")
    num_groups = model_config.get("num_groups", 0)
    # 获取 run 的数量，默认为 1 (兼容旧代码)
    num_runs = model_config.get("num_runs", 1)
    
    all_dfs_data = []
    
    # 外层循环 Group
    for g_id in range(1, num_groups + 1):
        # 内层循环 Run
        for r_id in range(1, num_runs + 1):
            # 根据模板格式化路径
            # 检查模板是否包含 run_id，兼容旧的只有 group_id 的模板
            if "{run_id}" in csv_template:
                filename = csv_template.format(group_id=g_id, run_id=r_id)
            else:
                # 如果是旧模板，只循环一次 run (或者你可以在这里加逻辑)
                if r_id > 1: continue 
                filename = csv_template.format(group_id=g_id)

            csv_path = os.path.join(base_path, filename)
            
            try:
                with open(csv_path, 'r', encoding='utf-8') as f_csv:
                    reader = csv.reader(f_csv)
                    for row in reader:
                        if len(row) == 6:
                            try:
                                current_partner_str = str(row[4]).strip()
                                all_dfs_data.append({
                                    'prompt': str(row[0]),
                                    'reason': str(row[1]),
                                    'target': int(row[2]),
                                    'proposer': int(row[3]),
                                    'current_partner': int(current_partner_str) if current_partner_str else None,
                                    'result': int(row[5]),
                                    'group': g_id,
                                    'run': r_id  # 记录是第几次 run
                                })
                            except (ValueError, IndexError):
                                pass
            except FileNotFoundError:
                # 很多时候只跑了部分组，文件找不到不报错，静默跳过
                pass 
            except Exception as e:
                print(f"  错误: 处理文件 {csv_path} 时发生错误: {e}")

    if not all_dfs_data:
        print(f"警告: 模型 '{model_config.get('label', model_key)}' 没有加载到数据。")
        return pd.DataFrame(), []

    combined_df = pd.DataFrame(all_dfs_data)
    return combined_df, []


def load_source_scores(path):
    """
    加载原始的Excel评分数据，并处理成方便查询的格式。
    使用你截图中的实际列名。
    """
    try:
        df = pd.read_excel(path)
        
        # 【关键修正】使用你Excel截图中实际存在的列名
        # 这些是评价者(iid)对被评价者(pid)的六个维度的打分
        score_cols = [
            'attractive_partner', 
            'sincere_partner', 
            'intelligence_partner', # 注意Excel中列名可能是 'intelliger'
            'funny_partner', 
            'ambition_partner', 
            'shared_interests_partner'
        ]
        
        # 检查并修正可能的列名拼写错误
        if 'intelliger' in df.columns and 'intelligence' not in df.columns:
            df.rename(columns={'intelliger': 'intelligence'}, inplace=True)
            print("注意: 已将列名 'intelliger' 重命名为 'intelligence'。")

        # 确保所有需要的列都存在
        required_cols = ['iid', 'pid'] + score_cols
        for col in required_cols:
            if col not in df.columns:
                print(f"错误: 源数据Excel文件中缺少必需的列: '{col}'")
                return None

        # 将评分列转为数值类型，无法转换的设为NaN
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 用0填充NaN值，或者你可以选择其他策略（如均值填充）
        df[score_cols] = df[score_cols].fillna(0)
        
        # 计算每个评价的总分 (S)
        df['total_score'] = df[score_cols].sum(axis=1)
        
        # 创建一个方便查询的字典: {(评价者ID, 被评价者ID): 总分}
        score_dict = df.set_index(['iid', 'pid'])['total_score'].to_dict()
        
        print("源数据Excel评分已成功加载并处理。")
        return score_dict
        
    except FileNotFoundError:
        print(f"错误: 源数据文件未找到: {path}")
        return None
    except Exception as e:
        print(f"加载源数据Excel文件时发生未知错误: {e}")
        return None