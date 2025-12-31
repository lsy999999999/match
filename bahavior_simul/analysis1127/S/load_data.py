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
            if "{run_id}" in csv_template:
                filename = csv_template.format(group_id=g_id, run_id=r_id)
            else:
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
                                    'run': r_id
                                })
                            except (ValueError, IndexError):
                                pass
            except FileNotFoundError:
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
    加载原始的Excel评分数据。
    
    【核心逻辑更新】：
    使用动态归一化计算加权分，投射到 100 分制。
    
    公式: Weighted Score = (Sum(Score * Weight) / Sum(Weight)) * 10.0
    
    优势:
    - 不再假设权重之和固定为 100。
    - 即使只有 6 个指标，也能正确计算出该维度的平均得分水平，并映射到 0-100。
    """
    try:
        df = pd.read_excel(path)
        
        # 定义维度列表
        dims = ['attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'shared_interests']
        
        # 1. 修正列名
        if 'intelliger' in df.columns and 'intelligence' not in df.columns:
            df.rename(columns={'intelliger': 'intelligence'}, inplace=True)

        # 2. 定义分数列 (*_partner) 和权重列 (*_important)
        score_cols = [f'{d}_partner' for d in dims]
        imp_cols = [f'{d}_important' for d in dims]

        # 3. 检查列是否存在
        required_cols = ['iid', 'pid'] + score_cols + imp_cols
        for col in required_cols:
            if col not in df.columns:
                print(f"错误: 源数据Excel文件中缺少必需的列: '{col}'")
                return None

        # 4. 数据清洗：转数值 & 填充0
        for col in score_cols + imp_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 5. 计算分子（加权总分）和分母（总权重）
        weighted_sum = pd.Series(0.0, index=df.index)
        total_weight = pd.Series(0.0, index=df.index)

        for i, dim in enumerate(dims):
            s_col = score_cols[i]
            w_col = imp_cols[i]
            weighted_sum += df[s_col] * df[w_col]
            total_weight += df[w_col]
            
        # 6. 动态归一化并投射到 100 分
        # (加权总分 / 总权重) = 0-10 分制的加权平均分
        # 再 * 10 = 0-100 分制
        # 使用 fillna(0) 处理总权重为 0 的情况
        df['weighted_score'] = (weighted_sum / total_weight) * 10.0
        df['weighted_score'] = df['weighted_score'].fillna(0)
        
        # 7. 构造字典 {(iid, pid): weighted_score}
        score_dict = df.set_index(['iid', 'pid'])['weighted_score'].to_dict()
        
        print(f"源数据加载成功。模式: Dynamic Weighted Score . 样本数: {len(score_dict)}")
        
        if len(score_dict) > 0:
            first_key = list(score_dict.keys())[0]
            print(f"  样本检查: {first_key} -> {score_dict[first_key]:.2f}")
            
        return score_dict
        
    except FileNotFoundError:
        print(f"错误: 源数据文件未找到: {path}")
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"加载源数据Excel文件时发生未知错误: {e}")
        return None