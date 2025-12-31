# 文件名: load_data.py

import pandas as pd
import json
import os
import re
import csv
from config import config

def load_matchings_from_json(model_key):
    """
    【新增函数】根据模型配置加载 JSON 匹配结果。
    
    注意：
    如果 config 中配置了多轮 (num_runs > 1) 且模板包含 {run_id}，
    此函数会将所有 Group 的所有 Run 展平成一个列表返回。
    列表顺序为: [G1_R1, G1_R2... G1_Rn, G2_R1...]
    """
    if model_key not in config:
        print(f"错误: 模型键 '{model_key}' 在 config.py 中未找到。")
        return []

    model_config = config[model_key]
    base_path = model_config.get("base_path", "")
    json_template = model_config.get("json_template", "")
    num_groups = model_config.get("num_groups", 0)
    num_runs = model_config.get("num_runs", 1)  # 默认为1，兼容旧配置

    all_matchings = []
    
    # 检查模板是否支持多轮
    is_multi_run = "{run_id}" in json_template

    print(f"正在加载 JSON 数据: {model_config.get('label', model_key)} (Total Groups: {num_groups})...")

    for g_id in range(1, num_groups + 1):
        # 确定该组要跑多少轮
        # 如果模板不支持 run_id，强制只跑1轮
        current_runs = num_runs if is_multi_run else 1
        
        for r_id in range(1, current_runs + 1):
            # 格式化文件名
            if is_multi_run:
                filename = json_template.format(group_id=g_id, run_id=r_id)
            else:
                filename = json_template.format(group_id=g_id)
            
            file_path = os.path.join(base_path, filename)

            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 简单的完整性检查：确保读取的是字典
                        if isinstance(data, dict):
                            all_matchings.append(data)
                        else:
                            # 如果读出来不是字典（比如是空的），存一个空字典占位，防止索引错乱
                            all_matchings.append({})
                except Exception as e:
                    print(f"  无法读取文件 {filename}: {e}")
                    all_matchings.append({}) # 同样存空字典占位
            else:
                # 文件不存在时（比如只跑了部分组），是否占位取决于你的分析代码逻辑
                # 如果你的分析代码是用 enumerate(list, start=1) 来对应 group_id
                # 那么这里必须 append 一个空字典 {} 来保持索引对齐
                # 但如果是多轮次数据，索引对齐逻辑本身就会失效（见下文警告）
                pass

    return all_matchings

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
    num_runs = model_config.get("num_runs", 1)
    
    all_dfs_data = []
    
    # 外层循环 Group
    for g_id in range(1, num_groups + 1):
        # 内层循环 Run
        for r_id in range(1, num_runs + 1):
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
    加载原始的Excel评分数据，并处理成方便查询的格式。
    """
    try:
        df = pd.read_excel(path)
        
        score_cols = [
            'attractive_partner', 
            'sincere_partner', 
            'intelligence_partner', 
            'funny_partner', 
            'ambition_partner', 
            'shared_interests_partner'
        ]
        
        # 修正可能的列名拼写错误
        if 'intelliger' in df.columns and 'intelligence' not in df.columns:
            df.rename(columns={'intelliger': 'intelligence'}, inplace=True)

        # 检查必要列
        required_cols = ['iid', 'pid'] + score_cols
        for col in required_cols:
            if col not in df.columns:
                print(f"错误: 源数据Excel文件中缺少必需的列: '{col}'")
                return None

        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df[score_cols] = df[score_cols].fillna(0)
        df['total_score'] = df[score_cols].sum(axis=1)
        
        score_dict = df.set_index(['iid', 'pid'])['total_score'].to_dict()
        
        # print("源数据Excel评分已成功加载。") 
        return score_dict
        
    except FileNotFoundError:
        print(f"错误: 源数据文件未找到: {path}")
        return None
    except Exception as e:
        print(f"加载源数据Excel文件时发生未知错误: {e}")
        return None