import pandas as pd
import json
import os
import re
import csv
from config import config

def load_all_group_data_for_model(model_key):
    """
    根据模型键加载CSV决策日志数据。
    """
    if model_key not in config:
        print(f"错误: 模型键 '{model_key}' 在 config.py 中未找到。")
        return pd.DataFrame(), [] # 返回空的DataFrame和列表

    model_config = config[model_key]
    base_path = model_config.get("base_path", "")
    csv_template = model_config.get("csv_template", "")
    num_groups = model_config.get("num_groups", 0)
    
    all_dfs_data = []
    
    for i in range(1, num_groups + 1):
        csv_path = os.path.join(base_path, csv_template.format(group_id=i))
        try:
            with open(csv_path, 'r', encoding='utf-8') as f_csv:
                reader = csv.reader(f_csv) # 使用更简单的默认reader
                for row_num, row in enumerate(reader):
                    # 你的CSV日志有6列，这是我们需要的
                    if len(row) == 6:
                        try:
                            # 提取数据，并进行类型转换
                            current_partner_str = str(row[4]).strip()
                            all_dfs_data.append({
                                'prompt': str(row[0]),
                                'reason': str(row[1]),
                                'target': int(row[2]),
                                'proposer': int(row[3]),
                                'current_partner': int(current_partner_str) if current_partner_str else None,
                                'result': int(row[5]),
                                'group': i
                            })
                        except (ValueError, IndexError):
                            # 跳过无法正确转换的行
                            pass
        except FileNotFoundError:
            pass # 文件不存在是正常情况，如果组数设多了
        except Exception as e:
            print(f"  错误: 处理CSV文件 {csv_path} 时发生未知错误: {e}")

    if not all_dfs_data:
        print(f"警告: 模型 '{model_config.get('label', model_key)}' 没有加载到任何有效的CSV数据。")
        return pd.DataFrame(), [] # 仍然需要返回两个值

    combined_df = pd.DataFrame(all_dfs_data)
    
    # 因为JSON文件在这个分析中不是必需的，我们返回一个空列表作为占位符
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