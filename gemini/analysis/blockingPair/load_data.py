# 文件名: load_data.py (完整、统一、健壮的最终版本)

import pandas as pd
import json
import os
import re
import csv
from config import config

def load_all_group_data_for_model(model_key):
    """
    根据模型键加载所有组的CSV决策日志和JSON最终匹配结果。
    这是一个统一的加载器，适用于你项目中的所有分析脚本。
    """
    if model_key not in config:
        print(f"错误: 模型键 '{model_key}' 在 config.py 中未找到。")
        return pd.DataFrame(), []

    model_config = config[model_key]
    base_path = model_config.get("base_path", "")
    csv_template = model_config.get("csv_template", "")
    json_template = model_config.get("json_template", "")
    num_groups = model_config.get("num_groups", 0)
    model_label = model_config.get("label", model_key)

    all_dfs_data = []
    all_matchings = []
    
    print(f"--- 正在加载模型: {model_label} (预期 {num_groups} 组) ---")

    for i in range(1, num_groups + 1):
        # --- 1. 加载和解析CSV文件 ---
        csv_path = os.path.join(base_path, csv_template.format(group_id=i))
        try:
            with open(csv_path, 'r', encoding='utf-8') as f_csv:
                reader = csv.reader(f_csv)
                for row_num, row in enumerate(reader):
                    if len(row) == 6: # 严格要求6列
                        try:
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
                            pass # 跳过无法转换的行
        except FileNotFoundError:
            pass # 正常现象，如果组数设多了
        except Exception as e:
            print(f"  错误: 处理CSV文件 {csv_path} 时发生未知错误: {e}")

        # --- 2. 加载和解析JSON文件 ---
        json_path = os.path.join(base_path, json_template.format(group_id=i))
        try:
            with open(json_path, 'r', encoding='utf-8') as f_json:
                file_content = f_json.read().strip()
                if not file_content:
                    continue
                
                # 正则表达式查找所有 {...} 结构，这是处理损坏JSON的关键
                json_objects = re.findall(r'\{[^{}]*\}', file_content)
                
                parsed_data = None
                if not json_objects:
                    # 尝试将整个文件作为单个JSON加载
                    try:
                        parsed_data = json.loads(file_content)
                    except json.JSONDecodeError:
                        pass
                elif len(json_objects) == 1:
                    try:
                        parsed_data = json.loads(json_objects[0])
                    except json.JSONDecodeError:
                        pass
                else: # 找到多个对象
                    try:
                        json_array_string = f"[{','.join(json_objects)}]"
                        data_list = json.loads(json_array_string)
                        if data_list:
                            parsed_data = data_list[-1] # 取最后一个作为结果
                    except json.JSONDecodeError:
                        pass
                
                if parsed_data is not None:
                    all_matchings.append(parsed_data)

        except FileNotFoundError:
            pass
        except Exception as e_json:
            print(f"  错误: 处理JSON文件 {json_path} 时发生未知错误: {e_json}")
            
    # --- 3. 整合和返回数据 ---
    if all_dfs_data:
        combined_df = pd.DataFrame(all_dfs_data)
    else:
        combined_df = pd.DataFrame()
        print(f"警告: 模型 '{model_label}' 没有加载到任何有效的CSV数据。")

    if not all_matchings and num_groups > 0:
        print(f"警告: 模型 '{model_label}' 没有加载到任何有效的JSON数据。")
        
    print(f"--- 模型 {model_label}: 加载完成。CSV行数: {len(combined_df)}, 成功加载的JSON组数: {len(all_matchings)} ---")
    
    return combined_df, all_matchings



def load_matchings_from_json(model_key):
    """
    【新函数】专门加载一个模型所有组的JSON最终匹配结果。
    """
    if model_key not in config:
        print(f"错误: 模型键 '{model_key}' 在 config.py 中未找到。")
        return []

    model_config = config[model_key]
    base_path = model_config.get("base_path", "")
    json_template = model_config.get("json_template", "")
    num_groups = model_config.get("num_groups", 0)
    model_label = model_config.get("label", model_key)
    
    all_matchings = []
    print(f"--- 正在为 '{model_label}' 加载JSON匹配结果 ---")

    for i in range(1, num_groups + 1):
        json_path = os.path.join(base_path, json_template.format(group_id=i))
        try:
            with open(json_path, 'r', encoding='utf-8') as f_json:
                file_content = f_json.read().strip()
                if not file_content: continue
                
                json_objects = re.findall(r'\{[^{}]*\}', file_content)
                parsed_data = None
                if not json_objects:
                    try: parsed_data = json.loads(file_content)
                    except json.JSONDecodeError: pass
                elif len(json_objects) == 1:
                    try: parsed_data = json.loads(json_objects[0])
                    except json.JSONDecodeError: pass
                else:
                    try:
                        json_array_string = f"[{','.join(json_objects)}]"
                        data_list = json.loads(json_array_string)
                        if data_list: parsed_data = data_list[-1]
                    except json.JSONDecodeError: pass
                
                if parsed_data is not None:
                    # 关键：将键和值都转为整数
                    cleaned_matching = {
                        int(k): (int(v) if str(v).isdigit() else v) 
                        for k, v in parsed_data.items()
                    }
                    all_matchings.append(cleaned_matching)
        except FileNotFoundError:
            # print(f"  提示: JSON文件不存在 {json_path}")
            pass
        except Exception as e:
            print(f"  错误: 处理JSON文件 {json_path} 时发生未知错误: {e}")

    print(f"--- 为 '{model_label}' 成功加载了 {len(all_matchings)} 个JSON匹配结果。 ---")
    return all_matchings

def load_source_scores(path):
    """加载原始Excel评分数据，计算并返回每个评价的总分。"""
    # (此函数保持不变)
    try:
        df = pd.read_excel(path)
        # 根据你之前的代码，列名应为 '..._partner'
        score_cols = [f'{d}_partner' for d in ['attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'shared_interests']]
        required_cols = ['iid', 'pid'] + score_cols
        for col in required_cols:
            if col not in df.columns: return None
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[score_cols] = df[score_cols].fillna(0)
        df['total_score'] = df[score_cols].sum(axis=1)
        score_dict = df.set_index(['iid', 'pid'])['total_score'].to_dict()
        print("源数据Excel评分 (总分) 已成功加载并处理。")
        return score_dict
    except Exception as e:
        print(f"加载源数据Excel文件时出错: {e}")
        return None

# def load_source_scores(path):
#     """
#     加载原始Excel评分数据，计算并返回每个评价的总分。
#     """
#     try:
#         df = pd.read_excel(path)
#         score_cols = ['attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'shared_interests']
        
#         # 修正可能的拼写错误 (根据你之前截图的信息)
#         if 'intelliger' in df.columns and 'intelligence' not in df.columns:
#             df.rename(columns={'intelliger': 'intelligence'}, inplace=True)
#         if 'ambition_partner' in df.columns and 'ambition' in df.columns: # 处理可能的多种列名
#              pass # 假设'ambition'是正确的
        

#         # 列名应为 '..._partner'
#         partner_score_cols = [f'{col}_partner' for col in score_cols]
#         if all(col in df.columns for col in partner_score_cols):
#              score_cols = partner_score_cols
#              print("注意: 正在使用 '_partner' 后缀的评分列。")
        
#         required_cols = ['iid', 'pid'] + score_cols
#         for col in required_cols:
#             if col not in df.columns:
#                 print(f"错误: 源数据Excel文件中缺少必需的列: '{col}'")
#                 return None
        
#         for col in score_cols:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#         df[score_cols] = df[score_cols].fillna(0)
#         df['total_score'] = df[score_cols].sum(axis=1)
        
#         score_dict = df.set_index(['iid', 'pid'])['total_score'].to_dict()
#         print("源数据Excel评分 (总分) 已成功加载并处理。")
#         return score_dict
        
#     except FileNotFoundError:
#         print(f"错误: 源数据文件未找到: {path}")
#         return None
#     except Exception as e:
#         print(f"加载源数据Excel文件时发生未知错误: {e}")
#         return None

def load_source_scores_vectorized(path):
    """
    加载原始Excel评分数据，返回每个评价的六维分数向量。
    """
    try:
        df = pd.read_excel(path)
        score_cols = ['attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'shared_interests']

        if 'intelliger' in df.columns: df.rename(columns={'intelliger': 'intelligence'}, inplace=True)
        partner_score_cols = [f'{col}_partner' for col in score_cols]
        if all(col in df.columns for col in partner_score_cols):
             score_cols = partner_score_cols
             print("注意: 正在使用 '_partner' 后缀的评分列。")
        
        required_cols = ['iid', 'pid'] + score_cols
        for col in required_cols:
            if col not in df.columns:
                print(f"错误: 源数据Excel文件中缺少必需的列: '{col}'")
                return None

        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[score_cols] = df[score_cols].fillna(0)
        
        score_dict = df.set_index(['iid', 'pid'])[score_cols].apply(list, axis=1).to_dict()
        print("源数据Excel评分 (向量化版本) 已成功加载并处理。")
        return score_dict, score_cols
        
    except FileNotFoundError:
        print(f"错误: 源数据文件未找到: {path}")
        return None, None
    except Exception as e:
        print(f"加载源数据Excel文件时发生未知错误: {e}")
        return None, None
    


def load_gender_map(path):
    """从源Excel加载所有参与者的性别映射。"""
    try:
        df = pd.read_excel(path)
        # iid是唯一的参与者ID
        gender_df = df[['iid', 'gender']].drop_duplicates().set_index('iid')
        # gender=1是男性，gender=0是女性
        gender_map = gender_df['gender'].to_dict()
        print("参与者性别映射已成功加载。")
        return gender_map
    except Exception as e:
        print(f"加载性别映射时出错: {e}")
        return {}