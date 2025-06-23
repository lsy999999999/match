# 文件名: load_data.py (支持多模型配置版)

import pandas as pd
import json
import os
import re
import csv
from config import config # 导入共享配置

def load_all_group_data_for_model(model_key):
    """根据模型键值 (如 'gpt4_en', 'deepseek_en') 加载所有组的数据"""
    if model_key not in config:
        raise ValueError(f"未在config.py中找到模型配置: {model_key}")

    model_config = config[model_key]
    base_path = model_config["base_path"]
    csv_template = model_config["csv_template"]
    json_template = model_config["json_template"]
    num_groups = model_config["num_groups"]

    all_dfs, all_matchings = [], []
    for i in range(1, num_groups + 1):
        # --- CSV 加载 (使用之前的手动解析逻辑) ---
        csv_path = os.path.join(base_path, csv_template.format(group_id=i))
        group_data = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, quotechar='"', delimiter=',',
                                    quoting=csv.QUOTE_ALL, skipinitialspace=True)
                for row_idx, row in enumerate(reader):
                    if len(row) >= 4: # 至少需要Prompt, Reason, 和最后4个我们关心的值
                        try:
                            # 假设你的CSV固定是 Prompt, Reason, target, proposer, current_partner, result
                            # 我们需要最后4个值
                            result = int(row[-1])
                            current_partner = row[-2] if row[-2] else None
                            proposer = int(row[-3])
                            target = int(row[-4])
                            prompt_text = row[0] # 保存Prompt文本
                            reason_text = row[1] # 保存Reason文本
                            
                            group_data.append({
                                'prompt': prompt_text,
                                'reason': reason_text,
                                'target': target,
                                'proposer': proposer,
                                'current_partner': current_partner,
                                'result': result
                            })
                        except (ValueError, IndexError) as parse_err:
                            # print(f"警告: 跳过CSV文件 {csv_path} 中的格式错误行 {row_idx+1}: {row} - {parse_err}")
                            pass
        except FileNotFoundError:
            print(f"警告: CSV文件不存在 {csv_path}")
        except Exception as e:
            print(f"处理CSV文件 {csv_path} 时发生未知错误: {e}")
        
        if group_data:
            df = pd.DataFrame(group_data)
            df['group'] = i
            all_dfs.append(df)
            
        # --- JSON 加载 (使用之前的正则修复版) ---
        json_path = os.path.join(base_path, json_template.format(group_id=i))
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()
                if not file_content: continue
                json_objects = re.findall(r'\{.*?\}', file_content)
                if not json_objects: continue
                if len(json_objects) == 1:
                    all_matchings.append(json.loads(json_objects[0]))
                else:
                    json_array_string = f"[{','.join(json_objects)}]"
                    data_list = json.loads(json_array_string)
                    if data_list: all_matchings.append(data_list[-1])
        except FileNotFoundError:
            # print(f"警告: JSON文件不存在 {json_path}")
            pass
        except Exception as e:
            # print(f"处理JSON文件 {json_path} 时发生错误: {e}")
            pass
            
    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return combined_df, all_matchings