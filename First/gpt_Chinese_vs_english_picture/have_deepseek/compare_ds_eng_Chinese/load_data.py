import pandas as pd
import json
import os
import re
import csv # 引入Python内置的csv模块
from config import config

def load_all_group_data(lang):
    """根据语言加载所有组的数据，手动解析非标准CSV文件。"""
    if lang == 'en':
        base_path = config["base_path_en"]
        csv_template = config["csv_template_en"]
        json_template = config["json_template_en"]
        num_groups = config["num_groups_en"]
    elif lang == 'zh':
        base_path = config["base_path_zh"]
        csv_template = config["csv_template_zh"]
        json_template = config["json_template_zh"]
        num_groups = config["num_groups_zh"]
    else:
        raise ValueError("语言参数必须是 'en' 或 'zh'")

    all_dfs, all_matchings = [], []
    for i in range(1, num_groups + 1):
        # --- CSV 加载部分 (手动解析) ---
        csv_path = os.path.join(base_path, csv_template.format(group_id=i))
        group_data = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                # 使用Python内置的csv reader，它能更好地处理带引号的字段
                reader = csv.reader(f, quotechar='"', delimiter=',',
                                    quoting=csv.QUOTE_ALL, skipinitialspace=True)
                for row in reader:
                    # 我们期望每行解析后至少有4个元素（最后4个ID/结果）
                    if len(row) >= 4:
                        # 倒数第4列: target
                        # 倒数第3列: proposer
                        # 倒数第2列: current_partner
                        # 倒数第1列: result
                        # 我们可以只提取我们需要的部分，忽略前面复杂的文本列
                        # 或者，如果需要文本，可以这样提取：
                        prompt = row[0]
                        reason = row[1]
                        # 剩余部分是我们的ID和结果
                        rest_of_row = row[2:]
                        
                        # 找到我们关心的最后4个值
                        # 注意：CSV解析后，所有值都是字符串
                        try:
                            result = int(rest_of_row[-1])
                            current_partner = rest_of_row[-2] if rest_of_row[-2] else None # 空字符串转为None
                            proposer = int(rest_of_row[-3])
                            target = int(rest_of_row[-4])
                            
                            group_data.append({
                                'prompt': prompt,
                                'reason': reason,
                                'target': target,
                                'proposer': proposer,
                                'current_partner': current_partner,
                                'result': result
                            })
                        except (ValueError, IndexError):
                             # 如果最后几列无法转成数字或索引不存在，说明这行格式有问题，跳过
                            # print(f"警告: 跳过CSV文件 {csv_path} 中的格式错误行: {row}")
                            pass

        except FileNotFoundError:
            print(f"警告: CSV文件不存在 {csv_path}")
        except Exception as e:
            print(f"处理CSV文件时发生未知错误 {csv_path}: {e}")
        
        if group_data:
            df = pd.DataFrame(group_data)
            df['group'] = i
            all_dfs.append(df)
            
        # --- JSON 加载部分 (使用之前的究极修复版) ---
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
        except (FileNotFoundError, json.JSONDecodeError):
            # print(f"警告: JSON文件加载失败 {json_path}")
            pass # 减少不必要的警告输出
            
    # 合并所有组的DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return combined_df, all_matchings