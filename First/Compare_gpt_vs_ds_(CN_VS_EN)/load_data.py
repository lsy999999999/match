# 文件名: load_data.py

import pandas as pd
import json
import os
import re
import csv # 使用Python内置的csv模块进行更可靠的CSV解析
from config import config # 假设你的config.py与此文件在同一目录或Python路径中

def load_all_group_data_for_model(model_key):
    """
    根据模型键加载所有组的CSV和JSON数据。
    CSV文件被假定为特定格式：
    第1列: Prompt文本 (可能包含逗号，被双引号包围)
    第2列: 模型回答文本 (Reason) (可能包含逗号，被双引号包围)
    第3列: best_choice (被求偶者ID)
    第4列: proposer (求偶者ID)
    第5列: current_partner (现任伴侣ID，如果单身则为空)
    第6列: result (数字决策结果, 1=accept, 0=reject)

    JSON文件包含最终的配对结果字典。
    """
    if model_key not in config:
        print(f"错误: 模型键 '{model_key}' 在 config.py 中未找到。")
        # 返回空的DataFrame和列表，让调用方可以处理
        return pd.DataFrame(columns=['prompt', 'reason', 'target', 'proposer', 'current_partner', 'result', 'group']), []

    model_config = config[model_key]
    base_path = model_config.get("base_path", "")
    csv_template = model_config.get("csv_template", "")
    json_template = model_config.get("json_template", "")
    num_groups = model_config.get("num_groups", 0)
    model_label = model_config.get("label", model_key)

    if not base_path or not csv_template or not json_template or num_groups == 0:
        print(f"警告: 模型 '{model_label}' ({model_key}) 的配置不完整 (路径、模板或组数缺失)。跳过加载。")
        return pd.DataFrame(columns=['prompt', 'reason', 'target', 'proposer', 'current_partner', 'result', 'group']), []

    all_dfs_data = [] # 用于收集所有成功解析的CSV行数据字典
    all_matchings = []
    
    print(f"--- 正在加载模型: {model_label} (共 {num_groups} 组) ---")

    for i in range(1, num_groups + 1):
        # --- CSV 加载与解析 ---
        csv_path = os.path.join(base_path, csv_template.format(group_id=i))
        try:
            with open(csv_path, 'r', encoding='utf-8') as f_csv:
                # 使用csv.reader处理带引号和内部逗号的字段
                # quotechar='"' : 指定双引号为包围字符
                # delimiter=',' : 指定逗号为分隔符
                # quoting=csv.QUOTE_MINIMAL : 只在必要时引用字段（例如字段内有分隔符或引号）
                #   如果你的CSV是所有文本字段都用引号包围，即使没有特殊字符，
                #   用csv.QUOTE_ALL 也可以，但QUOTE_MINIMAL更通用。
                #   根据你的CSV示例，你的文本字段总是被引起来的，所以QUOTE_ALL可能更精确。
                # skipinitialspace=True : 忽略分隔符后的空格
                reader = csv.reader(f_csv, quotechar='"', delimiter=',', 
                                    quoting=csv.QUOTE_ALL, skipinitialspace=True)
                rows_in_group = 0
                for row_num, row in enumerate(reader):
                    if len(row) == 6: # 严格要求6列
                        try:
                            prompt_text = str(row[0])
                            reason_text = str(row[1])
                            target_val = int(row[2])
                            proposer_val = int(row[3])
                            # current_partner 可能为空字符串，需特殊处理
                            current_partner_str = str(row[4])
                            current_partner_val = int(current_partner_str) if current_partner_str else None
                            result_val = int(row[5])
                            
                            all_dfs_data.append({
                                'prompt': prompt_text,
                                'reason': reason_text,
                                'target': target_val,
                                'proposer': proposer_val,
                                'current_partner': current_partner_val,
                                'result': result_val,
                                'group': i 
                            })
                            rows_in_group += 1
                        except (ValueError, IndexError) as e_row:
                            print(f"  警告: 跳过CSV文件 {csv_path} 第 {row_num+1} 行，因数据转换错误: {row} -> {e_row}")
                    # else:
                        # 如果行不等于6列，可以打印警告，但为了减少输出，暂时注释
                        # print(f"  警告: 跳过CSV文件 {csv_path} 第 {row_num+1} 行，因列数不为6: {len(row)}列, 内容: {str(row)[:100]}...")
                if rows_in_group == 0 and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                     print(f"  警告: CSV文件 {csv_path} 有内容但未解析出任何有效行。请检查CSV格式。")

        except FileNotFoundError:
            # print(f"  提示: CSV文件不存在 {csv_path} (组 {i})") # 正常现象，如果组数设多了
            pass
        except Exception as e_csv:
            print(f"  错误: 处理CSV文件 {csv_path} 时发生未知错误: {e_csv}")
        
        # --- JSON 加载与解析 ---
        json_path = os.path.join(base_path, json_template.format(group_id=i))
        try:
            with open(json_path, 'r', encoding='utf-8') as f_json:
                file_content = f_json.read().strip()
                if not file_content: 
                    # print(f"  提示: JSON文件为空 {json_path} (组 {i})")
                    continue
                
                # 正则表达式查找所有 {...} 结构
                json_objects = re.findall(r'\{[^{}]*\}', file_content) # 改进正则以处理简单嵌套
                
                if not json_objects:
                    # print(f"  警告: 在 {json_path} 中未找到有效的JSON对象结构。尝试作为单个对象加载。")
                    # 尝试将整个文件内容作为单个JSON对象加载
                    try:
                        data = json.loads(file_content)
                        all_matchings.append(data)
                    except json.JSONDecodeError:
                        print(f"  警告: 尝试将 {json_path} 作为单个JSON对象加载失败。")
                    continue # 跳到下一个文件

                parsed_data_for_group = None
                if len(json_objects) == 1:
                    try:
                        parsed_data_for_group = json.loads(json_objects[0])
                    except json.JSONDecodeError:
                        print(f"  警告: 单个JSON对象解析失败: {json_path}")
                else: # 找到多个对象
                    try:
                        # 将找到的所有JSON字符串用逗号连接，并用方括号包围
                        json_array_string = f"[{','.join(json_objects)}]"
                        data_list = json.loads(json_array_string)
                        if data_list:
                            parsed_data_for_group = data_list[-1] # 取最后一个作为该组的结果
                    except json.JSONDecodeError:
                        print(f"  警告: 修复后的多对象JSON解析失败: {json_path}")
                
                if parsed_data_for_group is not None:
                    all_matchings.append(parsed_data_for_group)

        except FileNotFoundError:
            # print(f"  提示: JSON文件不存在 {json_path} (组 {i})")
            pass
        except Exception as e_json:
            print(f"  错误: 处理JSON文件 {json_path} 时发生未知错误: {e_json}")
            
    # 合并所有组的DataFrame
    if all_dfs_data:
        combined_df = pd.DataFrame(all_dfs_data)
    else:
        # 如果没有加载到任何CSV数据，返回一个有正确列名的空DataFrame
        combined_df = pd.DataFrame(columns=['prompt', 'reason', 'target', 'proposer', 'current_partner', 'result', 'group'])
        print(f"注意: 模型 '{model_label}' 没有加载到任何有效的CSV数据。")

    if not all_matchings and num_groups > 0 : # 仅当预期有数据时才警告
        print(f"注意: 模型 '{model_label}' 没有加载到任何有效的JSON数据。")
        
    print(f"--- 模型 {model_label}: 加载完成。CSV行数: {len(combined_df)}, JSON组数: {len(all_matchings)} ---")
    
    
    # 【新增调试代码】
    if model_key == "deepseek_zh" and not combined_df.empty:
        print(f"--- 调试: DeepSeek (中文) DataFrame ({model_key}) ---")
        print("  current_partner 列的唯一值和计数:")
        print(combined_df['current_partner'].value_counts(dropna=False))
        print("  current_partner 列为空的行数 (isna()):")
        print(combined_df['current_partner'].isna().sum())
        print("  current_partner 列为None的行数 (is None):") # Python的None
        print((combined_df['current_partner'] == None).sum())
        print("  current_partner 列为空字符串的行数 (==''):")
        print((combined_df['current_partner'] == '').sum())
        print("  前5行包含 current_partner 的数据:")
        print(combined_df[['group', 'proposer', 'target', 'current_partner', 'result']].head())
    
    
    
    return combined_df, all_matchings

if __name__ == '__main__':
    # 测试加载函数
    print("开始测试数据加载功能...")
    
    # 假设config.py已配置好 'gpt4_en', 'gpt4_zh', 'deepseek_en', 'deepseek_zh'
    model_keys_to_test = ["gpt4_en", "gpt4_zh", "deepseek_en", "deepseek_zh"]
    
    for model_key in model_keys_to_test:
        if model_key in config:
            print(f"\n测试加载: {config[model_key].get('label', model_key)}")
            df, matchings = load_all_group_data_for_model(model_key)
            print(f"  加载的DataFrame形状: {df.shape}")
            if not df.empty:
                print("  DataFrame 前5行:")
                print(df.head())
                print("  DataFrame 列信息:")
                df.info()
            print(f"  加载的Matchings数量: {len(matchings)}")
            if matchings:
                print(f"  第一个Matching示例: {str(matchings[0])[:200]}...")
        else:
            print(f"\n测试加载: 模型键 '{model_key}' 在config.py中未配置，跳过测试。")
    
    print("\n数据加载功能测试结束。")