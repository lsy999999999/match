
from read_file_Chinese import read_file, read_file_attr
from ask_qw_Chinese import (
    ask_gpt_single_man_propose,
    ask_gpt_single_woman_propose,
    ask_gpt_non_single_man_propose,
    ask_gpt_non_single_woman_propose
)
import csv
import json
import os
import numpy as np
import random

# --------------------------------------------------------------------------
#  辅助函数和核心逻辑函数 (与我们之前讨论的最终版完全一致)
# --------------------------------------------------------------------------
def default_dump(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def create_score_details_dict(score_list):
    if len(score_list) < 16: return None
    return {
        'age': score_list[0], 'age_o': score_list[1], 'career': score_list[2],
        'overall_score': score_list[3], 'attractive': score_list[4], 'sincere': score_list[5],
        'intelligence': score_list[6], 'funny': score_list[7], 'ambition': score_list[8],
        'shared_interests': score_list[9], 'attractive_importance': score_list[10],
        'sincere_importance': score_list[11], 'intelligence_importance': score_list[12],
        'funny_importance': score_list[13], 'ambition_importance': score_list[14],
        'shared_interests_importance': score_list[15]
    }

def handle_single_choice(proposer, best_choice, matching, logtext, men_attr, women_attr, single_list):
    new_proposer, new_target = best_choice, proposer
    new_proposer_is_man = new_proposer in men_attr
    print(f"  [Counterfactual-Single] Original: {proposer} -> {best_choice}. Simulating: {new_proposer} -> {new_target}")
    attr = men_attr[new_proposer] if new_proposer_is_man else women_attr[new_proposer]
    if new_target not in attr:
        print(f"  [Warning] No rating data for {new_proposer} on {new_target}. Skipping.")
        return
    score_details = create_score_details_dict(attr[new_target])
    if score_details is None: return
    flag, log = ask_gpt_single_woman_propose(score_details) if new_proposer_is_man else ask_gpt_single_man_propose(score_details)
    log.extend([new_target, new_proposer, ""])
    if flag:
        matching[new_proposer], matching[new_target] = new_target, new_proposer
        log.append(1)
        if new_proposer in single_list: single_list.remove(new_proposer)
        if new_target in single_list: single_list.remove(new_target)
    else:
        log.append(0)
    logtext.append(log)

def handle_existing_choice(proposer, best_choice, matching, logtext, men_attr, women_attr, single_list):
    current_partner = matching[best_choice]
    new_proposer, new_current_partner = current_partner, proposer
    print(f"  [Counterfactual-Taken] Original: {proposer} -> {best_choice} (current: {current_partner}). Simulating: {new_proposer} -> {best_choice} (current: {new_current_partner})")
    decision_maker_is_man = best_choice in men_attr
    attr_new_proposer = men_attr[new_proposer] if new_proposer in men_attr else women_attr[new_proposer]
    if best_choice not in attr_new_proposer:
        print(f"  [Warning] No rating data for new proposer {new_proposer} on {best_choice}. Skipping.")
        return
    proposer_score_details = create_score_details_dict(attr_new_proposer[best_choice])
    attr_new_current = men_attr[new_current_partner] if new_current_partner in men_attr else women_attr[new_current_partner]
    if best_choice not in attr_new_current:
        print(f"  [Warning] No rating data for new current partner {new_current_partner} on {best_choice}. Skipping.")
        return
    current_partner_score_details = create_score_details_dict(attr_new_current[best_choice])
    if proposer_score_details is None or current_partner_score_details is None: return
    flag, log = ask_gpt_non_single_woman_propose(proposer_score_details, current_partner_score_details) if decision_maker_is_man else ask_gpt_non_single_man_propose(proposer_score_details, current_partner_score_details)
    log.extend([best_choice, new_proposer, new_current_partner])
    if flag:
        matching[new_current_partner] = None
        if new_current_partner not in single_list: single_list.append(new_current_partner)
        matching[new_proposer], matching[best_choice] = best_choice, new_proposer
        log.append(1)
        if new_proposer in single_list: single_list.remove(new_proposer)
    else:
        log.append(0)
    logtext.append(log)

def gale_shapley(men_preferences, women_preferences, men_attr, women_attr, output_idx):
    men, women = list(men_preferences.keys()), list(women_preferences.keys())
    single_list = men + women
    matching = {person: None for person in single_list}
    proposer_indices = {person: 0 for person in men + women}
    logtext = []
    
    # 使用经典的 while 循环，跑完整个算法
    while single_list:
        proposer = random.choice(single_list)
        is_man = proposer in men
        preferences = men_preferences.get(proposer) if is_man else women_preferences.get(proposer)
        index = proposer_indices.get(proposer, 0)
        
        if not preferences or index >= len(preferences):
            if proposer in single_list: single_list.remove(proposer)
            matching[proposer] = 'rejected'
            continue
            
        best_choice = preferences[index]
        proposer_indices[proposer] = index + 1
        
        if matching.get(best_choice) is None or str(matching.get(best_choice)).lower() == 'rejected':
            handle_single_choice(proposer, best_choice, matching, logtext, men_attr, women_attr, single_list)
        else:
            handle_existing_choice(proposer, best_choice, matching, logtext, men_attr, women_attr, single_list)
    
    # --- 文件保存 ---
    output_path = "/home/lsy/match/qwen/0707_qw_Chinese"
    os.makedirs(output_path, exist_ok=True)
    csv_filename = os.path.join(output_path, f"0707_qw_Chinese_group{output_idx}.csv")
    json_filename = os.path.join(output_path, f"0707_qw_Chinese_group{output_idx}.json")

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(logtext)
    with open(json_filename, "a", encoding='utf-8') as file:
        json.dump(matching, file, ensure_ascii=False, default=default_dump)
        
    return matching


# --------------------------------------------------------------------------
#  【核心修改】主程序入口 (if __name__ == '__main__':)
# --------------------------------------------------------------------------
if __name__ == '__main__':
  # 定义输入组号和输出组号的起始点
  input_start_group = 2
  output_start_group = 23
  num_simulations = 20 # 我们要进行21次模拟
  

  # 使用一个循环来控制21次模拟
  for i in range(num_simulations):
    # 计算当前的输入组号和输出组号
    current_input_group = input_start_group + i
    current_output_group = output_start_group + i

    print(f"\n=====================================================================")
    print(f"--- Running Counterfactual Sim #{i+1}: Input Group {current_input_group} -> Output Group {current_output_group} ---")
    print(f"=====================================================================")
    
    # 使用输入组号加载数据
    try:
        l1, l2 = read_file(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', current_input_group)
        dict1, dict2 = read_file_attr(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', current_input_group)
    except Exception as e:
        print(f"Error loading data for input group {current_input_group}: {e}")
        continue # 如果加载失败，跳到下一次模拟

    # 检查加载的数据是否完整
    if not l1 or not dict1 or not l2 or not dict2:
        print(f"Warning: Incomplete data loaded for input group {current_input_group}. Skipping.")
        continue
        
    # 调用gale_shapley函数，并传入输出组号用于保存文件
    gale_shapley(l1, l2, dict1, dict2, current_output_group)

  print("\nAll counterfactual simulations completed!")