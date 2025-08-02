# 文件名: run_parallel_simulations.py

import pandas as pd
import numpy as np
import os
import csv
import json
import random
import traceback

# 导入你的API调用函数和数据读取函数
from read_file_random import read_file, read_file_attr
from ask_grok_eng import (
    ask_gpt_single_man_propose,
    ask_gpt_single_woman_propose,
    ask_gpt_non_single_man_propose,
    ask_gpt_non_single_woman_propose
)

# --- 辅助函数 ---
def default_dump(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
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

# --- 核心逻辑函数 ---

# 1. 原始场景的决策处理函数
def handle_single_choice_original(proposer, best_choice, matching, logtext, men_attr, women_attr, single_list, is_man):
    attr = men_attr.get(proposer, {})
    if best_choice not in attr: return
    score_details = create_score_details_dict(attr[best_choice])
    if score_details is None: return
    
    # 决策者是被求偶的 best_choice
    decision_maker_is_man = best_choice in men_attr
    flag, log = ask_gpt_single_man_propose(score_details) if decision_maker_is_man else ask_gpt_single_woman_propose(score_details)
    
    log.extend([best_choice, proposer, "", 1 if flag else 0])
    logtext.append(log)

    if flag:
        matching[proposer], matching[best_choice] = best_choice, proposer
        if proposer in single_list: single_list.remove(proposer)
        if best_choice in single_list: single_list.remove(best_choice)

def handle_existing_choice_original(proposer, best_choice, matching, logtext, men_attr, women_attr, single_list, is_man):
    current_partner = matching[best_choice]
    decision_maker_is_man = best_choice in men_attr
    
    attr_proposer = men_attr.get(proposer, {}) if is_man else women_attr.get(proposer, {})
    score_details_proposer = create_score_details_dict(attr_proposer.get(best_choice, []))
    
    attr_current = men_attr.get(current_partner, {}) if not is_man else women_attr.get(current_partner, {})
    score_details_current = create_score_details_dict(attr_current.get(best_choice, []))

    if score_details_proposer is None or score_details_current is None: return
        
    flag, log = ask_gpt_non_single_woman_propose(score_details_proposer, score_details_current) if decision_maker_is_man else ask_gpt_non_single_man_propose(score_details_proposer, score_details_current)
    
    log.extend([best_choice, proposer, current_partner, 1 if flag else 0])
    logtext.append(log)

    if flag:
        matching[current_partner] = None
        if current_partner not in single_list: single_list.append(current_partner)
        matching[proposer], matching[best_choice] = best_choice, proposer
        if proposer in single_list: single_list.remove(proposer)

# 2. 反事实场景的决策处理函数
def handle_single_choice_counterfactual(proposer, best_choice, matching, logtext, men_attr, women_attr, single_list, is_man):
    new_proposer, new_target = best_choice, proposer
    new_proposer_is_man = new_proposer in men_attr
    
    attr = men_attr.get(new_proposer, {}) if new_proposer_is_man else women_attr.get(new_proposer, {})
    if new_target not in attr: return
    
    score_details = create_score_details_dict(attr[new_target])
    if score_details is None: return
        
    decision_maker_is_man = new_target in men_attr
    flag, log = ask_gpt_single_man_propose(score_details) if decision_maker_is_man else ask_gpt_single_woman_propose(score_details)
    
    log.extend([new_target, new_proposer, "", 1 if flag else 0])
    logtext.append(log)
    
    if flag:
        matching[new_proposer], matching[new_target] = new_target, new_proposer
        if new_proposer in single_list: single_list.remove(new_proposer)
        if new_target in single_list: single_list.remove(new_target)

def handle_existing_choice_counterfactual(proposer, best_choice, matching, logtext, men_attr, women_attr, single_list, is_man):
    current_partner = matching[best_choice]
    new_proposer, new_current_partner = current_partner, proposer
    decision_maker_is_man = best_choice in men_attr

    attr_new_proposer = men_attr.get(new_proposer, {}) if new_proposer in men_attr else women_attr.get(new_proposer, {})
    if best_choice not in attr_new_proposer: return
    proposer_score_details = create_score_details_dict(attr_new_proposer[best_choice])

    attr_new_current = men_attr.get(new_current_partner, {}) if new_current_partner in men_attr else women_attr.get(new_current_partner, {})
    if best_choice not in attr_new_current: return
    current_partner_score_details = create_score_details_dict(attr_new_current[best_choice])

    if proposer_score_details is None or current_partner_score_details is None: return
    
    flag, log = ask_gpt_non_single_woman_propose(proposer_score_details, current_partner_score_details) if decision_maker_is_man else ask_gpt_non_single_man_propose(proposer_score_details, current_partner_score_details)
    
    log.extend([best_choice, new_proposer, new_current_partner, 1 if flag else 0])
    logtext.append(log)
    
    if flag:
        matching[new_current_partner] = None
        if new_current_partner not in single_list: single_list.append(new_current_partner)
        matching[new_proposer], matching[best_choice] = best_choice, new_proposer
        if new_proposer in single_list: single_list.remove(new_proposer)

# 3. 通用的Gale-Shapley运行器
def run_gale_shapley_simulation(men_preferences, women_preferences, men_attr, women_attr, output_idx, is_counterfactual):
    men, women = list(men_preferences.keys()), list(women_preferences.keys())
    single_list = men + women
    matching = {person: None for person in single_list}
    proposer_indices = {person: 0 for person in single_list}
    logtext = []
    sim_type = "Counterfactual" if is_counterfactual else "Original"
    print(f"--- Starting {sim_type} Simulation for Output Group {output_idx} ---")
    
    while single_list:
        proposer = random.choice(single_list)
        is_man = proposer in men
        preferences = men_preferences.get(proposer, [])
        index = proposer_indices.get(proposer, 0)
        
        if not preferences or index >= len(preferences):
            if proposer in single_list: single_list.remove(proposer)
            matching[proposer] = 'rejected'
            continue
            
        best_choice = preferences[index]
        proposer_indices[proposer] = index + 1
        
        if matching.get(best_choice) is None or str(matching.get(best_choice)).lower() == 'rejected':
            if is_counterfactual:
                handle_single_choice_counterfactual(proposer, best_choice, matching, logtext, men_attr, women_attr, single_list, is_man)
            else:
                handle_single_choice_original(proposer, best_choice, matching, logtext, men_attr, women_attr, single_list, is_man)
        else:
            if is_counterfactual:
                handle_existing_choice_counterfactual(proposer, best_choice, matching, logtext, men_attr, women_attr, single_list, is_man)
            else:
                handle_existing_choice_original(proposer, best_choice, matching, logtext, men_attr, women_attr, single_list, is_man)

    # --- 文件保存 ---
    output_path = "/home/lsy/match/grok/0725_grok_eng/"
    os.makedirs(output_path, exist_ok=True)
    csv_filename = os.path.join(output_path, f"0725_grok_eng_group{output_idx}.csv")
    json_filename = os.path.join(output_path, f"0725_grok_eng_group{output_idx}.json")

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(logtext)
    with open(json_filename, "w", encoding='utf-8') as file:
        json.dump(matching, file, ensure_ascii=False, default=default_dump)
    print(f"--- {sim_type} Simulation for Output Group {output_idx} complete. Files saved. ---")

# --- 主程序入口 ---
if __name__ == '__main__':
  # 定义输入和输出的映射关系
  input_groups = range(1, 2)  # 1 到 21
  output_cf_start_group = 22 # 反事实输出从22开始

  for input_group in input_groups:
    output_cf_group = input_group + (output_cf_start_group - 1)
    
    print(f"\n========================================================")
    print(f"--- Processing Input Group {input_group} ---")
    print(f"========================================================")
    
    try:
        # 【关键】分别加载两种类型的数据
        men_prefs, women_prefs = read_file(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', input_group)
        men_attributes, women_attributes = read_file_attr(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', input_group)
        
        if not all([men_prefs, women_prefs, men_attributes, women_attributes]):
            print(f"Warning: Incomplete data for input group {input_group}. Skipping.")
            continue
            
        # 1. 运行原始模拟, 输出组号与输入组号相同 (1-21)
        run_gale_shapley_simulation(men_prefs, women_prefs, men_attributes, women_attributes, output_idx=input_group, is_counterfactual=False)

        # 2. 运行反事实模拟, 输出组号是偏移后的 (22-42)
        run_gale_shapley_simulation(men_prefs, women_prefs, men_attributes, women_attributes, output_idx=output_cf_group, is_counterfactual=True)

    except Exception as e:
        print(f"An unexpected error occurred while processing group {input_group}:")
        traceback.print_exc()
        continue

  print("\nAll simulations (original and counterfactual) completed!")