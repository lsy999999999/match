# 修正后的 gale-shapely-Chinese.py
from read_file_Chinese import *  # 修正：使用中文版本的读取文件
from ask_gpt_Chinese_random import *  # 修正：使用中文版本的GPT询问函数
import csv
import json
import datetime
import numpy as np
import random

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def gale_shapley(men_preferences, women_preferences, men_attr, women_attr, idx):
    men = list(men_preferences.keys())
    women = list(women_preferences.keys())
    single_list = men + women

    # 初始化matching和single_list
    matching = {person: None for person in single_list}
    rejected_list = []
    proposer_indices = {person: 0 for person in men + women}
    logtext = []
    print(f"Group {idx} - 总人数: {len(single_list)}")

    while single_list:
        proposer = random.choice(single_list)

        if proposer in men:
            attr = men_attr[proposer]
            preferences = men_preferences[proposer]
            is_man = True
        else:
            attr = women_attr[proposer]
            preferences = women_preferences[proposer]
            is_man = False
        
        index = proposer_indices[proposer]
        
        if index >= len(preferences):
            matching[proposer] = 'rejected'
            rejected_list.append(proposer)
            single_list.remove(proposer)
            continue
        
        best_choice = preferences[index]
        proposer_indices[proposer] += 1

        if matching[best_choice] is None or matching[best_choice] == 'rejected':
            handle_single_choice(proposer, best_choice, matching, logtext, attr, single_list, is_man)
        else:
            handle_existing_choice(proposer, best_choice, matching, logtext, attr, men_attr, women_attr, single_list, is_man)

    # 确保输出路径和文件名正确
    csv_path = f"/home/lsy/match/bahavior_simul/0629_gpt_Chinese/0629_gpt4_Chinese_group{idx}.csv"
    json_path = f"/home/lsy/match/bahavior_simul/0629_gpt_Chinese/0629_gpt4_Chinese_group{idx}.json"
    
    log_text(csv_path, logtext)
    
    js_obj = json.dumps(matching, ensure_ascii=False, default=default_dump)
    with open(json_path, "w", encoding='utf-8') as file:  # 修正：使用 'w' 而不是 'a'，并指定编码
        file.write(js_obj)
    
    print(f"Group {idx} 完成 - CSV: {csv_path}")
    return matching

def initialize_matching(men, women):
    men_current = {man: None for man in men}
    women_current = {woman: None for woman in women}
    return men_current, women_current

def handle_single_choice(proposer, best_choice, matching, logtext, attr, single_list, is_man):
    proposer_scores = attr[best_choice]
    
    score_details = {
        'age': proposer_scores[0],
        'age_o': proposer_scores[1],
        'career': proposer_scores[2],
        'overall_score': proposer_scores[3],
        'attractive': proposer_scores[4],
        'sincere': proposer_scores[5],
        'intelligence': proposer_scores[6],
        'funny': proposer_scores[7],
        'ambition': proposer_scores[8],
        'shared_interests': proposer_scores[9],
        'attractive_importance': proposer_scores[10],
        'sincere_importance': proposer_scores[11],
        'intelligence_importance': proposer_scores[12],
        'funny_importance': proposer_scores[13],
        'ambition_importance': proposer_scores[14],
        'shared_interests_importance': proposer_scores[15]
    }
    
    if is_man:
        flag, log = ask_gpt_single_man_propose(score_details)
    else:
        flag, log = ask_gpt_single_woman_propose(score_details)
    
    log.append(best_choice)
    log.append(proposer)
    log.append("")

    if flag == True:
        matching[proposer] = best_choice
        matching[best_choice] = proposer
        log.append(1)
        
        single_list.remove(proposer)
        try:
            single_list.remove(best_choice)
        except:
            pass
    else:
        log.append(0)
    
    logtext.append(log)

def handle_existing_choice(proposer, best_choice, matching, logtext, attr, men_attr, women_attr, single_list, is_man):
    current_partner = matching[best_choice]
    proposer_scores = attr[best_choice]
    
    proposer_score_details = {
        'age': proposer_scores[0],
        'age_o': proposer_scores[1],
        'career': proposer_scores[2],
        'overall_score': proposer_scores[3],
        'attractive': proposer_scores[4],
        'sincere': proposer_scores[5],
        'intelligence': proposer_scores[6],
        'funny': proposer_scores[7],
        'ambition': proposer_scores[8],
        'shared_interests': proposer_scores[9],
        'attractive_importance': proposer_scores[10],
        'sincere_importance': proposer_scores[11],
        'intelligence_importance': proposer_scores[12],
        'funny_importance': proposer_scores[13],
        'ambition_importance': proposer_scores[14],
        'shared_interests_importance': proposer_scores[15]
    }
    
    if is_man:
        current_partner_scores = women_attr[best_choice][current_partner]
    else:
        current_partner_scores = men_attr[best_choice][current_partner]
    
    current_partner_score_details = {
        'age': current_partner_scores[0],
        'age_o': current_partner_scores[1],
        'career': current_partner_scores[2],
        'overall_score': current_partner_scores[3],
        'attractive': current_partner_scores[4],
        'sincere': current_partner_scores[5],
        'intelligence': current_partner_scores[6],
        'funny': current_partner_scores[7],
        'ambition': current_partner_scores[8],
        'shared_interests': current_partner_scores[9],
        'attractive_importance': current_partner_scores[10],
        'sincere_importance': current_partner_scores[11],
        'intelligence_importance': current_partner_scores[12],
        'funny_importance': current_partner_scores[13],
        'ambition_importance': current_partner_scores[14],
        'shared_interests_importance': current_partner_scores[15]
    }
    
    if is_man:
        flag, log = ask_gpt_non_single_man_propose(proposer_score_details, current_partner_score_details)
    else:
        flag, log = ask_gpt_non_single_woman_propose(proposer_score_details, current_partner_score_details)

    log.append(best_choice)
    log.append(proposer)
    log.append(current_partner)
    
    if flag == True:
        matching[current_partner] = None
        single_list.append(current_partner)
        
        matching[proposer] = best_choice
        matching[best_choice] = proposer
        log.append(1)
        
        single_list.remove(proposer)
    else:
        log.append(0)
    
    logtext.append(log)

def log_text(logfile, logtext):
    # 使用 utf-8 编码确保中文正确保存
    with open(logfile, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for log in logtext:
            writer.writerow(log)

if __name__=='__main__':
    # 创建输出目录（如果不存在）
    import os
    output_dir = "/home/lsy/match/bahavior_simul/0629_gpt_Chinese/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理组 1-21
    for i in range(1, 22):
        print(f"\n{'='*50}")
        print(f"处理组 {i}")
        print(f"{'='*50}")
        
        try:
            # 读取数据
            l1, l2 = read_file(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', i)
            dict1, dict2 = read_file_attr(r'/home/lsy/match/dataset/save_merge_select_null_3.xlsx', i)
            
            print(f"男性数量: {len(l1)}, 女性数量: {len(l2)}")
            
            # 运行算法
            matching = gale_shapley(l1, l2, dict1, dict2, i)
            
            print(f"组 {i} 完成！")
            
        except Exception as e:
            print(f"组 {i} 出错: {e}")
            import traceback
            traceback.print_exc()